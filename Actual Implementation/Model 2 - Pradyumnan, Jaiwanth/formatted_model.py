import torch
import torch.nn as nn
from torchvision import transforms
import json
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x+=identity
        x=self.relu(x)
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
    def forward(self, x):
      identity = x.clone()
      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))
      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),nn.BatchNorm2d(planes*ResBlock.expansion))
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)
    
def ResNet152(channels=3):
    return ResNet(Bottleneck, [3,8,36,3], channels)

 

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageCaptionDataset(Dataset):
    def __init__(self, transform, data_path, split_type='train'):
        super(ImageCaptionDataset, self).__init__()
        self.split_type = split_type
        self.transform = transform
        self.word_count = Counter()
        self.caption_img_idx = {}
        self.img_paths = json.load(open(data_path + '/{}_img_paths.json'.format(split_type), 'r'))
        self.captions = json.load(open(data_path + '/{}_captions.json'.format(split_type), 'r'))
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.split_type == 'train':
            return torch.FloatTensor(img).to(device), torch.tensor(self.captions[index]).to(device)
        matching_idxs = [idx for idx, path in enumerate(self.img_paths) if path == img_path]
        all_captions = [self.captions[idx] for idx in matching_idxs]
        return torch.FloatTensor(img).to(device), torch.tensor(self.captions[index]).to(device), torch.tensor(all_captions).to(device)
    def __len__(self):
        return len(self.captions)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = ResNet152()
        self.dim = 2048
    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x.to(device)
    
class Attention(nn.Module):
    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha
    
class LSTMCell_un(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))  
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)) 
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))           
        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, x_t, h_t, c_t):
        HS = self.hidden_size
        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t = torch.sigmoid(gates[:, :HS])      
        f_t = torch.sigmoid(gates[:, HS:HS*2])  
        g_t = torch.tanh(gates[:, HS*2:HS*3])   
        o_t = torch.sigmoid(gates[:, HS*3:])    
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t 

class Decoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, tf=True):
        super(Decoder, self).__init__()
        self.use_tf = tf
        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim
        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()
        self.f_beta = nn.Linear(512, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.deep_output = nn.Linear(512, vocabulary_size)
        self.dropout = nn.Dropout()
        self.attention = Attention(encoder_dim)
        self.embedding = nn.Embedding(vocabulary_size, 512)
        self.lstm = LSTMCell_un(512 + encoder_dim, 512)

    def forward(self, img_features, captions):
        batch_size = img_features.size(0)
        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1
        prev_words = torch.zeros(batch_size, 1).long().to(device)
        if self.use_tf:
            embedding = self.embedding(captions).to(device) if self.training else self.embedding(prev_words).to(device)
        else:
            embedding = self.embedding(prev_words).to(device)
        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(device)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(device)
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, h, c)
            output = self.deep_output(self.dropout(h))
            preds[:, t] = output
            alphas[:, t] = alpha
            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1)).to(device)
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)
        c = self.init_c(avg_features)
        c = self.tanh(c)
        h = self.init_h(avg_features)
        h = self.tanh(h)
        return h, c

    def caption(self, img_features, beam_size):
        prev_words = torch.zeros(beam_size, 1).long().to(device)
        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1).to(device)
        alphas = torch.ones(beam_size, 1, img_features.size(1)).to(device)
        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []
        step = 1
        h, c = self.get_init_lstm_state(img_features)
        while True:
            embedding = self.embedding(prev_words).squeeze(1).to(device)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output
            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words // output.size(1)
            next_word_idxs = top_words % output.size(1)
            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)
            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))
            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)
            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)
            if step > 50:
                break
            step += 1
        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def calculate_caption_lengths(word_dict, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths
data_transforms = transforms.Compose([transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

def main():
    word_dict = json.load(open('data/coco/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)
    encoder = Encoder().to(device)
    decoder = Decoder(vocabulary_size, encoder.dim, True).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)
    train_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms,'data/coco'),
        batch_size=16, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, 'data/coco', split_type='val'),
        batch_size=16, shuffle=True, num_workers=4)
    print('Starting training ...')
    for epoch in range(1, 1 + 1):
        scheduler.step()
        train(epoch, encoder, decoder, optimizer, cross_entropy_loss,
              train_loader, word_dict, 1, 100)
        model_file = 'data/coco/model_' + str(epoch) +'train'+ '.pth'
        torch.save(decoder.state_dict(), model_file)
        try:
            validate(epoch, encoder, decoder, cross_entropy_loss, val_loader, word_dict, 1, 100)
        except Exception as e:
            print(f" Error during validation: {e}, continuing with the next iteration")
            continue  

        model_file = '/data/coco/model_' + str(epoch) +'train_val'+ '.pth'
        torch.save(decoder.state_dict(), model_file)
        print(f"Saved model to {model_file}")


def train(encoder, decoder, optimizer, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval):
    encoder.eval()
    decoder.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (imgs, captions) in enumerate(data_loader):
        total_caption_length = calculate_caption_lengths(word_dict, captions)
        if total_caption_length!=0:
            imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)
            img_features = encoder(imgs)
            optimizer.zero_grad()
            preds, alphas = decoder(img_features, captions)
            targets = captions[:, 1:]
            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
            att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
            loss = cross_entropy_loss(preds, targets)
            loss += att_regularization
            loss.backward()
            optimizer.step()
            total_caption_length = calculate_caption_lengths(word_dict, captions)
            acc1 = accuracy(preds, targets, 1)
            acc5 = accuracy(preds, targets, 5)
            losses.update(loss.item(), total_caption_length)
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)
        if batch_idx % log_interval == 0:
            print('Train Batch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
        if batch_idx==2000:
            break

    

def validate(epoch,encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval):
    encoder.eval()
    decoder.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    references = []
    hypotheses = []
    with torch.no_grad():
        for batch_idx, (imgs, captions, all_captions) in enumerate(data_loader):
            imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)
            img_features = encoder(imgs)
            preds, alphas = decoder(img_features, captions)
            targets = captions[:, 1:]

            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization=alpha_c*((1-alphas.sum(1))**2).mean()

            loss = cross_entropy_loss(packed_preds, targets)
            loss += att_regularization
            total_caption_length = calculate_caption_lengths(word_dict, captions)
            if total_caption_length!=0:
                acc1 = accuracy(packed_preds, targets, 1)
                acc5 = accuracy(packed_preds, targets, 5)

                losses.update(loss.item(), total_caption_length)
                top1.update(acc1, total_caption_length)
                top5.update(acc5, total_caption_length)

            for cap_set in all_captions.tolist():
                caps = []
                for caption in cap_set:
                    cap = [word_idx for word_idx in caption
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps.append(cap)
                references.append(caps)

            word_idxs = torch.max(preds, dim=2)[1]
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])
           
            if batch_idx % log_interval == 0:
                print('Validation Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(batch_idx, len(data_loader), loss=losses, top1=top1,top5=top5))
            if batch_idx==2000:
                break      
        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        print('Validation Epoch: {}\t'
              'BLEU-1 ({})\t'
              'BLEU-2 ({})\t'
              'BLEU-3 ({})\t'
              'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()