import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
from math import ceil
from PIL import Image
from image_captioning import pil_loader, Decoder, Encoder, data_transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_caption_visualization(encoder, decoder, img_path, word_dict, beam_size=3, smooth=True):
    img=pil_loader(img_path)
    img=data_transforms(img)
    img=torch.FloatTensor(img).to(device)
    img=img.unsqueeze(0)
    img_features=encoder(img)
    img_features=img_features.expand(beam_size,img_features.size(1),img_features.size(2))
    sentence,alpha=decoder.caption(img_features,beam_size)
    token_dict={idx: word for word, idx in word_dict.items()}
    sentence_tokens=[]
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx==word_dict['<eos>']:
            break
    img=Image.open(img_path)
    w,h=img.size
    if w>h:
        w=w*256/h
        h=256
    else:
        h=h*256/w
        w=256
    left=(w-224)/2
    top=(h-224)/2
    resized_img=img.resize((int(w),int(h)),Image.BICUBIC).crop((left,top,left+224,top+224))
    img=np.array(resized_img.convert('RGB').getdata()).reshape(224,224,3)
    img=img.astype('float32')/255
    num_words=len(sentence_tokens)
    w=np.round(np.sqrt(num_words))
    h=np.ceil(np.float32(num_words)/w)
    alpha=torch.tensor(alpha)
    plot_height=ceil((num_words+3)/4.0)
    ax1=plt.subplot(4,plot_height,1)
    plt.imshow(img)
    plt.axis('off')
    for idx in range(num_words):
        ax2=plt.subplot(4,plot_height,idx+2)
        label=sentence_tokens[idx]
        plt.text(0,1,label,backgroundcolor='white',fontsize=13)
        plt.text(0,1,label,color='black',fontsize=13)
        plt.imshow(img)
        shape_size=7
        if smooth:
            alpha_img=skimage.transform.pyramid_expand(alpha[idx,:].reshape(shape_size,shape_size),upscale=32,sigma=20)
        else:
            alpha_img=skimage.transform.resize(alpha[idx,:].reshape(shape_size,shape_size),[img.shape[0],img.shape[1]])
        plt.imshow(alpha_img,alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()
    img=pil_loader(img_path)
    plt.imshow(img)
    plt.text(0,0,'caption: '+' '.join(sentence_tokens[1:-1])+'.',fontsize='x-large')
    plt.show()

img_path="/data/coco"
model_path="/data/coco/model_1_train_val.pth"
if __name__=="__main__":
    word_dict=json.load(open('/data/coco/word_dict.json','r'))
    vocabulary_size=len(word_dict)
    encoder=Encoder()
    decoder=Decoder(vocabulary_size,encoder.dim)
    print(vocabulary_size,encoder.dim)
    encoder.to(device)
    decoder.to(device)
    decoder.load_state_dict(torch.load(img_path))
    encoder.eval()
    decoder.eval()
    generate_caption_visualization(encoder,decoder,img_path,word_dict)