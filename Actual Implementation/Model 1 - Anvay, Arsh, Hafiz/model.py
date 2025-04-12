from torch import nn, Tensor
import torchvision.models as models
import torch.nn as nn

class ImageEncoder(nn.Module):

    def __init__(self,encode_size=16,embed_dim=512):
        super(ImageEncoder,self).__init__()

        self.embed_dim = embed_dim

        # resnet = torchvision.models.resnet101(pretrained=True)
        # resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        
        # self.resnet = nn.Sequential(*list(resnet.children()))[:-2] #removing last linear and pool layer
        swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

        self.swin = nn.Sequential(*list(swin.children())[:-2])
    
        # self.downsampling = nn.Conv2d(in_channels=2048, out_channels = embed_dim,
                                     # kernel_size = 1, stride = 1, bias = False)
        self.downsampling = nn.Conv2d(768, embed_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

        self.adaptive_resize = nn.AdaptiveAvgPool2d(encode_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, encode_size**2, embed_dim))
        self._init_fine_tuning()

    def forward(self, images: Tensor):
        B = images.size()[0] 
        
        out = self.swin(images)

        out = self.relu(self.bn(self.downsampling(out)))

        out = self.adaptive_resize(out)

        out = out.view(B, self.embed_dim,-1).permute(0,2,1)
        out += self.pos_embedding

        return out

    def _init_fine_tuning(self):
        for param in self.swin.parameters():
            param.requires_grad = False
            
        # Unfreeze final blocks progressively
        unfreeze_layers = ['layer4', 'layer3']
        for name, child in self.swin.named_children():
            if name in unfreeze_layers:
                for param in child.parameters():
                    param.requires_grad = True

from torch import nn, Tensor
from torch.nn import MultiheadAttention


class LinearLayer(nn.Module):
    def __init__(self, encode_size: int, embed_dim: int, feedforward_dim: int, dropout: float):
        super(LinearLayer, self).__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=feedforward_dim, out_features=embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    # def forward(self, inputs: Tensor) -> Tensor:
    #     output = self.ffn(inputs.permute(0, 2, 1))  # Change to (B, embed_dim, encode_size)
    #     return self.layer_norm(output.permute(0, 2, 1) + inputs)  # Residual connection
    def forward(self, inputs: Tensor) -> Tensor:
        output = self.ffn(inputs)  # Change to (B, embed_dim, encode_size)
        return self.layer_norm(output + inputs)  # Residual connection


class EncSelfAttention(nn.Module):
    def __init__(self, img_embed_dim: int, num_heads: int, dropout: float):
        super(EncSelfAttention, self).__init__()
        
        self.multi_head_attn = MultiheadAttention(embed_dim=img_embed_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(img_embed_dim)
    
    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs, _ = self.multi_head_attn(
            enc_inputs.transpose(0,1),  
            enc_inputs.transpose(0,1),
            enc_inputs.transpose(0,1)
        )
        enc_outputs = enc_outputs.transpose(0, 1)  # Convert back to (B, encode_size, embed_dim)
        return self.layer_norm(enc_outputs + enc_inputs)  # Residual connection

class EncoderLayer(nn.Module):
    def __init__(self, img_encode_size: int, img_embed_dim: int, feedforward_dim: int, num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()
        
        self.enc_self_attn = EncSelfAttention(img_embed_dim=img_embed_dim, num_heads=num_heads, dropout=dropout)
        self.cnn_ff = LinearLayer(encode_size=img_encode_size, embed_dim=img_embed_dim, feedforward_dim=feedforward_dim, dropout=dropout)
    
    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs = self.enc_self_attn(enc_inputs)  # Pass through attention layer first
        enc_outputs = self.cnn_ff(enc_outputs)  # Then pass through feedforward CNN
        return enc_outputs

if __name__ == "__main__":
    import torch
    
    src_img = torch.rand(64, 196, 512)  # (B, encode, embed)
    model = EncoderLayer(196, 512, 512, 8, 0.1)
    output = model(src_img)
    print(output.size())


class EncoderStack(nn.Module):
    def __init__(self, num_layers=5, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                img_encode_size=16,  # Should match ImageEncoder's encode_size
                img_embed_dim=embed_dim,
                feedforward_dim=embed_dim*4,  # 4x expansion
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ff_hidden_dim=2048, dropout=0.1, vocab_size=len(word2idx)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.cross_attn = MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.fc = nn.Linear(embed_dim, vocab_size)
        
        # Weight tying
        self.fc.weight = self.embedding.weight

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None):
        # Embedding and positional encoding
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        
        # Self-attention
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + attn_out)
        
        # Cross-attention
        cross_out, attns = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + cross_out)
        
        # FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + ffn_out)
        
        logits = self.fc(tgt)
        return logits, attns
        
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generate causal attention mask for decoder"""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class PipelineModel(nn.Module):
    def __init__(self, image_encoder, encoder_stack, decoder_layer, device):
        super().__init__()
        self.image_encoder = image_encoder.to(device)
        self.encoder_stack = encoder_stack.to(device)
        self.decoder = decoder_layer.to(device)
        self.device = device
        
    def forward(self, images, captions, tgt_mask=None):
        images = images.to(self.device)
        captions = captions.to(self.device)
        
        # Image encoding
        visual_features = self.image_encoder(images)
        encoded_features = self.encoder_stack(visual_features)
        
        # Caption decoding
        logits, attns = self.decoder(captions, encoded_features, tgt_mask)
        return logits, attns
    
    def generate(self, images, max_len=20, beam_size=3):
        # Implementation of beam search
        pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_encoder = ImageEncoder(encode_size=16)
encoder_stack = EncoderStack(num_layers=3)
decoder = DecoderLayer(vocab_size=len(word2idx))

single_pipeline = PipelineModel(
    image_encoder=image_encoder,
    encoder_stack=encoder_stack,
    decoder_layer=decoder,
    device=device
)