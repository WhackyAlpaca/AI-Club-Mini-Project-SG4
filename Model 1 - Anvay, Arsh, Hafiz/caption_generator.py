### LOAD PRETRAINED MODEL
# single_pipeline = PipelineModel(image_encoder, encoder_layer, decoder_layer,device)
# single_pipeline.load_state_dict(torch.load('/kaggle/input/model-1/pytorch/default/1/model_state_2025-04-03 202132.820658.pt',weights_only=True))

import torch
from PIL import Image
import torchvision.transforms as transforms

def generate_caption(
    model, 
    image_tensor, 
    word2idx, 
    idx2word, 
    device, 
    max_length=20, 
    temperature=1.0, 
    top_k=0, 
    top_p=1.0,
    beam_size=1
):
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device).unsqueeze(0)  # Add batch dimension
    
    # Special tokens handling
    start_idx = word2idx["<start>"]
    end_idx = word2idx["<end>"]
    pad_idx = word2idx.get("<pad>", 0)
    
    # Initialize sequence
    caption_indices = [start_idx]
    
    with torch.no_grad():
        for i in range(max_length):
            dec_input = torch.tensor(caption_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            # Generate proper masks
            seq_len = dec_input.size(1)
            # tgt_mask = model.decoder.generate_square_subsequent_mask(seq_len).to(device)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
            tgt_padding_mask = (dec_input == pad_idx)
            
            # Forward pass through complete pipeline
            logits, _ = model(
                images=image_tensor,
                captions=dec_input,
                tgt_mask=tgt_mask
            )
            
            # Get last predicted token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=0, 
                    index=sorted_indices, 
                    src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Handle numerical stability
            probs = torch.nan_to_num(probs, nan=1e-5)
            probs = probs / probs.sum()
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            caption_indices.append(next_token)
            if next_token == end_idx:
                break

    # Convert indices to words
    caption_words = [
        idx2word[idx] 
        for idx in caption_indices 
        if idx not in {start_idx, end_idx, pad_idx}
    ]
    
    return " ".join(caption_words).capitalize() + "."


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from PIL import Image
import torchvision.transforms as transforms

def load_image_as_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image) 
    return image_tensor

# image_path = '/kaggle/input/flickr30k/Images/1010673430.jpg'
# tensor = load_image_as_tensor(image_path)
# # tensor = tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]

# import matplotlib.pyplot as plt

# image_array = plt.imread("/kaggle/input/flickr30k/Images/1010673430.jpg")
# plt.imshow(image_array)
# plt.axis("off")
# plt.show()

# generated_caption = generate_caption(single_pipeline, tensor, word2idx, idx2word, device, max_length=20)
# print("generated caption:" , generated_caption)