import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json
import re
import pandas as pd
from torchvision import transforms

nltk.download('punkt', quiet=True)

def load_model_and_vocab(model_path, word2idx_path, idx2word_path, device):
    with open(word2idx_path, 'r') as f:
        word2idx = json.load(f)

    with open(idx2word_path, 'r') as f:
        idx2word = json.load(f)
    
    idx2word = {int(k): v for k, v in idx2word.items()}

    image_encoder = ImageEncoder(encode_size=16, embed_dim=512)
    encoder_stack = EncoderStack(num_layers=3)  
    decoder = DecoderLayer(vocab_size=len(word2idx))

    model = PipelineModel(
        image_encoder=image_encoder,
        encoder_stack=encoder_stack,
        decoder_layer=decoder,
        device=device
    )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    return model, word2idx, idx2word

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

# Process captions file
def load_captions(captions_file):
    image_captions = {}
    
    with open(captions_file, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by first comma, then handle potential quotes
            parts = line.split(",", 1)
            if len(parts) < 2:
                print(f"Skipping malformed line: {line}")
                continue
            
            image_name = parts[0].strip().replace('"', '')
            caption = parts[1].strip().replace('"', '')
            
            if image_name not in image_captions:
                image_captions[image_name] = []
            image_captions[image_name].append(caption)
    
    return image_captions

def evaluate_model(model, image_captions, word2idx, idx2word, device, images_dir, num_samples=100, visualize=True):
    references = []
    hypotheses = []
    
    image_keys = list(image_captions.keys())[:num_samples]
    successful_evals = 0
    
    for i, image_key in enumerate(image_keys):
        image_path = os.path.join(images_dir, image_key)
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
            
        try:
            image_tensor = load_image_as_tensor(image_path)
            
            # Generate caption
            generated_caption = generate_caption(model, image_tensor, word2idx, idx2word, device)
            
            if not generated_caption:
                print(f"Empty caption generated for {image_key}")
                continue
                
            reference_captions = image_captions[image_key]
            processed_references = [nltk.word_tokenize(ref.lower()) for ref in reference_captions]
            
            hypothesis = nltk.word_tokenize(generated_caption.lower())
            
            references.append(processed_references)
            hypotheses.append(hypothesis)
            successful_evals += 1
            
            # Display image and caption if visualization is enabled
            if visualize and successful_evals <= 30:  
                plt.figure(figsize=(10, 8))
                image = Image.open(image_path)
                plt.imshow(image)
                plt.axis("off")
                plt.title(f"Generated: {generated_caption}", fontsize=12)
                plt.figtext(0.5, 0.01, f"References:", ha="center", fontsize=10, fontweight="bold")
                for j, ref in enumerate(reference_captions[:3]):
                    plt.figtext(0.5, -0.02-j*0.03, f"{j+1}. {ref}", ha="center", fontsize=9)
                
                plt.tight_layout()
                plt.show()
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(image_keys)} images, {successful_evals} successful")
            
        except Exception as e:
            import traceback
            print(f"Error processing {image_key}: {str(e)}")
            print(traceback.format_exc())
    
    # Check if we have enough data to calculate BLEU
    if len(references) == 0 or len(hypotheses) == 0:
        print("Error: No valid references or hypotheses to calculate BLEU scores")
        return 0, 0, 0, 0
    
    
    smoothing = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    
    return bleu1, bleu2, bleu3, bleu4

# Main evaluation function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    model_path = '/kaggle/input/image_captioning_swin/other/default/1/model_state_2025-04-07 20_23_05.707048.pt'
    word2idx_path = '/kaggle/input/image_captioning_swin/other/default/1/word2idx.json'
    idx2word_path = '/kaggle/input/image_captioning_swin/other/default/1/idx2word.json'
    captions_file = '/kaggle/input/flickr30k/captions.txt'
    images_dir = '/kaggle/input/flickr30k/Images'
    
    model, word2idx, idx2word = load_model_and_vocab(model_path, word2idx_path, idx2word_path, device)
    
    image_captions = load_captions(captions_file)
    
    bleu_scores = evaluate_model(
        model=model,
        image_captions=image_captions,
        word2idx=word2idx,
        idx2word=idx2word,
        device=device,
        images_dir=images_dir,
        num_samples=50,  
        visualize=True
    )
    
    return bleu_scores

if __name__ == "__main__":
    print(main())
