import torch

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Name:", torch.cuda.get_device_name(1))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# %env CUDA_LAUNCH_BLOCKING=1
torch.autograd.set_detect_anomaly(True)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images = []
    captions = []

    for img, cap in batch:
        img_tensor = img.clone().detach().float()
        images.append(img_tensor)

        cap_tensor = cap.clone().detach().long()  
        captions.append(cap_tensor)

    images = torch.stack(images)  
    captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions

'''
Imported FLickr30k from Kaggle (we ran the code directly on Kaggle) 
'''

import pandas as pd
import re

txt_file = "/kaggle/input/flickr30k/captions.txt"  
csv_file = "flickr30k_captions.csv"

with open(txt_file, "r", encoding="utf-8") as f:
    for _ in range(5):  
        print(f.readline().strip())  

data = []
with open(txt_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  
        
        # Split on first whitespace to separate image and caption
        parts = re.split(r'\s+', line, maxsplit=1)
        if len(parts) < 2:
            print(f"⚠️ Skipping malformed line: {line}")
            continue 
        
        image_name = parts[0].split("#")[0].strip('", ')
        caption = parts[1]
        
        # Clean the caption
        caption = caption.strip('", ')
        caption = re.sub(r'\s+', ' ', caption)  # Normalize multiple spaces
        caption = caption.replace('"""', '"').replace('""', '"')
        caption = caption.strip()
        
        data.append([image_name, caption])

df = pd.DataFrame(data, columns=["Image", "Caption"])
print(df.head())

if df.empty:
    print("⚠️ Warning: DataFrame is empty. Check text file format.")
else:
    print(f"✅ DataFrame has {len(df)} entries.")

df.to_csv(csv_file, index=False)
print(f"📂 CSV file '{csv_file}' created successfully with {len(df)} entries!")

import json
import os
import random
import shutil
from sklearn.model_selection import train_test_split

dataset_dir = "/kaggle/input/flickr30k/Images"
captions_file = "/kaggle/working/flickr30k_captions.csv"  

small_train_dir = "./small_flickr_train"
small_val_dir = "./small_flickr_val"

os.makedirs(small_train_dir, exist_ok=True)
os.makedirs(small_val_dir, exist_ok=True)

#We specifically chose the number of images so that there is a smaller 
#datset to train

num_train_samples = 30000
num_val_samples = 500

captions_data = {}

with open(captions_file, "r", encoding="utf-8") as f:
    next(f)  
    
    for line in f:
        line = line.strip()
        if not line:
            continue  
        
        parts = line.split(",", maxsplit=1)  
        
        if len(parts) < 2:
            print(f"Skipping malformed line: {line}")  
            continue  
        
        image_name, caption = parts[0].strip(), parts[1].strip()
        image_name = parts[0].strip().replace('"', '')  

        # Store captions in dictionary
        if image_name not in captions_data:
            captions_data[image_name] = []
        captions_data[image_name].append(caption)

all_images = list(captions_data.keys())

random.seed(42)
train_images, val_images = train_test_split(all_images, test_size=num_val_samples, train_size=num_train_samples, random_state=42)

for img in train_images:
    shutil.copy(os.path.join(dataset_dir, img), os.path.join(small_train_dir, img))

for img in val_images:
    shutil.copy(os.path.join(dataset_dir, img), os.path.join(small_val_dir, img))

train_annotations = {img: captions_data[img] for img in train_images}
val_annotations = {img: captions_data[img] for img in val_images}

with open("train_flickr_annotations.json", "w") as f:
    json.dump(train_annotations, f, indent=4)

with open("val_flickr_annotations.json", "w") as f:
    json.dump(val_annotations, f, indent=4)

print("Flickr30K subset dataset created successfully!")

import os
import json
from nltk.tokenize import word_tokenize
from nltk import download
import numpy as np
from PIL import Image
from collections import Counter
import re

download('punkt')

def load_flickr_annotations_json(captions_file):

    with open(captions_file, "r", encoding="utf-8") as f:
        captions_data = json.load(f)
    return captions_data

def load_image(image_path):
    with Image.open(image_path) as img:
        return np.array(img.convert('RGB'))
    
def build_dataset(captions_file, images_dir):

    captions_data = load_flickr_annotations_json(captions_file)
    dataset = {}
    
    for file_name, captions in captions_data.items():
        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found. Skipping.")
            continue
        
        image_data = load_image(image_path)
        dataset[file_name] = {
            'image': image_data,
            'captions': captions
        }
    return dataset 

def tokenize_caption(caption):
    return word_tokenize(caption.lower())

def build_vocab(captions_list, min_count=2):
    counter = Counter()
    for caption in captions_list:
        tokens = tokenize_caption(caption)
        counter.update(tokens)
    
    # Only include words that appear at least min_count times
    vocab_words = [word for word, count in counter.items() if count >= min_count]
    # Add special tokens
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
    vocab = special_tokens + sorted(vocab_words)
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def encode_caption_to_indices(caption, word2idx):
    tokens = tokenize_caption(caption)
    encoded = [word2idx['<start>']]
    for token in tokens:
        encoded.append(word2idx.get(token, word2idx['<unk>']))
    encoded.append(word2idx['<end>'])
    return encoded

def encode_all_captions_to_indices(captions_list, word2idx):
    return [encode_caption_to_indices(caption, word2idx) for caption in captions_list]

if __name__ == '__main__':
    train_captions_file = '/kaggle/working/train_flickr_annotations.json'
    train_images_dir = '/kaggle/working/small_flickr_train'
    val_captions_file = '/kaggle/working/val_flickr_annotations.json'
    val_images_dir = '/kaggle/working/small_flickr_val'

    print('Building train dataset...')
    train_dataset = build_dataset(train_captions_file, train_images_dir)
    print('Building validation dataset...')
    val_dataset = build_dataset(val_captions_file, val_images_dir)

    train_captions = [caption for entry in train_dataset.values() for caption in entry['captions']]
    
    print("Building vocabulary from training captions...")
    word2idx, idx2word = build_vocab(train_captions, min_count=1)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")

    def add_encoded_captions(dataset, word2idx):
        for key, entry in dataset.items():
            captions = entry['captions']
            entry['captions_encoded'] = [encode_caption_to_indices(caption, word2idx) for caption in captions]

    print('Encoding training captions...')
    add_encoded_captions(train_dataset, word2idx)
    print('Encoding validation captions...')
    add_encoded_captions(val_dataset, word2idx)

    sample_key = list(train_dataset.keys())[0]
    sample_entry = train_dataset[sample_key]

    print(f"\nSample image: {sample_key}")
    print(f"Image shape: {sample_entry['image'].shape}")
    print(f"Original captions: {sample_entry['captions']}")
    if sample_entry.get('captions_encoded'):
        print(f"Encoded caption (first): {sample_entry['captions_encoded'][0]}")
    else:
        print("No encoded caption.")

from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

class Flickr30Dataset(Dataset):
    def __init__(self, dataset, transform=None):

        self.dataset = dataset
        self.keys = list(dataset.keys())  
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        entry = self.dataset[key]

        image = entry['image']  # expected shape: (H, W, C)
        if self.transform:
            image = self.transform(image)

        captions_encoded = entry.get('captions_encoded', [])
        if len(captions_encoded) == 0:
            caption_encoded = torch.zeros(512)
        else:
            caption_encoded = torch.tensor(captions_encoded[np.random.randint(len(captions_encoded))], dtype=torch.float)
            #we could have used more than 1 caption?!?              
        return image, caption_encoded

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

flickr_train_dataset = Flickr30Dataset(train_dataset, transform=transform)
flickr_val_dataset = Flickr30Dataset(val_dataset, transform=transform)

train_loader = DataLoader(
    flickr_train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
)

val_loader = DataLoader(
    flickr_val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
)

for images, captions in train_loader:
    print("Images shape:", images.shape)
    print("Captions shape:", captions.shape)
    print("Captions dtype:", captions.dtype)
    print("Caption min:", captions.min().item(), "max:", captions.max().item())
    break

# ### LOAD VOCABULARY IF USING PRETRAINED MODEL
# import json

# with open('/kaggle/working/word2idx.json', 'r') as f:
#     word2idx = json.load(f)

# with open('/kaggle/working/idx2word.json', 'r') as f:
#     idx2word = json.load(f)