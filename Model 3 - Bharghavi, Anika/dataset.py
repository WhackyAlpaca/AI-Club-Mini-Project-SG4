#Define COCO dataset with subset option
class COCOCaptionDataset(Dataset):
    def __init__(self, coco_dir, annotation_file, transform=None, max_samples=None):
        self.coco_dir = coco_dir
        self.transform = transform

        #Initialize COCO API
        try:
            self.coco = COCO(annotation_file)
        
            #Load annotations - COCO format is different than simple JSON
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)

            #COCO annotations structure: {'images': [...], 'annotations':[...]}
            self.image_data = {img['id']: img for img in self.annotations['images']}
            self.captions = self.annotations['annotations']

            #Limit samples if requested
            if max_samples and max_samples < len(self.captions):
                print(f"Using {max_samples} samples out of {len(self.captions)}")
                self.captions = self.captions[:max_samples]
            
            #Build vocabulary
            self.vocab, self.itos, self.stoi = self.build_vocab()

        except Exception as e:
            print(f"Error loading annotation file: {e}")
            #Create dummy data if annotation loading fails
            self.create_dummy_data()

    def create_dummy_data(self):
        print("Creating dummy dataset for testing")
        vocab_size = 1000
        self.captions = [{'caption': 'dummy caption', 'image_id': i} for i in range(100)]
        self.image_data = {i: {'file_name': 'dummy.jpg'} for i in range(100)}
        self.vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        #Add some dummy words
        for i in range(4, vocab_size):
            self.vocab[f'word_{i}'] = i

        #Create reverse mappings
        self.itos = {idx: word for word, idx in self.vocab.items()}
        self.stoi = self.vocab

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        try:
            #Get annotation and image info
            annotation = self.captions[idx]
            image_id = annotation['image_id']
            image_info = self.coco.loadImgs(image_id)[0]

            #Get image
            image_path = os.path.join(self.coco_dir, image_info['file_name'])
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                #return a default blank image in case of error
                image = Image.new('RGB', (224, 224))

            #Get caption
            caption = annotation['caption']

            #Apply transforms
            if self.transform:
                image = self.transform(image)

            #Convert caption to tensor
            caption_tokens = [self.stoi['<start>']] + [self.stoi.get(word.lower(), self.stoi['<unk>'])
                                                      for word in nltk.word_tokenize(caption.lower())] + [self.stoi['<end>']]
            caption_tensor = torch.tensor(caption_tokens)

            return image, caption_tensor, caption, image_info['file_name']

        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            #return dummy data
            default_image = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224))
            default_caption = torch.tensor([self.vocab['<start>'], self.vocab['unk'], self.vocab['<end>']])
            return default_image, default_caption, "Error loading image", "dummy.jpg"
        
    def build_vocab(self):
        #Simple vocabulary builder
        word_counts = {}
        threshold = 5

        #Process all captions
        for annotation in self.captions:
            caption = annotation['caption']
            for word in nltk.word_tokenize(caption.lower()):
                word_counts[word] = word_counts.get(word, 0) + 1

        #Create vocab dictionary
        vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        word_idx = 4
        for word, count in word_counts.items():
            if count >= threshold:
                vocab[word] = word_idx
                word_idx += 1

        #Create index to word mapping
        itos = {idx: word for word, idx in vocab.items()}
        stoi = vocab #word to index

        print(f"Vocabulary built with {len(vocab)} words")
        return vocab, itos, stoi