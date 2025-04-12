#Areas of Attention for Image Captioning

This project implements the **Areas of Attention** model for **image captioning** in PyTorch.

The model dynamically **attends to image regions** while generating each word of the caption, leading to more accurate and grounded descriptions.

## Model Architecture

1. Encoder.py:

**SpatialTransformer(nn.Module)**: a lightweight neural network that helps the model focus dynamically on important regions within the image feature maps.
__init__(self, input_dim) : intilializes the spatial transformer network: two convolutional layers followed by a fully connected layer.
forward(self, x): computes a transformation matrix(theta) from the input feature map. Then applies affine transformation using affine_grid and grid_sample to warp the feature map.

**CNN_Encoder(nn.Module)**: the image encoder module. It extracts spatial features from images using a pre-trained ResNet-50 applies spatial transformation, resizes and then projects features into an embedding space.

__init__(self, embed_size): loads a ResNet-50 model up to the final convolutional layers. Adds a spatialtransformer for dynamic region focus. Adds an adaptive avg pooling to a 14*14 grid. Projects features into an embed_size-dimensional space using a fully connected layer.

forward(self, images): passes input images through ResNet-50
applies spatial transformer to focus attention
pools feature maps to fixed 14*14 spatial grid. Flattens the feature map into (batch, regions, channels). Projects to embedding dimension for attention decoder. 

The spatial transformer acts like a mini attention system inside the encoder itself. It warps the feature map to focus on key visual ares.
Using a ResNet backbone ensures strong feature extraction without training from scratch.
##
2.**Attention.py**:

**Attention(nn.Module)**: implements the three-way attention mechanism used in Areas of Attention. Dynamically computes attention scores by considering the image regions, current LSTM hidden state and current word embedding

__init__(self, feature_dim, hidden_dim, word_dim): initializes layers for projecting:
-image region features(feature_fc)
-hidden state(hidden_fc)
-word embedding(word_fc)
to a common hidden_dim space. Then computes scalar attention scores with alpha_fc.

forward(self, image_features, hidden_state, word_embedding): projects image features, hidden state and word embedding separately. Adds them together(3-way interaction). Applies tanh activation. Computes attention scores (alpha_fc) and normalizes them using softmax. Calculates a weighted sum of image features based on attention scores. Returns weighted features (context vector) and the raw attention weights. 

Unlike standard attention(which usually uses only the hidden state), this three-way attention uses word embedding too. This allows the model to adapt its focus not just based on where it is in the sequence but also what word it expects next. Weighted sum ensures that the model summarizes the image regions into a single context vector dynamically at each time step. 

##

3.**Decoder.py**:

**DecoderWithAttention(nn.Module)**: the caption generator module. It decodes the encoded image features into a natural language sentence, one word at a time, by attending dynamically to different spatial regions of the image. 

__init__(self, embed_size, hidden_size, vocab_size, feature_dim): Initializes attention module(three-way attention), embedding layer to convert word indices to dense vectors. LSTMCell that updates hidden and cell states each time step. fc_word fully connected layer to predict the next word. fc_region fully connected layer to predict the attended spatial region.

forward(self, word_inputs, image_features, hidden_state, cell_state): embeds the current input word. Computes context vector from image regions using attention. Concatenates word embedding and attended image features. Feeds them into the LSTMCell to update hidden and cell states. Uses fc_word to predict the next word. Uses fc_region to predict the most relevant image region. Returns word predictions, region predictions, new hidden states and attention weights.

At each decoding step, the decoder looks at a different region of the image depending on the caption generated so far. 
LSTMCell is used instead of standard LSTM so that we have more control at each time step (feeding in custom inputs).
##

4.**Caption_model.py**:

**ImageCaptioningModel(nn.Module)**: This is the full end-to-end model for image captioning. It combines the CNN-based Encoder and attention-based Decoder into a single system that can map an input image to an output caption sequence.

__init__(self, embed_size, hidden_size, vocab_size, feature_dim): Initializes encoder (CNN + Spatial Transformer) to process images. Initializes decoder (LSTM with Attention) to generate captions word-by-word. 

forward(self, images, captions): Encodes input images to get feature maps. Initializes hidden and cell states of the decoder to zeros. For each timestep (word) in the input caption: passes the previous word, image features, hidden state to the decoder. Gets the next word prediction. Stacks the outputs across timesteps to get full sequence prediction.

**Two-stage pipeline**:
Encoder extracts a compact visual representation of the image. Decoder learns to generate language conditioned on both image features and previously generated words. 
Teacher forcing: during training, decoder is fed the true previous word (from ground truth captions) not its own predicted word. This speeds up learning. 
##
5. **Dataset.py**:

**COCOCaptionDataset(Dataset)**: A custom PyTorch dataset class that loads images and their associated captions from the MSCOCO dataset. Handles vocabulary building, image preprocessing and dynamic sample limiting.

__init__(self, coco_dir, annotation_file, transform=None, max_samples=None): Loads the COCO annotation JSON using the COCO API. Maps images and captions. Allows limiting to a subset(max_samples) for faster experiments. Builds the vocabulary from loaded captions.

create_dummy_dataset(self): In case of dataset loading failure, creates dummy images and captions for testing the code pipeline without actual data.
__len__(self): returns the number of caption entries(i.e. the dataset size).

__getitem__(self, idx): retrieves the image and corresponding caption for a given index. Loads and transforms the image. Converts the caption into a tensor of word indices(with <start>, <end>, <unk> tokens). Returns transformed image, tokenized caption tensor, raw caption string and filename.

build_vocab(self): scans all captions. Counts word frequencies. Builds a vocabulary of words appearing at least 5 times (threshold). Creates mappings word to index (stoi) and index to word (itos).

The dataset is image + caption pairs.
Captions are tokenized using NLTK's word tokenizer and then numericalized using the built vocabulary.
##

6. **Training.py**:

**train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', save_path='./')**: trains and validates the image captioning model over multiple epochs, saving the best-performing checkpoint based on validation loss.

**training loop**: For each epoch:
               Puts model in train() mode.
               Iterates through training batches.
               For each batch: moves data to GPU, forward pass to compute predictions, reshapes outputs and targets, calculates cross-entropy loss, backward pass + optimizer step, clips gradients to stabilize training, logs and stores average training loss.
               
**validation loop**: After training each epoch:
                 Puts model in eval() mode.
                 Iterates through validation batches
                 Computes validation loss without updating weights
                 Logs and stores average validation loss
                 
**saving model**: if the validation loss improves, saves the model as best_model.pth
Always saves the final model after training as final_model.pth.
##

7.**Caption_generation.py**:

**predict_caption(model, image, vocab, max_length=20, device='cuda')**: takes a trained captioning model and an input image and generates a natural language caption word-by-word, dynamically attending to image regions.

**model.eval()**: puts the model into evaluation mode (disabling dropout, batchnorm updates).

**Move image to GPU and batch it**: adds batch dimension if needed and moves to GPU.

**Encode the image**: passes the image through the encoder to extract spatial feature maps.

**Initialize LSTM states**: creates zero tensors for the LSTM's hidden and cell states.

**Start from <start> token**: feeds the <start> token to the decoder to kick off caption generation.
At each step, predicts the next word.

**Decode step-by-step**: feeds current token, hidden state and image features into the decoder.
Gets next word prediction and attention weights
Updates hidden and cell states for next step.

**Stopping conditions**: if the model predicts the <end> token, caption generation stops.
Also stops after max_length words if no end is predicted. 

**Output**: returns generated caption as a string. Attention weights at each step (useful for visualization).
##

8.**Evaluate.py**:

**evaluate_model_with_saved_checkpoint(model_path, dataset, device='cuda', num_samples=5)**: Loads a trained model checkpoint, evaluates it on a small sample of the dataset, calculates the BLEU scores and visualizes predicted captions

**load model**: creates an instance of ImageCaptioningModel with the expected architecture.
Loads the trained weights from model_path.

**Create evaluation DataLoader**: creates a small DataLoader with a few shuffled samples for test evaluation. Uses a custom collate function to handle variable-length captions.

**Calculate BLEU scores**: uses calculate_bleu_scores() to compute BLEU-1 to BLEU-4 scores on predictions vs ground truth captions. Prints each score nicely. 

**Visualize predictions**: calls visualize_predictions() to show sample images along with reference caption, predicted caption, BLEU scores for that example
##
9.**Main.py**:
**Device setup**: detects if a GPU is available (cuda) otherwise uses CPU.

**Dataset Paths**: calls find_coco_paths() to locate COCO images and annotation files for training and validation.

**Data Transforms**: Defines image preprocessing: resize to 224*224, normalize pixel values.

**Dataset and DataLoader setup**: creates COCOCaptionDataset objects for training and validation. Applies a custom collate_fn() to handle variable-length captions by padding them. Builds train_loader and val_loader with mini batches.

**Model initialization**: creates an instance of the imagecaptioningmodel (encoder + decoder) and moves it to the device.

**Loss Function and Optimizer**: cross-entropy loss ignoring <pad> tokens. Adam optimizer with a learning rate of 3e-4.

**Training Loop**: trains the model for 10 epochs. Saves the best checkpoint(lowest validation loss) to model_checkpoints/.

**Best Model loading**: after training, loads the best saved model for evaluation and caption generation

**Visualization**: Visualizes a few predictions on validation set images. Shows generated captions vs ground truth

**BLEU score calculation**: calculates BLEU-1 to BLEU-4 scores across the entire validation set to assess caption quality.
##
**Credits**:

**Original paper: "Areas of Attention for Image Captioning"**

**Dataset: MSCOCO**



