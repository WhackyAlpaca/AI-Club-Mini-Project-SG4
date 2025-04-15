# Image Captioning Model

**Encoder**: ResNet-152  
**Decoder**: LSTM with Gated Attention Mechanism  

---

## Datasets Used

- MS COCO 2014  
- MS COCO 2017

We preprocess the datasets into:
- `train_image_paths`, `train_captions`
- `val_image_paths`, `val_captions`

Each image is associated with up to 5 different captions to ensure diversity and reduce bias in BLEU score evaluation.

---

##  Model Architecture

###  Encoder: ResNet-152

We use **ResNet-152**, a deep convolutional neural network with many residual connections. Its depth and skip connections help alleviate the vanishing gradient problem, allowing the model to learn complex visual patterns more effectively.

The encoder outputs feature maps of shape `[batch_size, 7, 7, 2048]`, which we flatten into 49 spatial image features (7x7), each of 2048 dimensions.

---

###  Decoder: LSTM with Gated Attention

The decoder consists of an **LSTM cell** combined with a **gated attention mechanism**.

#### Attention Module:

1. At each decoding step, the hidden state is compared with all 49 image features to compute **raw compatibility scores**.
2. These scores are passed through a softmax layer to obtain attention weights over spatial regions.
3. A **weighted context vector** is computed as a weighted sum of the image features.
4. A **gating scalar**, obtained by passing the hidden state through a sigmoid, determines how much of the context vector should influence the next word prediction.
5. This gated context is concatenated with the current word embedding and fed into the LSTM to predict the next token until an `<eos>` (end-of-sequence) token is generated.

This mechanism allows the model to "attend" to different parts of the image dynamically during caption generation.

---

## Evaluation Metric: BLEU Score

**BLEU (Bilingual Evaluation Understudy)** is used to evaluate the similarity between machine-generated captions and ground truth captions.

- Compares n-grams (up to 4-grams) between predicted and reference captions.
- Includes a **Brevity Penalty** to penalize overly short generated captions.

**Limitations**:

- Insensitive to synonyms and paraphrasing.
- Prefers exact n-gram matches, which may not always reflect true semantic similarity.

---

##  Visualizing Attention

At each decoding time step, we extract and visualize the attention scores to highlight **which regions of the image** the model focuses on while generating each word in the caption.

This helps interpret and analyze how the model aligns visual information with textual generation.
