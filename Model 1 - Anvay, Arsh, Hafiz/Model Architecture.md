Details about how the code is structured and how the model is doing what it's doing

## Data Loading
1. Importing torch and checking if GPU is available, and doing some CUDA debugging steps
2. Defining the collate function : It is a custom batch collator. It processes and organises the data into batches for processing. '
	- **Prepare a batch of images and captions** for model input, ensuring that both the images and captions are in the correct format and size for processing.
    
	- **Handle variable-length captions** by padding them to the same length, which is necessary because different captions can have different lengths.
    
	- **Clone and detach tensors** from the computation graph to prevent unnecessary computation of gradients for data (images and captions) that shouldn't require gradients.
3. Custom DataLoaders (seperate for MSCOCO and Flickr30k)
	1. Vocabulary building for NLP - *need to add more details on this*

## Model Architecture 
![[Pasted image 20250407023437.png]]

1. **Image Encoder Class :**  
	- ==Importing Resnet with pretrained weight (should I change the backbone?)== 
	- Removing the last 2 layers of Resnet -> last linear and pool layer 
		- list unpacks the submodules of resnets (the layers) into a list  
		- * unpacks the list of layers as individual arguments to nn.Sequential
		- removing the last two layers gives us the feature maps instead of the classification outputs
		- so resnet is essentially our backbone in this model which extracts feature maps
		- we can add custom layers after this which can be used to predict classification outputs
	- Downsampling our feature maps from 2048 (resnet output) to 512 (embedding dimension) by performing a convolution with kernel size 1. 
	-  Normalizes the output of a previous layer by subtracting batch mean and dividing by batch standard deviation for each channel. 
		- after learning, a scaling and shifting transformation is applied **with learnable parameters** 
	- ==Relu activation function== maybe I can try a different activation function?
	- Adaptive resize function -> resizes the image to encode size
	- Position Embedding to capture the relative positions of objects in the image
		- nn.Parameter basically implies that this function is a parameter that is to be learned during training. 
		- this tensor will be updated during backpropogation
		- these kind of embeddings help the model understand the spatial or sequential relationships between inputs
	- **FORWARD PASS**
		- out.view() reshapes the tensor, ***the -1 tells PyTorch to infer the size of that dimension so that the total number of elements remain the same*** (this dimension usually corresponds to the flattened spatial dimensions, say 256 in this case)
		- the permute function rearranged the order -> (B,N,embed_dim)
	- FINE TUNING
		- the first loop freezes all layers of resnet
		- it unfreezes the last few layers to let them adapt to the given image captioning function (because our model is pretrained, we want to retain the weights)
		- ==I don't understand why only the last two layers are unfrozen though?== -> because we are adding attention before passing through the last linear layer, we only need the feature maps
2. **Encoder Layer Class -> Encodes the output image feature maps** 
	1. ==CNNFeedForward== - It's a feedforward network that uses 1d convolution 
	   [Paper on CNNs for FeedForward Acceleration](https://arxiv.org/pdf/1412.5474)
	- ==I should reduce dropout while training, I think I lose data as losing 25.6 pixels(data points) per image per epoch is just losing a lot of data.== 
	- it's basically a 1 hidden layer neural network
	- forward pass: permute is required for compatibility for 1d convolution function
	- adding inputs before normalizing the layer makes a residual connection
	2. Encoder Self Attention Class 
		- multi head attention on the images -> transpose is for dimensionality purposes ==why is the query key and value the same tensor??== because self attention
		- we create a residual connection as well in this
	- Then we put attention on the inputs and then pass it to the feedforward layer
	- the EncoderStack class is just stacking encoder layers on top of each other for more depth 
3. **Decoder Layer Class ->** Generates the final tokenized words
	1. position encoding class :
		- classic position embedding from the paper 'attention is all you need' -> basically captures the spatial information in the image
	2. Decoder class
		- nn.Embedding basically tokenizes the unique words in a embed_dim dimensional feature space
		- self attention and cross attention are initialised 
		- feed forward network is just FCL with 1 hidden layer with ReLU as activation function
		- self.fc is a linear projection from embed_dim back to vocab_size
		- self.fc.weight = self.embedding.weight **the first and last layers basically perform the same function, so the weights are the same, so this line just helps in reducing compute**
		- **FORWARD PASS**
			- embedding the input, and then addition positional encoding to it
			- self attention using the same input + normalisation of the residual connection
			- cross attention with the ==memory== tensor and then similar normalisation
			- feedforward and then normalisation
			- then we get our tokens from the self.fc function as logits. 
			- ==I feel like the decoder architecture can be significantly improved==
		- Generating Attention Mask (so that the model doesn't look into the future)
4. **Trainer Class** 
	1. **Single Pipeline Class :** Basically it just combines all the three classes that we just defined to create a model that generates captions, that is why it is called single_pipeline lmao. 
	2. **Trainer Class :** 
		- Generate Subsequent Mask : mask to prevent non-causality
		- padding mask for sequence -> so that all captions are of the same length 
		- train_one_epoch function
			- get image and caption, and then get masks for them
			- optimizer -> ADAM
			- loss is calculated between logits(generated output) and the available captions
			- loss.backward() is for back propogation
			- self.optimizer.step() does optimization on the weights
		- validation is done without any sort of backpropogation (no_grad fxn)
		- the run function just passes inputs in the trainer. 
	*-> Did I create caption padding masks before? I might need to do that*
5. **Caption Generation Function** :
	- temperature : a scaling factor applied to the logits to control the randomness of predictions. lower values make predictions more confident, while higher values flatten the distribution
	- top_k : the function limits the predictions to the top-k most likely tokens. 
	- top_p : Also known as nucleus sampling, this parameter restricts the predictions to the smallest set of tokens with cumulative probability ≥ top_p.
	- model.eval() -> sets model to evaluation mode
	- unsqueezing to add batch dimensions
	- < start >, < end >,< pad > are unique tokens used in the sentences
	- caption_indices = [start_idx] initialise the list with the start index
	- dec_input -> handles the captions (decoder inputs)
	- generating the required masks
	- next_token_logits = logits[0, -1, :] / temperature -> logits are selected for the last time step 
	- temperature scaling adjusts the probability distribution
	- top_k filtering -> it ensures that only the top-k tokens have non-zero probability 
	- top_p 
		- **When `top_p < 1.0`:**
		- **Sorting:** The logits are sorted in descending order.
		- **Cumulative Probability:** The softmax is applied to these sorted logits to obtain probabilities, and then a cumulative sum is computed.
		- **Thresholding:** Tokens are flagged for removal if adding them would exceed the cumulative probability threshold `top_p`. The first token is always kept
		- **Masking:** Using `scatter`, the original indices are restored to a mask that aligns with `next_token_logits`, and those logits are set to `-inf` so that only the most probable tokens (that together account for at least `top_p` of the probability mass) remain.


