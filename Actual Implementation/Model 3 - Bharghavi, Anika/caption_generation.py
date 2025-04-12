#Caption generation function
def predict_caption(model, image, vocab, max_length=20, device=torch.device('cuda')):
    
    model.eval()

    #Move image to device and add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)    
    
    with torch.no_grad():
        #Encode image to get features
        image_features = model.encoder(image)

        #Initialize decoder states
        batch_size = image.size(0)
        hidden_state = torch.zeros(batch_size, model.decoder.lstm.hidden_size).to(device)
        cell_state = torch.zeros(batch_size, model.decoder.lstm.hidden_size).to(device)

        #Get start token index
        start_idx = vocab['<start>']
        end_idx = vocab['<end>']

        #Start with <start> token
        curr_token = torch.tensor([start_idx]).to(device)

        caption = []
        attention_weights_list = []

        #Generate caption word by word
        for _ in range(max_length):
            #Forward pass through decoder
            output, hidden_state, cell_state, attention_weights = model.decoder(
                curr_token, image_features, hidden_state, cell_state
            )

            #Get the predicted word (highest score)
            _, predicted = output.max(1)

            #Convert token to word
            curr_token_idx = predicted.item()

            #Store attention weights for visualization
            attention_weights_list.append(attention_weights.cpu().numpy())

            #if end token is predicted, stop generation
            if curr_token_idx == end_idx:
                break

            #Add word to caption
            itos = {idx: word for word, idx in vocab.items()}
            word = itos[curr_token_idx]
            if word not in ['<start>', '<pad>', '<unk>', '<end>']:
                caption.append(word)

            #Update current token for next iteration
            curr_token = predicted

    #Join words to create complete caption
    return ' '.join(caption), attention_weights_list