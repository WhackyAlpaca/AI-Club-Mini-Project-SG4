# Define complete Image Captioning model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, feature_dim):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = CNN_Encoder(embed_size)
        self.decoder = DecoderWithAttention(embed_size, hidden_size, vocab_size, feature_dim)

    def forward(self, images, captions):
        image_features = self.encoder(images)
        batch_size = images.size(0)
        hidden_state = torch.zeros(batch_size, self.decoder.lstm.hidden_size).to(images.device)
        cell_state = torch.zeros(batch_size, self.decoder.lstm.hidden_size).to(images.device)

        outputs = []
        for t in range(captions.size(1) - 1): #-1 because we dont need to predict after <end>
            word_input = captions[:, t]
            output, hidden_state, cell_state, _ = self.decoder(
                word_input, image_features, hidden_state, cell_state
            )
            outputs.append(output)

        return torch.stack(outputs, dim=1)