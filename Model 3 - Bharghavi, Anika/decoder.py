import torch
import torch.nn as nn

class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, feature_dim):
        super(DecoderWithAttention, self).__init__()
        self.attention = Attention(feature_dim, hidden_size, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + feature_dim, hidden_size)
        self.fc_word = nn.Linear(hidden_size, vocab_size)
        self.fc_region = nn.Linear(hidden_size, 196)  # 14x14 grid

    def forward(self, word_inputs, image_features, hidden_state, cell_state):
        embeddings = self.embedding(word_inputs)
        attention_features, attention_weights = self.attention(image_features, hidden_state, embeddings)
        lstm_input = torch.cat((embeddings, attention_features), dim=1)
        hidden_state, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))
        word_output = self.fc_word(hidden_state)
        region_output = self.fc_region(hidden_state)
        return word_output, region_output, hidden_state, cell_state, attention_weights
