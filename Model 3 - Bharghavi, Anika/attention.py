import torch
import torch.nn as nn

class Attention(nn.Module):
    """Three-way attention between image regions, RNN hidden state, and word embedding."""
    def __init__(self, feature_dim, hidden_dim, word_dim):
        super(Attention, self).__init__()
        self.feature_fc = nn.Linear(feature_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.word_fc = nn.Linear(word_dim, hidden_dim)
        self.alpha_fc = nn.Linear(hidden_dim, 1)

    def forward(self, image_features, hidden_state, word_embedding):
        image_proj = self.feature_fc(image_features)
        hidden_proj = self.hidden_fc(hidden_state).unsqueeze(1)
        word_proj = self.word_fc(word_embedding).unsqueeze(1)
        combined = torch.tanh(image_proj + hidden_proj + word_proj)
        attention_weights = self.alpha_fc(combined)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_features = (attention_weights * image_features).sum(dim=1)
        return weighted_features, attention_weights.squeeze(-1)
