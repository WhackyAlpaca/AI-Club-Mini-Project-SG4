import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialTransformer(nn.Module):
    def __init__(self, input_dim):
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 6)
        )


        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.localization(x).view(-1, 2, 3)
        gird = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)
    
class CNN_Encoder(nn.Module):
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_transformer = SpatialTransformer(2048)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.fc = nn.Linear(2048, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = self.spatial_transformer(features)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1).view(features.size(0), -1, features.size(-1))
        features = self.fc(features)
        return features