import torch
import torch.nn as nn
import torchvision.transforms as transforms

ARCH = {
    "VGG16": [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP'],
    "VGG19": [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512, 512, 'MP', 512, 512, 512, 512, 'MP']
}

class VGG_net(torch.nn.Module):
  def __init__(self, vgg_layout, num_classes=1000, in_channels=3, image_dim=224): #224x244 image input
    super(VGG_net, self).__init__()

    self.vgg_layout = vgg_layout
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.image_dim = image_dim

    self.features = self.create_features()
    self.fcl = self.create_fcl()

  def forward(self, x):
    x = self.features(x)
    x = x.reshape(x.shape[0], -1) #flattens output to fully-connected layer
    x = self.fcl(x)
    return x

  def create_features(self):
    layers = []
    in_channels = self.in_channels

    for layer in self.vgg_layout:
      if layer == 'MP':
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
      else:
        out_channels = layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels

    return nn.Sequential(*layers)

  def create_fcl(self):
    trunc_image_dim = int(self.image_dim / (2**5)) #for vgg arch, must be multiple of 32 by def
    layers = []
    layers += [nn.Linear(trunc_image_dim*trunc_image_dim*512, 4096),
               nn.ReLU(), nn.Dropout(p=0.5)]
    layers += [nn.Linear(4096, 4096),
               nn.ReLU(), nn.Dropout(p=0.5)]
    layers.append(nn.Linear(4096, self.num_classes))

    return nn.Sequential(*layers)

model1 = VGG_net(ARCH['VGG16'], num_classes=1000, in_channels=3)
x = torch.randn(1, 3, 224, 224)
print(model1(x).shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model2 = VGG_net(ARCH['VGG19'], num_classes=80, in_channels=3).to(device)
x = torch.randn(1, 3, 224, 224).to(device)
print(model2(x).shape)

#do not use softmax in arch for training with cross entropy loss

#softmax
def conv_to_prob(output): #output in shape [1, num_classes]
  prob = torch.exp(output)
  return prob/torch.sum(prob)

print(conv_to_prob(model2(x)).shape)

from google.colab import drive
drive.mount('/content/drive/')

from pycocotools.coco import COCO

ann_file = "/content/drive/MyDrive/AISGProject_VGG_Coco/VGG/coco/annotations/instances_train2017.json"
coco = COCO(ann_file)

import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Get image IDs for a specific category (e.g., 'cat')
cat_ids = coco.getCatIds() #catNms=['cat'])  # Get category ID for "cat"
img_ids = coco.getImgIds(catIds=cat_ids)  # Get image IDs

all_img_ids = []
for cat_id in cat_ids:
    img_ids = coco.getImgIds(catIds=[cat_id])  # Get image IDs for category
    img_info = coco.loadImgs(img_ids)[0]
    all_img_ids.extend(img_ids)  # Add to list

# Remove duplicates
all_img_ids = list(set(all_img_ids))

# Load one image's metadata
img_info = coco.loadImgs(all_img_ids[100])[0]
img_url = img_info['coco_url']  # Get image URL

response = requests.get(img_url)
og_image = Image.open(BytesIO(response.content))
image = og_image.resize((224, 224))

# Display image
plt.imshow(image)
plt.axis("off")
plt.show()

print(f"Image URL: {img_url}", image.size)

ann_ids = coco.getAnnIds(imgIds=img_info['id'])
annotations = coco.loadAnns(ann_ids)

cat_id = annotations[0]['category_id']  # Get category ID from annotation
category = coco.loadCats(cat_id)[0]['name']  # Load category name

print(f"Category Name: {category}")

normalize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG16 pre-trained normalization
])

image_tensor = normalize_transform(image).unsqueeze(0)  # Final input for VGG16

print(image_tensor.shape)  # torch.Size([1, 3, 224, 224])

print(model2(image_tensor))
