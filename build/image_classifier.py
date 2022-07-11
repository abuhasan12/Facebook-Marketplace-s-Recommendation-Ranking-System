import torch
import torch.nn as nn
from torchvision import models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 13, decoder: dict = None, fc = None, device = "cpu"):
        super(ImageClassifier, self).__init__()
        image_model = models.resnet50(pretrained=True)
        in_features = image_model.fc.in_features
        if fc:
            image_model.fc = fc
        else:
            image_model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        self.main = nn.Sequential(image_model).to(device)
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x
    
    def predict_proba(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)

    def predict_classes(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder[int(torch.argmax(x, dim=1))]