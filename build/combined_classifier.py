import torch
import torch.nn as nn
from torchvision import models
from text_classifier import TextClassifier
from image_classifier import ImageClassifier

class ImageAndTextModel(nn.Module):
    def __init__(self, num_classes: int = 13, input_size: int = 768, decoder: dict = None):
        super(ImageAndTextModel, self).__init__()
        device = "cpu"
        self.num_classes = num_classes
        self.input_size = input_size
        self.decoder = decoder

        text_classifier_fc = nn.Sequential(
            nn.Linear(192, 128)
        )
        self.text_classifier = TextClassifier(
            num_classes=self.num_classes,
            input_size=self.input_size,
            decoder=self.decoder,
            max_length=50,
            fc=text_classifier_fc
        ).to(device)

        image_model = models.resnet50(pretrained=True)
        in_features = image_model.fc.in_features
        image_classifier_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.image_classifier = ImageClassifier(
            decoder=decoder,
            fc=image_classifier_fc
        ).to(device)

        self.main = nn.Sequential(
            nn.Linear(256, self.num_classes)
        ).to(device)
        self.decoder = decoder

    def forward(self, text_inp, image_inp):
        text_features = self.text_classifier(text_inp)
        image_features = self.image_classifier(image_inp)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

    def predict(self, text_inp, image_inp):
        with torch.no_grad():
            x = self.forward(text_inp, image_inp)
            return x

    def predict_proba(self, text_inp, image_inp):
        with torch.no_grad():
            x = self.forward(text_inp, image_inp)
            return torch.softmax(x, dim=1)

    def predict_classes(self, text_inp, image_inp):
        with torch.no_grad():
            x = self.forward(text_inp, image_inp)
            return self.decoder[int(torch.argmax(x, dim=1))]