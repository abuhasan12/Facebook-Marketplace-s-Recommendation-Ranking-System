import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, num_classes: int = 13, input_size: int = 768, decoder: dict = None, max_length: int = 100, fc = None):
        super(TextClassifier, self).__init__()
        device = "cpu"
        self.max_length = max_length
        if not fc:
            if self.max_length == 100:
                self.fc = nn.Sequential(
                    nn.Linear(384, 128),
                    nn.ReLU(),
                    nn.Linear (128, num_classes)
                )
            elif self.max_length == 50:
                self.fc = nn.Sequential(
                    nn.Linear(192, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
        else:
            self.fc = fc
        self.main = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            self.fc
        ).to(device)
        self.decoder = decoder

    def forward(self, inp):
        print(inp.size())
        x = self.main(inp)
        return x

    def predict(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return x
    
    def predict_proba(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return torch.softmax(x, dim=1)

    def predict_classes(self, inp):
        with torch.no_grad():
            x = self.forward(inp)
            return self.decoder[int(torch.argmax(x, dim=1))]