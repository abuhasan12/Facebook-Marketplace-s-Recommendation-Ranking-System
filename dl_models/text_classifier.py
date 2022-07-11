import torch
import torch.nn as nn

class TextClassifier(nn.Module):
  def __init__(self, input_size: int=768, num_classes: int = 13, decoder: dict = None):
    super(TextClassifier, self).__init__()
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
        nn.Linear(192, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )
    self.decoder = decoder

  def forward(self, inp):
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