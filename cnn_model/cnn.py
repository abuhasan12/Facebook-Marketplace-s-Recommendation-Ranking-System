import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from image_torch_dataset import ProductsImDataset

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(21632 , 1352),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1352, 338),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(338, 75),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(75, 13),
            torch.nn.Softmax()
        )
    
    def forward(self, X):
        return self.layers(X.float())

def train(model, epochs=10):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels)
            optimiser.zero_grad()
            loss.backward()
            print(loss.item())
            optimiser.step()

if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.Grayscale()
    ])

    dataset = ProductsImDataset(transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = CNN()
    train(model)