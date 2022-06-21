import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from image_torch_dataset import ProductsImDataset, ProductsImTrainDataset, ProductsImTestDataset

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

    # train_transform = transforms.Compose([
    #     transforms.Grayscale()
    # ])

    # transform = transforms.Grayscale()

    train_dataset = ProductsImDataset()
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model = models.resnet50(pretrained=True)
    no_features = model.fc.in_features
    model.fc = torch.nn.Linear(no_features, 13)
    train(model)