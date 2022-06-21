from tokenize import Double
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import seaborn as sns

class ProductsImDataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()
        self.data = pd.read_pickle('product_images_df.pkl')
        self.transform = transform
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        features = torch.moveaxis(torch.tensor(item[0]), 2, 0)
        features = features/255
        if self.transform:
            features = self.transform(features)
        label = torch.tensor(int(item[1]))
        return (features, label)
    
    def __len__(self):
        return len(self.data)

class ProductsImTrainDataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()
        self.data = pd.read_pickle('product_training_df.pkl')
        self.transform = transform
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        features = torch.moveaxis(torch.tensor(item[0]), 2, 0)
        features = features/255
        if self.transform:
            features = self.transform(features)
        label = torch.tensor(int(item[1]))
        return (features, label)
    
    def __len__(self):
        return len(self.data)

class ProductsImTestDataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()
        self.data = pd.read_pickle('product_testing_df.pkl')
        self.transform = transform
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        features = torch.moveaxis(torch.tensor(item[0]), 2, 0)
        features = features/255
        if self.transform:
            features = self.transform(features)
        label = torch.tensor(int(item[1]))
        return (features, label)
    
    def __len__(self):
        return len(self.data)
    
if __name__ == '__main__':
    # transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.ToTensor()
    # ])

    transform = transforms.Grayscale()

    im_dataset = ProductsImDataset(transform=transform)
    print(len(im_dataset))
    print(im_dataset[12])

    loader = DataLoader(im_dataset, batch_size=7, shuffle=True)

    for batch in loader:
        print(batch)
        features, labels = batch
        print(features.shape)
        print(labels.shape)
        break