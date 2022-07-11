import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ProductsImDataset(Dataset):
    def __init__(self, transform = None):
        super().__init__()
        self.data = pd.read_pickle('product_images_df.pkl')
        self.transform = transform
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        features = torch.tensor(np.moveaxis(item[0], 2, 0))/255
        if self.transform:
            features = self.transform(features)
        label = torch.tensor(int(item[1]))
        return (features, label)
    
    def __len__(self):
        return len(self.data)
    
if __name__ == '__main__':
    im_dataset = ProductsImDataset()
    print(len(im_dataset))
    print(im_dataset[12])

    loader = DataLoader(im_dataset, batch_size=7, shuffle=True)

    for batch in loader:
        print(batch)
        features, labels = batch
        print(features.shape)
        print(labels.shape)
        break