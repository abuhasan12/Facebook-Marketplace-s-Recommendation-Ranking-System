import os
import pandas as pd
from transformers import BertTokenizer, BertModel
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageAndTextDataset(Dataset):
  def __init__(self, train_test: str, root_dir: str, labels_level: int = 0, max_length: int = 50, transform: transforms = None):
    self.train_test = train_test
    if self.train_test not in {'train', 'val'}:
      raise Exception("You can only create a 'train' or 'val' dataset.")
    self.root_dir = root_dir
    if not os.path.exists(self.root_dir):
      raise FileNotFoundError(f"The file {self.root_dir} does not exist")
    self.products = pd.read_csv(self.root_dir, lineterminator='\n', index_col=[0])
    self.products['category'] = self.products['category'].str.split('/', expand=True)[labels_level]
    self.labels = self.products['category'].to_list()
    self.descriptions = self.products['product_description'].to_list()
    self.classes = list(set(self.labels))
    self.num_classes = len(set(self.labels))
    self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
    self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    self.max_length = max_length
    self.files = self.products['id']
    self.transform = transform
    if transform is None:
      if self.train_test == 'train':
        self.transform = transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.4217, 0.3923, 0.3633], std=[0.3117, 0.2967, 0.2931])
                                             ]),
      else:
        self.transform = transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.4217, 0.3923, 0.3633], std=[0.3117, 0.2967, 0.2931])
                                             ])
  
  def __getitem__(self, index):
    label = self.labels[index]
    label = self.encoder[label]
    label = torch.as_tensor(label)
    image = Image.open(f'resized_images/{self.files[index]}.jpg')
    image = self.transform(image)
    description = self.descriptions[index]
    encoded = self.tokenizer.batch_encode_plus([description], max_length=self.max_length, padding='max_length', truncation=True)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    with torch.no_grad():
      description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
    description = description.squeeze(0)
    return description, image, label
  
  def __len__(self):
    return len(self.files)