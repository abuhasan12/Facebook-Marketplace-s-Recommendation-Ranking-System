import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset

class TextDatasetBert(Dataset):
  def __init__(self, root_dir: str, labels_level: int = 0, max_length: int = 50):
    self.root_dir = root_dir
    if not os.path.exists(self.root_dir):
      raise FileNotFoundError(f"The file {self.root_dir} does not exist")
    products = pd.read_csv(self.root_dir, lineterminator='\n', index_col=[0])
    products['category'] = products['category'].str.split('/', expand=True)[labels_level]
    self.labels = products['category'].to_list()
    self.descriptions = products['product_description'].to_list()
    self.classes = sorted(list(set(self.labels)))
    self.num_classes = len(self.classes)
    self.encoder = {y: x for x,y in enumerate(self.classes)}
    self.decoder = {x: y for x,y in enumerate(self.classes)}
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    self.max_length = max_length

  def __getitem__(self, index):
    label = self.labels[index]
    label = self.encoder[label]
    label = torch.as_tensor(label)
    description = self.descriptions[index]
    encoded = self.tokenizer.batch_encode_plus([description], max_length=self.max_length, padding='max_length', truncation=True)
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    with torch.no_grad():
      description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
    description = description.squeeze(0)
    return description, label
  
  def __len__(self):
    return len(self.labels)