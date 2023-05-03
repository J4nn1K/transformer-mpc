from torch.utils.data import Dataset, DataLoader, random_split
import torch
import jax
import numpy as np

from src.config import config


class FieldDataset(Dataset):
  ''' Dataset class specific to saved format.'''
  def __init__(self, file_dir=config['data']['path']):      
    self.horizon = config['model']['solver']['horizon']
    
    data = np.load(file_dir)
    
    if config['data']['type'] == 'map':
      self.inputs = data['maps']
    elif config['data']['type'] == 'rgb':
      self.inputs = data['color_images']
    elif config['data']['type'] == 'depth':
      self.inputs = data['depth_images']
    else:
      print('unkown data type')
    
    self.commands = data['cmd_vels']

  def __len__(self):
    return len(self.commands) - self.horizon
  
  def __getitem__(self, idx):
    input = self.inputs[idx]
    target = self.commands[idx:idx+self.horizon]

    return input, target
  
def create_dataset():
  return FieldDataset()

def create_dataset_split():
  dataset = create_dataset()
    
  train_size = int((config['training']['train_ratio']) * len(dataset))
  val_size = len(dataset) - train_size

  return random_split(dataset, [train_size, val_size])


def create_dataloaders():
  train_dataset, val_dataset = create_dataset_split()
  
  train_loader = DataLoader(train_dataset, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=True)
  val_loader = DataLoader(val_dataset, 
                          batch_size=config['training']['batch_size'],
                          shuffle=False)
  
  return train_loader, val_loader