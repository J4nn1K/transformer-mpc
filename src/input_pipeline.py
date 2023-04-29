from torch.utils.data import Dataset, DataLoader, random_split
import torch
import jax

from src.config import config


class FieldDataset(Dataset):
  ''' Dataset class specific to saved format.'''
  def __init__(self, file_dir=config['data']['path']):      
    every_n = int(config['model']['solver']['dt']/config['data']['dt'])
    self.horizon = config['model']['solver']['horizon']

    grids, commands = torch.load(file_dir)
    
    self.grids = grids[::every_n]
    self.commands = commands[::every_n][:,[0,1,5]]  # grab u_x, u_y, u_w

  def __len__(self):
    return len(self.grids) - self.horizon
  
  def __getitem__(self, idx):
    grid = jax.numpy.expand_dims(self.grids.numpy(), axis=-1)
    commands = self.commands[idx:idx+self.horizon].numpy()

    return grid, commands


def create_datasets():
  dataset = FieldDataset()
  
  train_size = int((config['training']['train_ratio']) * len(dataset))
  val_size = len(dataset) - train_size

  return random_split(dataset, [train_size, val_size])


def create_dataloaders():
  train_dataset, val_dataset = create_datasets()
  
  train_loader = DataLoader(train_dataset, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=True)
  val_loader = DataLoader(val_dataset, 
                          batch_size=config['training']['batch_size'],
                          shuffle=False)
  
  return train_loader, val_loader