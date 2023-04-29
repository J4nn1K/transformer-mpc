from torch.utils.data import Dataset, DataLoader, random_split
import torch

from src.config import config

class FieldDataset(Dataset):
    def __init__(self, file_dir=config['data']['path']):      
        every_n = int(config['control']['dt']/config['data']['dt'])
        self.horizon = config['control']['horizon']
  
        grids, commands = torch.load(file_dir)
        
        self.grids = grids[::every_n]
        self.commands = commands[::every_n][:,[0,1,5]]  # grab u_x, u_y, u_w

    def __len__(self):
        return len(self.grids) - self.horizon

    def __getitem__(self, idx):
        grid = self.grids[idx]
        commands = self.commands[idx:idx+self.horizon]
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