from src.config import config
from src.input_pipeline import create_dataloaders
from src.models import MPCTransformer
from src.train import train


train_loader, val_loader = create_dataloaders()

model_config = config['model']
model = MPCTransformer(**model_config)

num_epochs = 20
state, metrics_history = train(model, num_epochs, train_loader, val_loader)

print(metrics_history)