from flax.training import train_state, checkpoints
import jax

from src.config import config
from src.input_pipeline import create_dataloaders
from src.models import MPCTransformer
from src.train import train
import wandb

wandb.init(project="transformer-mpc", config=config)

train_loader, val_loader = create_dataloaders()

model_config = config['model']
model = MPCTransformer(**model_config)

num_epochs = config['training']['num_epochs'] 
state, metrics_history = train(model, num_epochs, train_loader, val_loader)

CKPT_DIR = config['training']['checkpoint_dir']

checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=0)
restored_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)

assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), state.params, restored_state.params))