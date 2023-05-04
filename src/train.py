from flax import struct 
from flax.training import train_state 
from clu import metrics
import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm
import wandb


from src.config import config
from src.models import MPCTransformer
from src.input_pipeline import create_dataloaders


@struct.dataclass
class Metrics(metrics.Collection):
  # accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
  metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum, train=True):
  """Creates an initial `TrainState`."""
  
  # initialize parameters by passing a template image
  template_shape = (1,) + config['data']['shape']
  params = module.init(rng, jnp.ones(template_shape), train=train)['params'] 
  
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())


@jax.jit
def train_step(state, batch, train=True):
  """Train for a single step."""
  def loss_fn(params):
    dropout_rng = jax.random.PRNGKey(1)
    
    pred = state.apply_fn({'params': params},
                            rngs={'dropout': dropout_rng},
                            inputs=batch['inputs'], 
                            train=train)
        
    loss = optax.l2_loss(pred, batch['targets']).mean()
    return loss
  
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  
  state = state.apply_gradients(grads=grads)
  return state


@jax.jit
def compute_metrics(*, state, batch, train=True):
  dropout_rng = jax.random.PRNGKey(1)
  
  pred = state.apply_fn({'params': state.params},
                          rngs={'dropout': dropout_rng},
                          inputs=batch['inputs'], 
                          train=train)
  
  loss = optax.l2_loss(pred, batch['targets']).mean()
  
  metric_updates = state.metrics.single_from_model_output(loss=loss)
  metrics = state.metrics.merge(metric_updates)
  
  state = state.replace(metrics=metrics)
  return state


def train(model, num_epochs, train_loader, val_loader, use_wandb=False):
  init_rng = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
  state = create_train_state(model, 
                             init_rng, 
                             config['training']['learning_rate'], 
                             config['training']['momentum'])
  del init_rng
  
  metrics_history = {'train_loss': [],
                     'val_loss': []}
  
  epoch_iterator = range(num_epochs)
  
  for epoch in epoch_iterator:
    
    # TRAINING
    train_iterator = train_loader
    # train_iterator = tqdm(train_loader)
    for grids, commands in train_iterator:   
      # total_steps += 1   
      
      batch = {'inputs': grids.numpy(), 'targets': commands.numpy()}
      
      state = train_step(state, batch)
      state = compute_metrics(state=state, batch=batch)
      
      # train_iterator.set_postfix(train_loss=state.metrics.loss.compute_value().value)

    for metric, value in state.metrics.compute().items():  # compute metrics
      metrics_history[f'train_{metric}'].append(value)     # record metrics
    state = state.replace(metrics=state.metrics.empty())   # reset metrics
    
    test_state = state
    
    # EVALUATION
    val_iterator = val_loader
    # val_iterator = tqdm(val_loader)
    for grids, commands in val_iterator:   
      # total_steps += 1   
      
      batch = {'inputs': grids.numpy(), 'targets': commands.numpy()}
      test_state = compute_metrics(state=test_state, batch=batch)
      
      # val_iterator.set_postfix(val_loss=test_state.metrics.loss.compute_value().value)
      
    for metric, value in test_state.metrics.compute().items():
      metrics_history[f'val_{metric}'].append(value)
      
    test_state = state.replace(metrics=test_state.metrics.empty())
  
    print(f"Epoch: {epoch+1}, "
          f"Train Loss: {metrics_history['train_loss'][-1]:.6f}, "
          f"Val Loss: {metrics_history['val_loss'][-1]:.6f}, ")  

    wandb.log({'Train Loss': metrics_history['train_loss'][-1],
               'Validation Loss': metrics_history['val_loss'][-1]})
          
  return state, metrics_history
