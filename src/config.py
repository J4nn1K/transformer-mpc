config = {
  'data': {
    'path': 'data/obstacles.npz',
    'type': 'map',          # ['rgb', 'depth', 'map']
    'shape': (100, 100, 1)  # H x W x C
  },
  'model': {
    'patches': (5,5),
    'transformer': {
      'num_layers': 3,
      'mlp_dim': 64,
      'num_heads': 1,
      'dropout_rate': 0.1,
      'attention_dropout_rate': 0.1,
    },
    'solver': {
      'dt': 0.1,
      'horizon': 10,
      'cost_weights': {
        'learned': 1.0,
        'reference' : 1e0,
      },
      'u_des': [0.4, 0.0, 0.0] 
    },
    'hidden_size': 42,
    'num_output': 42,
  }, 
  'training': {
    'train_ratio': 0.8,
    'batch_size': 1,
    'learning_rate': 0.01,
    'num_epochs': 50,
    'momentum' : 0.9,
  }
}