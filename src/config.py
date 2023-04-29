config = {
  'data': {
    'path': '../data/impossible_obstacle.pt',
    'dt': 0.025,
    'shape': (100, 100, 1) # HxWxC
  },
  'model': {
    'patches': (4,4),
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
    },
    'hidden_size': 42,
    'num_output': 42,
    'representation_size': 64,
  }, 
  'training': {
    'train_ratio': 0.8,
    'batch_size': 1,
    'learning_rate': 0.01,
    'momentum' : 0.9,
  }
}