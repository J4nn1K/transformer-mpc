config = {
  'data': {
    'path': '../data/impossible_obstacle.pt',
    'dt': 0.025,
    'shape': (100, 100, 1) # HxWxC
  },
  'model': {
    'patches': (4,4),
    'transformer': {
      'num_layers': 6,
      'mlp_dim': 512,
      'num_heads': 3,
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
    'momentum' : 0.9,
  }
}