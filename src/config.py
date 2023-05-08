config = {
  'data': {
    'path': 'data/obstacles_test_scenario.npz',
    'type': 'map',          # ['rgb', 'depth', 'map']
    'shape': (100, 100, 1)  # H x W x C
  },
  'model': {
    # 'pooling': (2,2),
    'patches': (5,5),
    'mlp_head': True,
    'num_output': 46, # (6x6, 6, 4)
    'hidden_size': 768,
    'transformer': {
      'num_layers': 12,
      'mlp_dim': 3072,
      'num_heads': 12,
      'dropout_rate': 0.1,
      'attention_dropout_rate': 0.1,
    },
    'solver': {
      'dt': 0.1,
      'horizon': 10,
      # 'cost_weights': {
      #   'learned': 1.0,
      #   'reference' : 1.0,
      # },
      # 'u_des': [0.4, 0.0, 0.0] 
    },
  }, 
  'training': {
    'train_ratio': 0.8,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 40,
    'momentum' : 0.9,
    'checkpoint_dir': 'checkpoints/map/'
  }
}