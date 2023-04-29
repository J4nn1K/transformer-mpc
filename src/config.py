config = {
  'data': {
    'path': '../data/robot_field_data.pt',
    'dt': 0.025,
  },
  'control': {
    'horizon': 10,
    'dt': 0.1
  },
  'training': {
    'train_ratio': 0.8,
    'batch_size': 1
  }
}