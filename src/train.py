from src.config import config
from src.models import MPCTransformer

model_config = config['model']

model = MPCTransformer(**model_config)

