from dataclasses import dataclass
from .model_config import ModelConfig


class Resnet101Config(ModelConfig):
  def __init__(self, config_path: str):
    super().__init__(config_path)
    
    self.load_configs()
    self.valid_data_root_path()
    self.init_project_dirs()
  
  def load_configs(self):
    super().load_configs()
    if self.train_config.pretrained == None:
      self.train_config.pretrained = 'DEFAULT'
  