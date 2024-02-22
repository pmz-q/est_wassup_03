from dataclasses import dataclass
from .model_config import ModelConfig
import torch
from typing import Optional, Union, Literal


YOLO_OPTIM_MAPPER = {
  "adamw": "AdamW",
  "adam": "Adam",
  "sgd": "SGD"
}

@dataclass
class YOLOLRSchedulerParams:
  lr0: float
  lrf: float

@dataclass
class YOLOTrainConfig:
  epochs: int
  batch: int
  imgsz: int
  device: Union[int, str]
  optimizer: Literal["adamw", "adam", "sgd"]
  optimizer_params: dict
  seed: int
  dropout: float
  lr_scheduler: Optional[Literal["coslr", "steplr"]]=None
  lr_scheduler_params: Optional[dict]=None
  pretrained: Optional[str]=None
  conf: Optional[float]=None
  cls: Optional[float]=None
  num_workers: Optional[int]=4

class YOLOConfig(ModelConfig):
  def __init__(self, config_path: str):
    super().__init__(config_path)
    self._train_config_class: object = YOLOTrainConfig
    self._train_config: YOLOTrainConfig = None
    
    self.load_configs()
    self.valid_data_root_path()
    self.init_project_dirs()
  
  @property
  def data_yaml_path(self):
    return f"{self.data_root_path}/yolo-dataset.yaml"
  
  @property
  def device(self) -> str:
    if torch.cuda.is_available():
      return self.train_config.device
    else:
      return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
  
  @property
  def optimizer(self):
    return YOLO_OPTIM_MAPPER[self.train_config.optimizer]
  
  @property
  def lr_scheduler(self):
    return {
      "cos_lr": self.train_config.lr_scheduler == "coslr",
      "lrf": self.train_config.lr_scheduler_params["lrf"]
    }
  