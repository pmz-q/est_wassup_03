from dataclasses import dataclass
from os import makedirs, listdir
from os.path import exists, isfile
import torch
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from typing import Optional, Union, Literal, Type, Callable
from ..utils import get_root_path
import yaml


LR_SCHEDULER_MAPPER = {
  "coslr": CosineAnnealingLR,
  "steplr": StepLR 
}

OPTIMIZER_MAPPER = {
  "sgd": SGD,
  "adamw": AdamW,
  "adam": Adam
}

LOSS_FN_MAPPER = {
  "cross_entropy": cross_entropy
}

@dataclass
class ConfigInfo:
  project_name:str
  model_name: str
  model_task: str
  data_root_dir: str

@dataclass
class TrainConfig:
  epochs: int
  lr_scheduler: Optional[Literal["coslr", "steplr"]]
  lr_scheduler_params: Optional[dict]
  batch: int
  # TODO: img size train 에 추가하기
  imgsz: int
  device: Union[int, tuple]
  optimizer: Literal["adamw", "adam", "sgd"]
  optimizer_params: dict
  pretrained: Optional[str]
  seed: int
  dropout: float
  conf: Optional[float]
  loss_fn: Callable=cross_entropy
  num_workers: Optional[int]=4

class ModelConfig:
  def __init__(
    self,
    config_path: str
  ):
    self.project_name: str = None
    self.model_name: str = None
    self.model_task: str = None
    self.data_root_dir: str = None
    
    self._optimizer: Type[Optimizer] = None
    self._lr_scheduler: Type[LRScheduler] = None
    self._loss_fn: Callable = None
    
    self._config_class: object = TrainConfig
    self._train_config: TrainConfig = None
    
    self.config_path = config_path
  
  def load_configs(self):
    if not exists(self.config_path) or not isfile(self.config_path):
      raise FileNotFoundError(f"Invalid config path: [{self.config_path}]")
    
    with open(self.config_path, encoding="cp949") as f:
      config = yaml.safe_load(f)
    
    info = ConfigInfo(**config['info'])
    self.project_name = info.project_name
    self.model_name = info.model_name
    self.model_task = info.model_task
    self.data_root_path = info.data_root_dir
    
    self._train_config = self._config_class(**config['train'])
  
  def valid_data_root_path(self):
    if not exists(self.data_root_path) or isfile(self.data_root_path):
      raise FileNotFoundError(f"Invalid data_root_path: [{self.data_root_path}]. Directory not found.")
    
    listdir_data_root = listdir(self.data_root_path)
    if "images" not in listdir_data_root or "labels" not in listdir_data_root:
      raise FileNotFoundError(f"Data root directory must contain both [images] and [labels] directories.")
    
    if self.model_name == "yolo" and self.model_task == "detection" and "yolo-dataset.yaml" not in listdir_data_root:
      raise FileNotFoundError(f"For running model [yolo] task [detection], data root directory must contain [yolo-dataset.yaml]")
  
  def init_project_dirs(self):
    root_dir = get_root_path()
    results_dir = f"{root_dir}/results"
    makedirs(f"{results_dir}/{self.project_name}/inference", exist_ok=True)
    makedirs(f"{results_dir}/{self.project_name}/train", exist_ok=True)
    makedirs(f"{results_dir}/{self.project_name}/eval_test", exist_ok=True)
  
  @property
  def device(self) -> str:
    if torch.cuda.is_available():
      device = self.train_config.device
      if type(device) == int:
        return f"cuda:{device}"
      else:
        return f"cuda:{','.join(device)}"
    else:
      return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
  
  @property
  def train_config(self):
    return self._train_config
  
  @property
  def optimizer(self):
    if self._optimizer == None:
      self._optimizer = OPTIMIZER_MAPPER[self.train_config.optimizer](**self.train_config.optimizer_params)
    return self._optimizer
  
  @property
  def lr_scheduler(self):
    if self._lr_scheduler == None:
      self._lr_scheduler = LR_SCHEDULER_MAPPER[self.train_config.lr_scheduler](
        optimizer=self.optimizer,
        **self.train_config.lr_scheduler_params
      )
    return self._lr_scheduler
  
  @property
  def loss_fn(self):
    if self._loss_fn == None:
      self._loss_fn = self.train_config.loss_fn
    return self._loss_fn
  
  def get_ann_file_path(self, mode: Literal["train", "val", "test"]):
    return f"{self.data_root_path}/labels/{mode}/annotation.json"
  
  def get_img_dir_path(self, mode: Literal["train", "val", "test"]=None):
    if mode == None:
      return f"{self.data_root_path}/images"
    return f"{self.data_root_path}/images/{mode}"

  def get_output_path(self, run_type: Literal["train", "inference", "eval"]="train"):
    """
    est_wassup_03/
      results/
        project_name/
          inference/
            1/
            2/
          train/
          eval/
    Returns:
      ( yolo_project_name, yolo_name )
    """
    num = 1
    project_name = f"{get_root_path()}/results/{self.project_name}/{run_type}"
    while exists(f"{project_name}/{num}"):
      num += 1
    return project_name, f"{project_name}/{num}"
