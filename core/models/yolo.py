import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from ultralytics.engine.model import Model
from ultralytics.engine.trainer import BaseTrainer
from typing import Type, Callable, Literal


class YOLO(Model):
	"""
  Custom YOLO class
	"""
	def __init__(self, model:str=None, task:Literal["classify", "detect"]="classify", verbose:bool=False):
		if model == None:
			model= "yolov8n.pt"
			if task == "classify":
				model = "yolov8n-cls.pt"
    
		super().__init__(model=model, task=task, verbose=verbose)

	def train(
		self, train_img_path: str, val_img_path: str, train_ann_path: str, val_ann_path: str, epochs: int,
		lr_scheduler: Type[LRScheduler], optimizer: Type[Optimizer], loss_fn: Callable,
		batch_size: int, num_workers: int, device: Type[torch.device]
  ):
		self.trainer: Type[BaseTrainer] = self._smart_load("trainer")
		self.trainer()