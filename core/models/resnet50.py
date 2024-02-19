from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from ..utils import collate_fn, train_one_epoch, evaluate
from ..datasets import CocoClassificationDataset
from torch.utils.tensorboard import SummaryWriter
from typing import Type, Callable


class ResNet50(nn.Module):
	def __init__(self, weights: str):
		super(ResNet50, self).__init__()
		
		if weights == 'IMAGENET1K_V2':
			pretrained_weights = ResNet50_Weights.IMAGENET1K_V2
		elif weights == 'IMAGENET1K_V1':
			pretrained_weights = ResNet50_Weights.IMAGENET1K_V1
		else:
			raise ValueError(f'Invalid weights name: {weights}')

		self.model = resnet50(weights=pretrained_weights)


	def train(
		self, train_img_path: str, val_img_path: str, train_ann_path: str, val_ann_path: str, epochs: int,
		lr_scheduler: Type[LRScheduler], optimizer: Type[Optimizer], loss_fn: Callable,
		batch_size: int, num_workers: int, device: Type[torch.device]
  ):
		writer = SummaryWriter()

		self.model.to(device)
		
		train_dataset = CocoClassificationDataset(root=train_img_path, annFile=train_ann_path)
		train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
		
		val_dataset = CocoClassificationDataset(root=val_img_path, annFile=val_ann_path)
		val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

		for epoch in range(epochs):
			# TODO: Scaler 추가: GradScaler
			epoch_loss, train_accuracy = train_one_epoch(self.model, optimizer, loss_fn, train_data_loader, device, epoch, print_freq=10, scaler=None)
			val_loss, val_accuracy = evaluate(self.model, val_data_loader, device)

			lr_scheduler.step()

			writer.add_scalar('Loss/Train', epoch_loss, epoch)
			writer.add_scalar('Loss/Validation', val_loss, epoch)
			writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
			writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
			writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
				
		writer.close()
