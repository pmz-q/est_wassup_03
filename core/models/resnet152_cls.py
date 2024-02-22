import copy
from ..datasets import CocoClassificationDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet152, ResNet152_Weights
from torch import nn
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from typing import Type, Callable
from .utils import train_one_epoch, evaluate


class ResNet152Cls(nn.Module):
  def __init__(self, weights: str, num_classes: int):
    super(ResNet152Cls, self).__init__()
    
    if weights == 'DEFAULT':
      pretrained_weights = ResNet152_Weights.DEFAULT
    else:
      raise ValueError(f'Invalid weights name: {weights}')

    self.model = resnet152(weights=pretrained_weights)

    # Freezing
    # ref doc: https://pytorch.org/vision/stable/models.html
    for param in self.model.parameters():
      param.requires_grad = False
    
    # fc layer
    fc_in_features = self.model.fc.in_features
    self.model.fc = nn.Linear(in_features=fc_in_features, out_features=num_classes)

  def train(
    self, tensorboard_name: str, train_img_path: str, val_img_path: str, train_ann_path: str, val_ann_path: str, epochs: int,
    lr_scheduler: Type[LRScheduler], lr_scheduler_params: dict, optimizer: Type[Optimizer], optimizer_params: dict,
    loss_fn: Callable, batch_size: int, num_workers: int, device: str
  ):
    writer = SummaryWriter(tensorboard_name)

    self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
    self.scheduler = lr_scheduler(optimizer=self.optimizer, **lr_scheduler_params)

    self.model.to(device)
    
    train_dataset = CocoClassificationDataset(src_root_path=train_img_path, ann_path=train_ann_path)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_dataset = CocoClassificationDataset(src_root_path=val_img_path, ann_path=val_ann_path)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    best_acc = 0
    pbar = trange(epochs)
    for epoch in pbar:
      # TODO: Scaler 추가: GradScaler
      # epoch_loss, train_accuracy = train_one_epoch(self.model, self.optimizer, loss_fn, train_data_loader, device, epoch, print_freq=10, scaler=None)
      trn_loss, trn_acc = train_one_epoch(train_data_loader, self.model, loss_fn, self.optimizer, device)
      val_loss, val_acc = evaluate(val_data_loader, self.model, loss_fn, device)

      self.scheduler.step()
      
      # Save model
      if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(self.model.state_dict())
        torch.save(best_model_wts, f"{tensorboard_name}/resnet50_cls_best.pt")
      
      writer.add_scalar('Loss/Train', trn_loss, epoch)
      writer.add_scalar('Loss/Validation', val_loss, epoch)
      writer.add_scalar('Accuracy/Train', trn_acc, epoch)
      writer.add_scalar('Accuracy/Validation', val_acc, epoch)
      writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
      
      pbar.set_postfix({'trn_loss': trn_loss, 'trn_acc': trn_acc, 'val_loss': val_loss, 'val_acc': val_acc})
      
    writer.close()
    torch.save(self.model, f"{tensorboard_name}/resnet50_cls.pt")
  
  def test(
    self, tensorboard_name:str, test_img_path:str, test_ann_path:str, batch_size:int, num_workers:int, device: str, criterion: Callable
  ):
    test_dataset = CocoClassificationDataset(src_root_path=test_img_path, ann_path=test_ann_path)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    self.model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    categories = ["anger", "anxiety", "embarrass", "happy", "normal", "pain", "sad"]
    
    list_y = []
    list_pred = []
    pbar = tqdm(test_data_loader)
    with torch.no_grad():
      for x, y in pbar:
        x, y = x.to(device), y.to(device)
        y -= 1 # due to coco format, y values are in range of 1 to 7, where they should be in between 0 to 6.
        
        pred = self.model(x)
        loss = criterion(pred, y)

        test_loss += loss.item()*x.size(0)
        _, predicted = pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        # Confusion matrix
        list_y.extend([categories[true] for true in y.to("cpu")])
        list_pred.extend([categories[p] for p in predicted.to("cpu")])
      
      epoch_loss = test_loss/total
      epoch_acc = correct/total
      print("loss:", epoch_loss, "acc:", epoch_acc)
    
    cm = confusion_matrix(list_y, list_pred, labels=categories, normalize="true")
    plt.figure(figsize=(20,15))
    plt.rcParams.update({"font.size": 18})
    plt.title(f"Confusion Matrix Normalized - TEST ACC: {epoch_acc}", fontdict={"size": 24})
    heatmap = sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    heatmap.yaxis.set_ticklabels(categories, rotation=0, ha="right", fontsize=18)
    heatmap.xaxis.set_ticklabels(categories, rotation=45, ha="right", fontsize=18)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{tensorboard_name}/confusion_matrix_normalized.png")
    
    cm = confusion_matrix(list_y, list_pred, labels=categories)
    plt.figure(figsize=(20,15))
    plt.title(f"Confusion Matrix - TEST ACC: {epoch_acc}", fontdict={"size": 24})
    heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    heatmap.yaxis.set_ticklabels(categories, rotation=0, ha="right", fontsize=18)
    heatmap.xaxis.set_ticklabels(categories, rotation=45, ha="right", fontsize=18)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{tensorboard_name}/confusion_matrix.png")
    
    return epoch_loss, epoch_acc
