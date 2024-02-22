import torch


def train_one_epoch(dataloader, model, criterion, optimizer, device):
  model.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)
    y -= 1 # due to coco format, y values are in range of 1 to 7, where they should be in between 0 to 6.
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()*x.size(0)
    _, predicted = pred.max(1)
    total += y.size(0)
    correct += predicted.eq(y).sum().item()
  
  epoch_loss = train_loss/total
  epoch_acc = correct/total
  return epoch_loss, epoch_acc

def evaluate(dataloader, model, criterion, device):
  model.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (x, y) in enumerate(dataloader):
      x, y = x.to(device), y.to(device)
      y -= 1 # due to coco format, y values are in range of 1 to 7, where they should be in between 0 to 6.
      
      pred = model(x)
      loss = criterion(pred, y)

      test_loss += loss.item()*x.size(0)
      _, predicted = pred.max(1)
      total += y.size(0)
      correct += predicted.eq(y).sum().item()
    
    epoch_loss = test_loss/total
    epoch_acc = correct/total
  return epoch_loss, epoch_acc