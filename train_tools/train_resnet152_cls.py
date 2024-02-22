import argparse
from core.models import ResNet152Cls
from core.configs import Resnet50Config
import json
import numpy as np
import random
import torch


def resnet152_cls(cfg: Resnet50Config):
  random.seed(cfg.train_config.seed)
  np.random.seed(cfg.train_config.seed)
  torch.manual_seed(cfg.train_config.seed)
  torch.cuda.manual_seed_all(cfg.train_config.seed)
  # if deterministic:
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  
  model = ResNet152Cls(weights=cfg.train_config.pretrained, num_classes=7)
  _, tensorboard_name = cfg.get_output_path("train")
  model.train(
    tensorboard_name=tensorboard_name,
    train_img_path=cfg.get_img_dir_path("train"),
    val_img_path=cfg.get_img_dir_path("val"),
    train_ann_path=cfg.get_ann_file_path("train"),
    val_ann_path=cfg.get_ann_file_path("val"),
    epochs=cfg.train_config.epochs,
    lr_scheduler=cfg.lr_scheduler,
    lr_scheduler_params=cfg.train_config.lr_scheduler_params,
    optimizer=cfg.optimizer,
    optimizer_params=cfg.train_config.optimizer_params,
    loss_fn=cfg.loss_fn,
    batch_size=cfg.train_config.batch,
    num_workers=cfg.train_config.num_workers,
    device=cfg.device
  )

  with open(f"{tensorboard_name}/configs.json", "w", encoding="cp949") as f:
    json.dump(cfg.dict(), f)
  
  model.test(
    tensorboard_name=tensorboard_name,
    test_img_path=cfg.get_img_dir_path("test"),
    test_ann_path=cfg.get_ann_file_path("test"),
    batch_size=cfg.train_config.batch,
    num_workers=cfg.train_config.num_workers,
    device=cfg.device,
    criterion=cfg.loss_fn
  )

# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
  
#   parser.add_argument("--config-path", type=str, default="../configs/project_1_yolo.yaml")
  
#   config = parser.parse_args()
#   resnet50_cls(config)