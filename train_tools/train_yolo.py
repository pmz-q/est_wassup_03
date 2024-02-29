import argparse
from core.configs import YOLOConfig
from os.path import exists
import ultralytics
from ultralytics import YOLO


def yolo_det(
  cfg: YOLOConfig
):
  if not exists(cfg.data_yaml_path):
    raise FileNotFoundError(f"For training model [yolo] task [detection], data root directory must contain [yolo-dataset.yaml]")
  
  ultralytics.checks()
  train_config = cfg.train_config
  if train_config.pretrained == None:
    train_config.pretrained = "yolov8n.pt"
  output_project, output_name = cfg.get_output_path("train")
  
  model = YOLO(train_config.pretrained)
  model.train(
    data=cfg.data_yaml_path,
    epochs=train_config.epochs,
    device=cfg.device,
    project=output_project,
    name=output_name,
    seed=train_config.seed,
    plots=True,
    save=True,
    batch=train_config.batch,
    workers=train_config.num_workers,
    dropout=train_config.dropout,
    optimizer=cfg.optimizer,
    **train_config.optimizer_params,
    **cfg.lr_scheduler,
    imgsz=train_config.imgsz
  )  # train the model


# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
  
#   parser.add_argument("--config-path", type=str, default="../configs/project_2_yolo.yaml")
  
#   config = parser.parse_args()
#   yolo_det(config)