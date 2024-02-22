import argparse
from core.configs import YOLOConfig, ModelConfig, Resnet50Config
from train_tools import (
  yolo_det, yolo_cls,
  resnet50_cls,
  resnet101_cls,
  resnet152_cls
)
from infer_tools import (
  yolo_det as infer_yolo_det,
  yolo_cls as infer_yolo_cls,
  resnet50_cls as infer_resnet50_cls,
  resnet101_cls as infer_resnet101_cls,
  resnet152_cls as infer_resnet152_cls,
)
from typing import Dict, Type


def main(cfg):
  config_path = cfg.config_path
  config_type = cfg.config_type
  run_type = cfg.run_type
  
  config_mapper:Dict[str, Type[ModelConfig]] = {
    "yolo": YOLOConfig,
    "coco": Resnet50Config
  }
  
  train_mapper: Dict[str, function] = {
    "classification": {
      "yolo": yolo_cls,
      "resnet50": resnet50_cls,
      "resnet101": resnet101_cls,
      "resnet152": resnet152_cls
    },
    "detection": {
      "yolo": yolo_det
    }
  }
  
  infer_mapper: Dict[str, function] = {
    "detection": {
      "yolo": infer_yolo_det
    },
    "classification": {
      "yolo": infer_yolo_cls,
      "resnet50": infer_resnet50_cls,
      "resnet101": infer_resnet101_cls,
      "resnet152": infer_resnet152_cls,
    }
  }
  
  config = config_mapper[config_type](config_path=config_path)
  if "train" in run_type:
    train_mapper[config.model_task][config.model_name](config)
  
  if "infer" in run_type:
    infer_mapper[config.model_task][config.model_name](config)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument("-run", "--run-type", choices=["train", "infer", "eval"], default=["train"], nargs="+")
  parser.add_argument("-type", "--config-type", choices=["yolo", "coco"], default="yolo")
  parser.add_argument("-cfg", "--config-path", type=str, required=True)
  
  config = parser.parse_args()
  main(config)