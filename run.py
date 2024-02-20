import argparse
from core.configs import YOLOConfig, ModelConfig
from train_tools import yolo_det, yolo_cls, resnet50_cls
from typing import Dict, Type

def main(cfg):
  config_path = cfg.config_path
  config_type = cfg.config_type
  run_type = cfg.run_type
  
  config_mapper:Dict[str, Type[ModelConfig]] = {
    "yolo": YOLOConfig,
    "coco": ModelConfig
  }
  
  train_mapper: Dict[str, function] = {
    "classification": {
      "yolo": yolo_cls,
      "resnet50": resnet50_cls
    },
    "detection": {
      "yolo": yolo_det
    }
  }
  
  if "train" in run_type:
    config = config_mapper[config_type](config_path=config_path)
    train_mapper[config.model_task][config.model_name](config)
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument("-run", "--run-type", choices=["train", "inference", "eval"], default=["train"], nargs="+")
  parser.add_argument("-type", "--config-type", choices=["yolo", "coco"], default="yolo")
  parser.add_argument("-cfg", "--config-path", type=str, required=True)
  
  config = parser.parse_args()
  main(config)