import argparse
import ultralytics
from ultralytics import YOLO


def main(cfg):
  ultralytics.checks()
  model_weight = cfg.model_weight_path
  src_data_dir = cfg.src_data_path
  dst_root_dir = cfg.dst_root_path

  model = YOLO(model_weight)
  model.predict(
    src_data_dir,
    save=True,
    device="cuda", max_det=1, save_crop=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--model-weight-path", type=str, default="../results/train/train1/weights/best.pt")
  parser.add_argument("--src-data-path", type=str, default="../features/images/test")
  parser.add_argument("--dst-root-path", type=str, default="../results/inference")
  
  config = parser.parse_args()
  main(config)