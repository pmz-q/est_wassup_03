import argparse
from core.configs import YOLOConfig
import os
from PIL import Image
from ultralytics import YOLO


def yolo_cls(
    cfg: YOLOConfig
):
    src_dir = cfg.infer_config.src_dir
    dst_dir = cfg.get_output_path("inference")[1]
    pretrained = cfg.infer_config.pretrained
    
    model = YOLO(pretrained)
    
    # TODO: CROP THEM FIST THEN DO INFERENCE
    os.makedirs(dst_dir, exist_ok=True)
    print(dst_dir)
    
    for root, dirs, files in os.walk(src_dir):
      for file in files:
        try:
          img_path = f"{root}/{file}"
          Image.open(img_path)
        except IOError:
          # not an image file
          continue
        
        results = model(img_path)
        for result in results:
          # bbox 결과 이미지 저장
          head, tail = os.path.split(result.path)
          result.save(filename=os.path.join(dst_dir, tail))
    

# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
  
#   parser.add_argument("--src-dir", type=str, default="../data", help="path where contains images folder")
#   parser.add_argument("--pretrained", type=str, default="/home/KDT-admin/work/weights/yolov8n-face.pt", help="pretrained weights path to load")
#   config = parser.parse_args()

#   yolo_cls(config)
