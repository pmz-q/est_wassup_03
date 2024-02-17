import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from crop_face import crop_face_yolo
from ultralytics import YOLO

def detect_bbox_yolo(source_root:str, save_cropped_dir:str=None, save_bbox_dir:str=None, weights_path:str=None, target_size:tuple=(224,224), custom_padding:bool=True, yolo_padding:bool=False):
    """Detect face using yolov8 pretrained, save predicted bbox information
    Args:
        source_root (str): source directory
        save_bbox_dir (str, optional): save bbox image directory. Defaults to None.
        save_cropped_dir (str, optional): save cropped image directory. Defaults to None.
        weights_path (str): a path of pretrained weights to load 
        target_size (tuple, optional): (width, height) target size of cropped image. Defaults to (224,224).
        custom_padding (bool, optional): if True: custom padding - PIL, add black pixels. if False: no padding, only crop (size will be different). Defaults to True.
        yolo_padding (bool, optional): yolo padding - expand facial area. Defaults to False.
    Returns: 
        all_pred_bbox_results (dict): predicted bbox results for all directories
    """
    source_root = source_root + "/images"
    dirs = os.listdir(source_root)
    
    model = YOLO(weights_path)

    all_pred_bbox_results = {}
    for dir in dirs:
        pred_bbox_results = {}
        emotions = os.listdir(os.path.join(source_root, dir))
        for emotion in emotions:
            sources = os.path.join(source_root, dir, emotion) + "/*.jpg"
            results = model(sources, stream=True) # YOLOv8 detection, return: predicted bbox, cropped image array
            for result in results:
                head, tail = os.path.split(result.path)
                pred_bbox_results[emotion] = {tail: { 
                    "filename": os.path.join(head.split('/')[-1], tail), # head.split('/')[-2]: emotion, tail: filename,
                    "bbox_xywh": result.boxes.xywh,  # Boxes object for bounding box outputs
                    "bbox_xyxy": result.boxes.xyxy,  # Boxes object for cropping
                    "orig_shape":  result.orig_shape, # original image shape
                    "origin_array": result.orig_img, # original image array 
                }}
                
                if save_bbox_dir is not None:
                    # bbox 결과 이미지 저장
                    save_path = os.path.join(save_bbox_dir, dir, emotion)
                    os.makedirs(save_path, exist_ok=True)
                    result.save(filename=os.path.join(save_path, tail))

                if save_cropped_dir is not None:
                    # padding처리한 cropped 이미지 저장
                    cropped_img_array = crop_face_yolo(target_size, result.boxes.xyxy[0], result.orig_img, custom_padding, yolo_padding)
                    save_path = os.path.join(save_cropped_dir, dir, emotion)
                    os.makedirs(save_path, exist_ok=True)
                    plt.imsave(os.path.join(save_path, tail), cropped_img_array)

        all_pred_bbox_results[dir] = pred_bbox_results
        print(dir, ": finished!")
    return all_pred_bbox_results

def main(cfg):
    src_data_path = cfg.src_data_path
    dst_data_path = cfg.dst_data_path
    bbox_data_path = cfg.bbox_data_path
    weights_path = cfg.bbox_weights_path
    target_size = cfg.target_size
    custom_padding = cfg.custom_padding
    yolo_padding = cfg.yolo_padding
    detect_bbox_yolo(src_data_path, dst_data_path, bbox_data_path, target_size, custom_padding, yolo_padding)

    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--src-data-path", type=str, default="../data")
  parser.add_argument("--dst-data-path", type=str, default="../cropped_data") # destination path for cropped image  
  parser.add_argument("--bbox-data-path", type=str, default=None) # for checking bbox in orginal image
  parser.add_argument("--weights-path", type=str, default="/home/KDT-admin/work/weights/yolov8n-face.pt") # put pretrained weights path to load
  parser.add_argument("--target_size", type=tuple, default=(224,224)) # target cropping image size
  parser.add_argument("--custom_padding", type=bool, default=True) # if True: custom padding - PIL, add black pixels. if False: no padding, only crop (size will be different)
  parser.add_argument("--yolo_padding", type=bool, default=False) # expand facial area.
  config = parser.parse_args()

  main(config)
