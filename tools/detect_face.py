import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from crop_face import crop_face_yolo
from ultralytics import YOLO

def detect_bbox_yolo(dir_option:tuple, source_root:str, save_cropped_dir:str=None, save_bbox_dir:str=None, weights_path:str=None, target_size:tuple=(224,224), padding_option:str="custom"):
    """Detect face using yolov8 pretrained, save predicted bbox information
    Args:
        source_root (str): source directory
        save_bbox_dir (str, optional): save bbox image directory. Defaults to None.
        save_cropped_dir (str, optional): save cropped image directory. Defaults to None.
        weights_path (str): a path of pretrained weights to load 
        target_size (tuple, optional): (width, height) target size of cropped image. Defaults to (224,224).
        padding_option (str): 
            1) custom - PIL, add black pixels. 
            2) yolo - yolo padding - expand facial area
            3) no_padding - only crop, no resizing (size will be different)
    Returns: 
        all_pred_bbox_results (dict): predicted bbox results for all directories
    """
    # dirs = ["train", "val", "test"]
    dirs = dir_option
    model = YOLO(weights_path)

    failed_detect_files = [] # to save detectionf failed images (path)
    # all_pred_bbox_results = {}
    for dir in dirs:
        # pred_bbox_results = {}
        emotions = os.listdir(os.path.join(source_root, dir))
        for emotion in emotions:
            sources = os.path.join(source_root, dir, emotion) + "/*.jpg"
            results = model(sources, stream=True) # YOLOv8 detection, return: predicted bbox, cropped image array
            for i, result in enumerate(results):
                head, tail = os.path.split(result.path)
                # pred_bbox_results[emotion]= {tail: { 
                #     "filename": os.path.join(head.split('/')[-1], tail), # head.split('/')[-2]: emotion, tail: filename,
                #     "bbox_xywh": result.boxes.xywh,  # Boxes object for bounding box outputs
                #     "bbox_xyxy": result.boxes.xyxy,  # Boxes object for cropping
                #     "orig_shape":  result.orig_shape, # original image shape
                #     "origin_array": result.orig_img, # original image array 
                # }}
                
                if save_bbox_dir is not None and i < 20:  # 처음 20개에 대해서만 bbox 결과 이미지 저장
                    # bbox 결과 이미지 저장
                    if len(result.boxes.xyxy)==0: # error handling
                        failed_detect_files.append(result.path)
                        continue
                    else:
                        save_path = os.path.join(save_bbox_dir, dir, emotion)
                        os.makedirs(save_path, exist_ok=True)
                        result.save(filename=os.path.join(save_path, tail))

                if save_cropped_dir is not None:
                    # padding처리한 cropped 이미지 저장
                    if len(result.boxes.xyxy)==0: # error handling
                        failed_detect_files.append(result.path)
                        continue
                    else:
                        cropped_img_array = crop_face_yolo(target_size, result.boxes.xyxy[0], result.orig_img, padding_option)
            
                    # pred_bbox_results[emotion][tail].update({"cropped_array": cropped_img_array.tolist()})
                    save_path = os.path.join(save_cropped_dir, dir, emotion)
                    os.makedirs(save_path, exist_ok=True)
                    plt.imsave(os.path.join(save_path, tail), cropped_img_array)

    #     all_pred_bbox_results[dir] = pred_bbox_results
    # return all_pred_bbox_results

# def convert_to_json_serializable(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif torch.is_tensor(obj):
#         return obj.cpu().numpy().tolist()
#     else:
#         raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main(cfg):
    dir_option = tuple(cfg.dir_option)
    src_data_path = cfg.src_data_path
    dst_data_path = cfg.dst_data_path
    bbox_data_path = cfg.bbox_data_path
    weights_path = cfg.weights_path
    target_size = tuple(cfg.target_size)
    padding_option = cfg.padding_option
    detect_bbox_yolo(dir_option, src_data_path, dst_data_path, bbox_data_path, weights_path, target_size, padding_option)
    
    # bbox 정보 저장하는 부분 주석 처리함.
    # detect_results = detect_bbox_yolo(dir_option, src_data_path, dst_data_path, bbox_data_path, weights_path, target_size, padding_option)
    # with open(os.path.join(dst_data_path, "detect_info.json"), "w") as json_file:
    #     json.dumps(detect_results, indent=4, default=convert_to_json_serializable)
    # print(dir, ": finished!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--dir-option", nargs="+", type=str, default=("train", "val", "test"), choices=["train", "val", "test"], help="choose specific folders to detect&crop")
  parser.add_argument("--src-data-path", type=str, default="../data/images", help="path where contains source images") 
  parser.add_argument("--dst-data-path", type=str, default="../cropped_data", help="destination path for cropped image") 
  parser.add_argument("--bbox-data-path", type=str, default=None, help="save original image with bbox printed") 
  parser.add_argument("--weights-path", type=str, help="pretrained weights path to load") 
  parser.add_argument("--target-size", nargs=2, type=int, default=(224, 224), help="target cropping image size, separated by space, example: 224 224")  
  parser.add_argument("--padding-option", type=str, default=None, choices=["custom", "yolo"], help="custom: resizing and add black pixels, yolo: no black pad, expand facial area to resize, no_padding: only crop, no resizing")
  config = parser.parse_args()

  main(config)
