import argparse
from copy import deepcopy
import json
from os import listdir, makedirs
from os.path import exists
from os.path import abspath
from pycocotools.coco import COCO
import random
import shutil
from typing import Literal
import yaml


"""
<COCO and YOLO classification data format> 

/images
    /train
        /anger
        /happy
    /test
        /anger
        /happy
/labels
    /train
        /anger
        /happy
    /test
        /anger
        /happy

<YOLO Detection data format>

/images
    /train
    /test
/labels
    /train
    /test
"""

COCO_ANNOT = {
    "info": {
        "description": "facial-expression-classification:",
        "url": "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=82",
        "version": "1.2",
        "year": 2023,
        "contributor": "한국과학기술원",
        "date_created": "2023/10/10"
    },
    "images": [
      # {
      #   "id": 1, # "id" must be int >= 1
      #   "width": 426,
      #   "height": 640,
      #   "file_name": "xxxxxxxxx.jpg",
      #   "date_captured": "2013-11-15 02:41:42"
      # }
    ],
    "annotations": [
      # {
      #     "id": 1, # "id" must be int >= 1
      #     "category_id": 1, # "category_id" must be int >= 1
      #     "image_id": 1, # "image_id" must be int >= 1
      #     "bbox": [86, 65, 220, 334] # [x,y,width,height]
      # }
    ],
    "categories": [
      # {
      #   "id": 2, # "id" must be int >= 1
      #   "name": "happy"
      # }
      { "id": 1, "name": "anger" },
      { "id": 2, "name": "anxiety" },
      { "id": 3, "name": "embarrass" },
      { "id": 4, "name": "happy" },
      { "id": 5, "name": "normal" },
      { "id": 6, "name": "pain" },
      { "id": 7, "name": "sad" },
    ]
  }

def copy_file(
  src_root_dir: str, dst_root_dir: str, img_name: str, 
  mode: Literal["train", "val"], do_for_label: bool=False
):
  shutil.copy(f"{src_root_dir}/images/train/{img_name}", f"{dst_root_dir}/images/{mode}/{img_name}")
  if do_for_label:
    shutil.copy(f"{src_root_dir}/labels/train/{img_name.split('.')[0]}.txt", f"{dst_root_dir}/labels/{mode}/{img_name.split('.')[0]}.txt")

def move_file(
  src_root_dir: str, img_name: str,
  mode: Literal["train", "val"], do_for_label: bool=False
):
  if mode == "train": return
  shutil.move(f"{src_root_dir}/images/train/{img_name}", f"{src_root_dir}/images/val/{img_name}")
  if do_for_label:
    shutil.move(f"{src_root_dir}/labels/train/{img_name.split('.')[0]}.txt", f"{src_root_dir}/labels/val/{img_name.split('.')[0]}.txt")

def split_list_val_train(list_images: list, trn_ratio: float):
  """
  Returns:
      tuple: ( trn_images, val_images )
  """
  list_images = list_images.copy()
  random.shuffle(list_images)
  middle = int(len(list_images) * trn_ratio)
  trn_images = list_images[:middle]
  val_images = list_images[middle:]
  return trn_images, val_images

def process_per_emotion(src_root_dir:str, dst_root_dir:str, e:str, trn_ratio:float):
  """
  Returns:
      tuple: ( e_trn_images, e_val_images )
  """
  process_file_action = move_file
  do_copy = src_root_dir != dst_root_dir
  if do_copy:
    process_file_action = copy_file
    shutil.rmtree(dst_root_dir, ignore_errors=True)
  
  makedirs(f"{dst_root_dir}/images/train/{e}", exist_ok=True)
  makedirs(f"{dst_root_dir}/images/val/{e}", exist_ok=True)
  makedirs(f"{dst_root_dir}/labels/train/{e}", exist_ok=True)
  makedirs(f"{dst_root_dir}/labels/val/{e}", exist_ok=True)
  
  e_images = listdir(f"{src_root_dir}/images/train/{e}")
  e_trn_images, e_val_images = split_list_val_train(e_images, trn_ratio)
  for img in e_trn_images:
    if not do_copy: break
    process_file_action(src_root_dir, dst_root_dir, f"{e}/{img}", "train", do_for_label=False)
  for img in e_val_images:
    process_file_action(src_root_dir, dst_root_dir, f"{e}/{img}", "val", do_for_label=False)

  return e_trn_images, e_val_images

def coco_annotation_split(annot_path:str, trn_images:list, val_images:list, val_annot:dict, trn_annot:dict):
  coco_annot = COCO(annot_path)
  
  for img_id in coco_annot.getImgIds():
    img = coco_annot.imgs[img_id]
    ann = coco_annot.imgToAnns[img_id]
    img_name_only = img["file_name"].split("/")[1]
    if img_name_only in val_images:
      val_images.remove(img_name_only)
      val_annot["images"].append(img)
      val_annot["annotations"].append(ann)
    else:
      trn_images.remove(img_name_only)
      trn_annot["images"].append(img)
      trn_annot["annotations"].append(ann)

def write_yolo_dataset_yaml(dst_root_dir: str):
  dst_root_abs = abspath(dst_root_dir)
  with open(f"{dst_root_dir}/yolo-dataset.yaml", "w", encoding="cp949") as f:
    yaml.dump({
      "path": f"{dst_root_abs}",
      "train": "images/train",
      "val": "images/val",
      "test": "images/test",
      "names": {
        1: "anger",
        2: "anxiety",
        3: "embarrass",
        4: "happy",
        5: "normal",
        6: "pain",
        7: "sad"
      }
    }, f)

def copy_test_set(src_root_dir: str, dst_root_dir: str):
  if src_root_dir != dst_root_dir:
    dst_images_test = f"{dst_root_dir}/images/test"
    dst_labels_test = f"{dst_root_dir}/labels/test"
    shutil.rmtree(dst_images_test, ignore_errors=True)
    shutil.rmtree(dst_labels_test, ignore_errors=True)
    shutil.copytree(f"{src_root_dir}/images/test", dst_images_test)
    shutil.copytree(f"{src_root_dir}/labels/test", dst_labels_test)

def yolo_detection_split(src_root_dir: str, dst_root_dir: str, train_ratio:float):
  """
  this will reformat the source root dir if src_root_dir == dst_root_dir
  if not are same, this will copy images and labels to the dst_root_dir
  """
  process_file_action = move_file
  do_copy = src_root_dir != dst_root_dir
  if do_copy:
    process_file_action = copy_file
    shutil.rmtree(dst_root_dir, ignore_errors=True)

  makedirs(f"{dst_root_dir}/images/train", exist_ok=True)
  makedirs(f"{dst_root_dir}/images/val", exist_ok=True)
  makedirs(f"{dst_root_dir}/labels/train", exist_ok=True)
  makedirs(f"{dst_root_dir}/labels/val", exist_ok=True)
  
  list_images = listdir(f"{src_root_dir}/images/train")
  trn_images, val_images = split_list_val_train(list_images, train_ratio)
  for img in trn_images:
    if not do_copy: break
    process_file_action(src_root_dir, dst_root_dir, img, "train", do_for_label=True)
  for img in val_images:
    process_file_action(src_root_dir, dst_root_dir, img, "val", do_for_label=True)
  
  write_yolo_dataset_yaml(dst_root_dir)

def yolo_classification_split(src_root_dir: str, dst_root_dir: str, train_ratio:float):
  """
  this will reformat the source root dir if src_root_dir == dst_root_dir
  if not are same, this will copy images and labels to the dst_root_dir
  """
  emotions = listdir(f"{src_root_dir}/images/train")
  
  val_annot = deepcopy(COCO_ANNOT)
  trn_annot = deepcopy(COCO_ANNOT)
  
  trn_images = []
  val_images = []
  
  for e in emotions:
    e_trn_images, e_val_images = process_per_emotion(src_root_dir, dst_root_dir, e, train_ratio)
    trn_images.extend(e_trn_images)
    val_images.extend(e_val_images)
  
  coco_annotation_split(f"{src_root_dir}/labels/train/annotation.json", trn_images, val_images, val_annot, trn_annot)
  
  with open(f"{dst_root_dir}/labels/train/annotation.json", "w", encoding="cp949") as f:
    json.dump(trn_annot, f)
  
  with open(f"{dst_root_dir}/labels/val/annotation.json", "w", encoding="cp949") as f:
    json.dump(val_annot, f)

def coco_classification_split(src_root_dir: str, dst_root_dir: str, train_ratio:float):
  """
  this will reformat the source root dir if src_root_dir == dst_root_dir
  if not are same, this will copy images and labels to the dst_root_dir
  """
  emotions = listdir(f"{src_root_dir}/images/train")
  
  val_annot = deepcopy(COCO_ANNOT)
  trn_annot = deepcopy(COCO_ANNOT)
  
  trn_images = []
  val_images = []
  
  for e in emotions:
    e_trn_images, e_val_images = process_per_emotion(src_root_dir, dst_root_dir, e, train_ratio)
    trn_images.extend(e_trn_images)
    val_images.extend(e_val_images)
  
  coco_annotation_split(f"{src_root_dir}/labels/train/annotation.json", trn_images, val_images, val_annot, trn_annot)
  
  with open(f"{dst_root_dir}/labels/train/annotation.json", "w", encoding="cp949") as f:
    json.dump(trn_annot, f)
  
  with open(f"{dst_root_dir}/labels/val/annotation.json", "w", encoding="cp949") as f:
    json.dump(val_annot, f)

def coco_detection_split(src_root_dir: str, dst_root_dir: str, train_ratio:float):
  """
  this will reformat the source root dir if src_root_dir == dst_root_dir
  if not are same, this will copy images and labels to the dst_root_dir
  """
  emotions = listdir(f"{src_root_dir}/images/train")
  
  val_annot = deepcopy(COCO_ANNOT)
  trn_annot = deepcopy(COCO_ANNOT)
  
  trn_images = []
  val_images = []
  
  for e in emotions:
    e_trn_images, e_val_images = process_per_emotion(src_root_dir, dst_root_dir, e, train_ratio)
    trn_images.extend(e_trn_images)
    val_images.extend(e_val_images)
  
  coco_annotation_split(f"{src_root_dir}/labels/train/annotation.json", trn_images, val_images, val_annot, trn_annot)
  
  with open(f"{dst_root_dir}/labels/train/annotation.json", "w", encoding="cp949") as f:
    json.dump(trn_annot, f)
  
  with open(f"{dst_root_dir}/labels/val/annotation.json", "w", encoding="cp949") as f:
    json.dump(val_annot, f)

def main(cfg):
  annot_format = cfg.annot_format[0]
  task = cfg.task[0]
  train_ratio = cfg.train_ratio
  src_root_dir = cfg.src_root_path
  dst_root_dir = cfg.dst_root_path 

  splitter_map = {
    "classification": {
      "yolo": yolo_classification_split,
      "coco": coco_classification_split
    },
    "detection": {
      "yolo": yolo_detection_split,
      "coco": coco_detection_split
    }
  }
  
  splitter_map[task][annot_format](src_root_dir, dst_root_dir, train_ratio)
  copy_test_set(src_root_dir, dst_root_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  # 주의!!!!!
  # split 하려는 src 폴더 포맷이 annot-format, task 와 같은지 확인하십시오!!!!
  # Make sure there are two folders in the src-root-path: images & labels
  parser.add_argument("--annot-format", choices=["coco", "yolo"], nargs=1, default="coco")
  parser.add_argument("--task", choices=["classification", "detection"], nargs=1, default="classification")
  parser.add_argument("--src-root-path", type=str, default="../data")
  parser.add_argument("--dst-root-path", type=str, default="../features")
  parser.add_argument("--train-ratio", type=float, default=0.8)
  
  config = parser.parse_args()
  main(config)