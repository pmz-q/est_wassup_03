import argparse
from os import listdir, makedirs
import random
import shutil
from typing import Literal


"""
<COCO and YOLO classification data format> 

/images
    /train
        /anger
        /happy
    /tst
        /anger
        /happy
/labels
    /train
        /anger
        /happy
    /tst
        /anger
        /happy

<YOLO Detection data format>

/images
    /train
    /tst
/labels
    /train
    /tst
"""

def copy_file(
  src_root_dir: str, dst_root_dir: str, img_name: str, 
  mode: Literal["train", "val"], annot_format: Literal["coco", "yolo"]
):
  if annot_format == "coco":
    pass
  else:
    shutil.copy(f"{src_root_dir}/images/train/{img_name}", f"{dst_root_dir}/images/{mode}/{img_name}")
    shutil.copy(f"{src_root_dir}/labels/train/{img_name.split('.')[0]}.txt", f"{dst_root_dir}/labels/{mode}/{img_name.split('.')[0]}.txt")

def move_file(
  src_root_dir: str, img_name: str,
  mode: Literal["train", "val"], annot_format: Literal["coco", "yolo"]
):
  if mode == "train": return
  if annot_format == "coco":
    pass
  else:
    shutil.move(f"{src_root_dir}/images/train/{img_name}", f"{src_root_dir}/images/val/{img_name}")
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
  print(len(trn_images), len(val_images))
  for img in trn_images:
    if not do_copy: break
    process_file_action(src_root_dir, dst_root_dir, img, "train", "yolo")
  for img in val_images:
    process_file_action(src_root_dir, dst_root_dir, img, "val", "yolo")

def yolo_classification_split():
  raise NotImplementedError("yolo classification train/val split")

def coco_classification_split():
  raise NotImplementedError("coco classification train/val split")

def coco_detection_split():
  raise NotImplementedError("coco detection train/val split")

def main(cfg):
  print(cfg)
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