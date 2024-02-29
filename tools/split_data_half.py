import argparse
import os
from os.path import exists, isfile
from pathlib import Path
import random
import shutil
import warnings
import glob

def main(cfg):
  src_data_dir = cfg.src_data_path
  dst_data_dir = cfg.dst_data_path
  selection_ratio = cfg.selection_ratio
  
  if not exists(src_data_dir): raise FileNotFoundError(f"directory [{src_data_dir}] does not exists")
  elif isfile(src_data_dir): raise FileNotFoundError(f"[{src_data_dir}] is not a directory. It supposed to be a directory!")
  else:
    image_dir = src_data_dir + "/images"
    label_dir = src_data_dir + "/labels"
    dirs = ["train", "val", "test"]
    emotions = ["anger", "anxiety", "embarrass", "happy", "normal", "pain", "sad"]

    for dir in dirs:
        # image
        for emotion in emotions: 
            num_images_to_copy = int(len(os.listdir(os.path.join(image_dir, dir, emotion)))*selection_ratio)
            print(num_images_to_copy)
            img_paths_list = os.listdir(os.path.join(image_dir, dir, emotion))
            # print(img_paths_list)
            # random.shuffle(img_paths_list)
            for img in img_paths_list[:num_images_to_copy]:
               img_src = os.path.join(image_dir, dir, emotion)
               img_dst = os.path.join(dst_data_dir, "images", dir, emotion)
               os.makedirs(img_dst, exist_ok=True)
               shutil.copy(os.path.join(img_src, img), os.path.join(img_dst, img))
    # label
    shutil.copytree(label_dir, os.path.join(dst_data_dir, "labels"))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--selection-ratio", type=float, default=0.5)
  parser.add_argument("--src-data-path", type=str, default="/home/KDT-admin/work/data/splitted_custom_cropped_0.5/cleaned_0.5")
  parser.add_argument("--dst-data-path", type=str, default="/home/KDT-admin/work/data/features_0.25")
  
  config = parser.parse_args()
  main(config)