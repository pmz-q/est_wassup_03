import argparse
from convert_label_to_coco import convert_origin_to_coco
import json
from os.path import isfile, exists
from os import listdir, walk, makedirs
import shutil
from typing import Literal
from uuid import uuid4


def dfs(
  content:dict={"name": "root", "content": {}}, parent_path:str="../data",
  filename_old_to_new:dict={}, change_folder_name:bool=False,
  src_root_dir:str="/home/KDT-admin/work/selected_images", dst_root_dir:str="../data", mode:Literal["train", "tst"]="train"
):
  for dir in listdir(parent_path):
    uuid = str(uuid4())
    if isfile(f"{parent_path}/{dir}"): # rename files
      if "." in dir:
        uuid = f"{uuid}.{dir.split('.')[-1]}"
      content["content"][uuid] = dir
      shutil.copy(f"{parent_path}/{dir}", f"{dst_root_dir}/{uuid}")
      if f"{src_root_dir}/images/{mode}/" in parent_path:
        # TODO: 데이터 폴더 구조에 따라서 수정해야 할 수 있음
        # 주의!!!!
        # 이거 개발할 땐 data/images/anger... data/labels/anger... 이런식의 구조에서 짬
        filename_old_to_new[dir] = f"{parent_path.replace(f'{src_root_dir}/images/{mode}/', '')}/{uuid}"
    else: # rename dirs
      if change_folder_name:
        content["content"][uuid] = { "name": dir, "content": {} }
        new_path = f"{parent_path}/{dir}"
        new_dst_root_dir = f"{dst_root_dir}/{uuid}"
        makedirs(new_dst_root_dir)
        dfs(content["content"][uuid], new_path, filename_old_to_new, change_folder_name, src_root_dir, new_dst_root_dir, mode)
      else:
        content["content"][dir] = { "name": dir, "content": {} }
        new_path = f"{parent_path}/{dir}"
        new_dst_root_dir = f"{dst_root_dir}/{dir}"
        makedirs(new_dst_root_dir, exist_ok=True)
        dfs(content["content"][dir], new_path, filename_old_to_new, change_folder_name, src_root_dir, new_dst_root_dir, mode)

def create_structure_dataset(src_root_dir:str, dst_root_dir:str, filename_history:str, mode:Literal["train", "tst"]="train"):
  filename_old_to_new = {}
  new_coco_annot = None
    
  dfs(filename_history["root"]["images"][mode], f"{src_root_dir}/images/{mode}", filename_old_to_new, False, src_root_dir, f"{dst_root_dir}/images/{mode}", mode)
  dfs(filename_history["root"]["labels"][mode], f"{src_root_dir}/labels/{mode}", filename_old_to_new, False, src_root_dir, f"{dst_root_dir}/labels/{mode}", mode)

  # TODO: 데이터 폴더 구조에 따라서 수정해야 할 수 있음
  # 주의!!!!
  # 이거 개발할 땐 data/images/anger... data/labels/anger... 이런식의 구조에서 짬
  for (root, dirs, files) in walk(f"{src_root_dir}/labels/{mode}"):
      if root == f"{src_root_dir}/labels/{mode}": continue
      for file in files:
        with open(f"{root}/{file}", encoding="cp949") as f:
          origin_annots = json.load(f)
        new_coco_annot = convert_origin_to_coco(origin_annots, f"{dst_root_dir}/images/{mode}", new_coco_annot, change_img_name=True, img_names=filename_old_to_new)
    
  with open(f"{dst_root_dir}/labels/{mode}/annotation.json", "w", encoding="cp949") as f:
    json.dump(new_coco_annot, f)

def main(cfg):
  src_root_dir = cfg.src_dir_path
  dst_root_dir = cfg.dst_dir_path
  
  filename_history = {
    "root": {
      "images": {
        "train": {"name": f"{src_root_dir}/images/train", "content": {}},
        "tst": {"name": f"{src_root_dir}/images/tst", "content": {}}
      },
      "labels": {
        "train": {"name": f"{src_root_dir}/labels/train", "content": {}},
        "tst": {"name": f"{src_root_dir}/labels/tst", "content": {}}
      },
    }
  }
  
  makedirs(dst_root_dir, exist_ok=True)
  if exists(src_root_dir) and not isfile(src_root_dir):  
    if len(listdir(dst_root_dir)) != 0: shutil.rmtree(dst_root_dir)
    makedirs(f"{dst_root_dir}/images/train", exist_ok=True)
    makedirs(f"{dst_root_dir}/images/tst", exist_ok=True)
    makedirs(f"{dst_root_dir}/labels/train", exist_ok=True)
    makedirs(f"{dst_root_dir}/labels/tst", exist_ok=True)
  
  create_structure_dataset(src_root_dir, dst_root_dir, filename_history, "train")
  create_structure_dataset(src_root_dir, dst_root_dir, filename_history, "tst")
  
  with open(f"{dst_root_dir}/filename_mapper.json", "w", encoding="cp949") as f:
    json.dump(filename_history, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--src-dir-path", type=str, default="/home/KDT-admin/work/selected_images")
  parser.add_argument("--dst-dir-path", type=str, default="../data")
  
  config = parser.parse_args()
  main(config)