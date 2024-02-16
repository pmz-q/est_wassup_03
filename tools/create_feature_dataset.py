import argparse
from convert_label_to_coco import convert_origin_to_coco
import json
from os.path import isfile, exists
from os import listdir, rename, walk
from typing import Literal
from uuid import uuid4


def dfs(
  content:dict={"name": "root", "content": {}}, parent_path:str="../data",
  filename_old_to_new:dict={}, change_folder_name:bool=False,
  root_dir:str="../data", mode:Literal["train", "tst"]="train"
):
  for dir in listdir(parent_path):
    uuid = str(uuid4())
    if isfile(f"{parent_path}/{dir}"): # rename files
      if "." in dir:
        uuid = f"{uuid}.{dir.split('.')[-1]}"
      content["content"][uuid] = dir
      rename(f"{parent_path}/{dir}", f"{parent_path}/{uuid}")
      if f"{root_dir}/images/{mode}/" in parent_path:
        # TODO: 데이터 폴더 구조에 따라서 수정해야 할 수 있음
        # 주의!!!!
        # 이거 개발할 땐 data/images/anger... data/labels/anger... 이런식의 구조에서 짬
        filename_old_to_new[dir] = f"{parent_path.replace(f'{parent_path}/images/{mode}/', '')}/{uuid}"
    else: # rename dirs
      if change_folder_name:
        content["content"][uuid] = { "name": dir, "content": {} }
        new_path = f"{parent_path}/{uuid}"
        rename(f"{parent_path}/{dir}", new_path)
        dfs(content["content"][uuid], new_path, filename_old_to_new, change_folder_name, root_dir, mode)
      else:
        content["content"][dir] = { "name": dir, "content": {} }
        new_path = f"{parent_path}/{dir}"
        dfs(content["content"][dir], new_path, filename_old_to_new, change_folder_name, root_dir, mode)

def create_structure_dataset(root_dir:str, filename_history:str, mode:Literal["train", "tst"]="train"):
  filename_old_to_new = {}
  new_coco_annot = {}
   
  if exists(root_dir) and not isfile(root_dir):
    dfs(filename_history["root"]["images"][mode], f"{root_dir}/images/{mode}", filename_old_to_new, False, root_dir, mode)
    dfs(filename_history["root"]["labels"][mode], f"{root_dir}/labels/{mode}", filename_old_to_new, False, root_dir, mode)
  
  # TODO: 데이터 폴더 구조에 따라서 수정해야 할 수 있음
  # 주의!!!!
  # 이거 개발할 땐 data/images/anger... data/labels/anger... 이런식의 구조에서 짬
  for (root, dirs, files) in walk(f"{root_dir}/labels/{mode}"):
      if root == f"{root_dir}/labels/{mode}": continue
      for file in files:
        with open(f"{root}/{file}", encoding="cp949") as f:
          origin_annots = json.load(f)
        new_coco_annot = convert_origin_to_coco(origin_annots, change_img_name=True, img_names=filename_old_to_new)
  
  with open(f"{root_dir}/labels/{mode}/annotation.json", "w", encoding="cp949") as f:
    json.dump(new_coco_annot, f)

def kor_to_uuid(cfg):
  root_dir = cfg.dir_path
  
  filename_history = {
    "root": {
      "images": {
        "train": {"name": f"{root_dir}/images/train", "content": {}},
        "tst": {"name": f"{root_dir}/images/tst", "content": {}}
      },
      "labels": {
        "train": {"name": f"{root_dir}/labels/train", "content": {}},
        "tst": {"name": f"{root_dir}/labels/tst", "content": {}}
      },
    }
  }
  
  create_structure_dataset(root_dir, filename_history, "train")
  create_structure_dataset(root_dir, filename_history, "tst")
  
  with open(f"{root_dir}/filename_mapper.json", "w", encoding="cp949") as f:
    json.dump(filename_history, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--dir-path", type=str, default="../data")
  
  config = parser.parse_args()
  kor_to_uuid(config)