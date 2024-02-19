import argparse
from copy import deepcopy
# import cv2
from datetime import datetime
import json
import os
from os import walk
from os.path import isfile, exists
from PIL import Image
import sys


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

CAT_MAPPER = {
  "분노": 1,
  "불안": 2,
  "당황": 3,
  "기쁨": 4,
  "중립": 5,
  "상처": 6,
  "슬픔": 7,
}

def convert_origin_to_coco(
  origin_annots:list, img_root_dir:str, coco_annot:dict=None,
  change_img_name:bool=False, img_names:dict={"old": "new"}
):
  if coco_annot == None: coco_annot = deepcopy(COCO_ANNOT)
  for annot in origin_annots:
    img_filename = ""
    try:
      img_filename = img_names[annot["filename"]]
    except KeyError:
      if change_img_name: continue
      else: img_filename = annot["filename"]
    
    img_id = len(coco_annot["images"]) + 1
    boxes = None
    
    for annot_type in ["A", "B", "C"]:
      try:
        boxes = annot[f"annot_{annot_type}"]["boxes"]
        assert boxes["maxX"] > boxes["minX"] and boxes["maxY"] > boxes["minY"], f"Box size error !: (xmin, ymin, xmax, ymax): {boxes['minX'], boxes['minY'], boxes['maxX'], boxes['maxY']}"
      except:
        boxes = None
        continue
      else:
        break
    
    # when bbox errors, remove the image
    if boxes == None:
      os.remove(f"{img_root_dir}/{img_filename}")
      continue
    
    width = boxes["maxX"] - boxes["minX"]
    height = boxes["maxY"] - boxes["minY"]
    
    # img_height, img_width, _ = cv2.imread(f"{img_root_dir}/{img_filename}").shape
    img_width, img_height = Image.open(f"{img_root_dir}/{img_filename}").size
    
    # images
    coco_annot["images"].append({
      "id": img_id,
      "width": img_width,
      "height": img_height,
      "file_name": img_filename,
      "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # annotations
    coco_annot["annotations"].append({
      "id": len(coco_annot["annotations"]) + 1,
      "category_id": CAT_MAPPER[annot["faceExp_uploader"]],
      "image_id": img_id,
      "bbox": [
        boxes["minX"], # x
        boxes["minY"], # y
        width, # width
        height # height
      ]
    })
  
  return coco_annot

def main(cfg):
  # TODO: functions has been updated. This main function need to be updated as so.
  root_dir = cfg.dir_path
  new_coco_annot = {}
  
  if not exists(root_dir): raise FileNotFoundError(f"directory [{root_dir}] does not exists")
  elif isfile(root_dir): raise FileNotFoundError(f"[{root_dir}] is not a directory. It supposed to be a directory!")
  else:
    for (root, dirs, files) in walk(root_dir):
      if root == root_dir: continue
      for file in files:
        with open(f"{root}/{file}", encoding="cp949") as f:
          origin_annots = json.load(f)
        new_coco_annot = convert_origin_to_coco(origin_annots)
  
  with open(f"{root_dir}/annotation.json", "w", encoding="cp949") as f:
    json.dump(new_coco_annot, f)
  
# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
  
#   parser.add_argument("--dir-path", type=str, default="../data/labels")
  
#   config = parser.parse_args()
#   main(config)