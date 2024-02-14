import argparse
import json
from os.path import isfile, exists
from os import walk
from datetime import datetime


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
  origin_annots:list, coco_annot:dict=COCO_ANNOT,
  change_img_name:bool=False, img_names:dict={"old": "new"}
):
  for annot in origin_annots:
    boxes = annot["annot_A"]["boxes"]
    img_id = len(coco_annot["images"]) + 1
    width = boxes["maxX"] - boxes["minX"]
    height = boxes["maxY"] - boxes["minY"]
    
    # images
    coco_annot["images"].append({
      "id": img_id,
      "width": width,
      "height": height,
      "file_name": annot["filename"] if not change_img_name else img_names[annot["filename"]],
      "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # annotations
    coco_annot["annotations"].append({
      "id": len(coco_annot["annotations"]) + 1,
      "category_id": CAT_MAPPER[annot["faceExp_uploader"]],
      "image_id": img_id,
      "bbox": [
        boxes["minX"], # x
        boxes["maxY"], # y
        width, # width
        height # height
      ]
    })
  
  return coco_annot

def main(cfg):
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
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--dir-path", type=str, default="../data/labels")
  
  config = parser.parse_args()
  main(config)