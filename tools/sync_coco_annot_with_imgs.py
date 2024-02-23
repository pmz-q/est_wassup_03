import argparse
from copy import deepcopy
import json
from PIL import Image
from pycocotools.coco import COCO


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

def sync_coco_annot_imgs(
  dir_option: list,
  data_path: str
):
  for mode in dir_option:
    img_dir = f"{data_path}/images/{mode}"
    ann_path = f"{data_path}/labels/{mode}/annotation.json"
    
    coco_annot = COCO(ann_path)
    
    cnt = 0
    new_annot = deepcopy(COCO_ANNOT)
    print(f"{mode} images detected:", coco_annot.getImgIds())
    for img_id in coco_annot.getImgIds():
      img = coco_annot.imgs[img_id]
      
      try:
        img_width, img_height = Image.open(f"{img_dir}/{img['file_name']}").size
      except FileNotFoundError:
        cnt += 1
        continue
      
      img = {**img, "width": img_width, "height": img_height}
      anns = coco_annot.imgToAnns[img_id]
      
      new_annot["images"].append(img)
      for ann in anns:
        new_annot["annotations"].append(ann)
    
    print(f"{mode} file_not_found count:", cnt)
    with open(ann_path, "w", encoding="cp949") as f:
      json.dump(new_annot, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("-op", "--dir-option", choices=["train", "val", "test"], nargs="+", default=["train", "val", "test"])
  parser.add_argument("-path","--data-path", type=str, default="../features")
  
  config = parser.parse_args()
  sync_coco_annot_imgs(config.dir_option, config.data_path)