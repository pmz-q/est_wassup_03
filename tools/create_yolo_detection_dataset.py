import argparse
import json
from os import makedirs, listdir
import shutil
from typing import Literal

"""
<FROM> 

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

<TO>

/images
    /train
    /test
/labels
    /train
    /test
"""

DIR_LEVEL_1 = ("images", "labels")
DIR_LEVEL_2 = ("train", "test")

def bbox_2_yolo(bbox, img_w, img_h):
  x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
  centerx = bbox[0] + w / 2
  centery = bbox[1] + h / 2
  dw = 1 / img_w
  dh = 1 / img_h
  centerx *= dw
  w *= dw
  centery *= dh
  h *= dh
  return centerx, centery, w, h


def convert_anno(src_data_path:str, mode:Literal["train", "test"]="train"):
  """
  Returns:
      { "image_id": [(image_name, category_id, yolobox)] }
  """
  # read annotation data
  with open(f"{src_data_path}/labels/{mode}/annotation.json", encoding="cp949") as f:
    data = json.load(f)
  
  # create image info dictionary { image_id = image_info }
  images = dict()
  for image in data['images']:
    id = image['id']
    file_name = image['file_name']
    w = image['width']
    h = image['height']
    images[id] = (file_name, w, h)
  
  # create yolo format annotations
  anno_dict = dict()
  for anno in data['annotations']:
    bbox = anno['bbox']
    image_id = anno['image_id']
    category_id = anno['category_id']

    image_info = images.get(image_id)
    image_name = image_info[0]
    img_w = image_info[1]
    img_h = image_info[2]
    yolo_box = bbox_2_yolo(bbox, img_w, img_h)

    anno_info = (image_name, category_id, yolo_box)
    anno_infos = anno_dict.get(image_id)
    if not anno_infos:
      anno_dict[image_id] = [anno_info]
    else:
      anno_infos.append(anno_info)
      anno_dict[image_id] = anno_infos
  
  return anno_dict

def write_yolo_annot_txt(src_data_path:str, dst_data_path:str, anno_dict:dict, mode:Literal["train", "test"]="train"):
  for k, v in anno_dict.items():
    emotion, origin_file_name = v[0][0].split("/")
    # copy img
    shutil.copy(f"{src_data_path}/images/{mode}/{emotion}/{origin_file_name}", f"{dst_data_path}/images/{mode}/{origin_file_name}")
    
    # write yolo txt
    file_name = origin_file_name.split(".")[0] + ".txt"
    with open(f"{dst_data_path}/labels/{mode}/{file_name}", 'w', encoding='cp949') as f:
      for obj in v:
        # category_id = obj[1]
        category_id = 0 # detect face only
        box = ['{:.6f}'.format(x) for x in obj[2]]
        box = ' '.join(box)
        line = str(category_id) + ' ' + box
        f.write(line + '\n')

def main(cfg):
  src_data_path = cfg.src_data_path
  dst_data_path = cfg.dst_data_path
  
  # create features folder if not exists
  makedirs(dst_data_path, exist_ok=True)
  
  # create destination directories
  for level1 in DIR_LEVEL_1:
    dir_level1 = f"{dst_data_path}/{level1}"
    makedirs(dir_level1, exist_ok=True)
    if len(listdir(dir_level1)) != 0:
      shutil.rmtree(dir_level1)
    for level2 in DIR_LEVEL_2:
      makedirs(f"{dir_level1}/{level2}", exist_ok=True)
  
  train_anns = convert_anno(src_data_path, "train")
  test_anns = convert_anno(src_data_path, "test")
  
  # save annotation txt files
  write_yolo_annot_txt(src_data_path, dst_data_path, train_anns, "train")
  write_yolo_annot_txt(src_data_path, dst_data_path, test_anns, "test")
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--src-data-path", type=str, default="../data")
  parser.add_argument("--dst-data-path", type=str, default="../yolo_detection_data")
  
  config = parser.parse_args()
  main(config)
