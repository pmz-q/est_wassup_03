
from copy import deepcopy
from core.configs import ModelConfig
import cv2
import dlib
import json
import os
from pycocotools.coco import COCO
from tqdm import tqdm
from typing import Literal, Type


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
  "anger": 1,
  "anxiety": 2,
  "embarrass": 3,
  "happy": 4,
  "normal": 5,
  "pain": 6,
  "sad": 7,
}

def dlib_det(
  infer_images_path: str,
  dat_path: str, # shape_predictor_68_face_landmarks.dat # .dat file to predict landmarks # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  dst_dir_path: str,
  save_coco: dict=None,
  old_coco: Type[COCO]=None,
  max_save_image: int=20,
  emotion:Literal["anger", "anxiety", "embarrass", "happy", "normal", "pain", "sad"]=None,
  progress_bar=None,
  detected_more_than_one_face: list=None
):
  if save_coco != None and emotion == None:
    raise ValueError("[ save_coco ] is True but [ emotion ] is missing. Emotion is mandatory to save coco.")
  
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(dat_path)
  
  for idx, img_name in enumerate(os.listdir(f"{infer_images_path}/{emotion}")):
    img = cv2.imread(f"{infer_images_path}/{emotion}/{img_name}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, scores, index = detector.run(gray, upsample_num_times=1, adjust_threshold=0.0)
    if len(faces) > 1: 
      print("detected more than one face!!")
      detected_more_than_one_face.append(img_name)
      
    for i, face in enumerate(faces):
      if i > 1: break # assume one face per image
      landmarks = predictor(gray, face)
      keypoints = []
      
      # Cv2 Draw bbox
      x1 = face.left()
      y1 = face.top()
      x2 = face.right()
      y2 = face.bottom()
      # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
      
      # Landmarks
      for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        keypoints += [x, y, 2]  # COCO visibility flag '2' means visible
        # Draw a circle at each landmark point
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

      if save_coco != None and old_coco == None: # Assumes one face per image
        # save_coco["images"].append({
        #   "id": idx+1,
        #   "file_name": f"{emotion}{img_name}",
        #   "width": img.shape[1],
        #   "height": img.shape[0]
        # })
        
        # save_coco["annotations"].append({
        #   "id": len(save_coco["annotations"]) + 1,
        #   "image_id": idx+1,
        #   "category_id": CAT_MAPPER[emotion],
        #   "bbox": [face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()],
        #   "keypoints": keypoints,
        #   "num_keypoints": 68
        # })
      elif save_coco != None and old_coco != None:
        # img_ids = old_coco.getImgIds()
        # img_infos = old_coco.loadImgs(img_ids)
        # for img_info in img_infos:
        #   save_coco["images"].append(img_info)
        #   if img_info["file_name"] == f"{emotion}/{img_name}":
        #     ann = old_coco.imgToAnns[img_info["id"]][0]
        #     save_coco["annotations"].append({
        #         **ann,
        #         "key_points": keypoints,
        #         "num_keypoints": 68
        #     })
              
      if idx % 10 == 0:
        progress_bar.set_postfix(score="{0:.2f}".format(scores[i]))

    # Save the image with landmarks
    # if idx < max_save_image:
    cv2.imwrite(f"{dst_dir_path}/{emotion}/{img_name}", img)
    # flush cv2
    cv2.destroyAllWindows()
    progress_bar.update(1)

def main(cfg:ModelConfig):
  _, dst_dir_path = cfg.get_output_path("inference")
  
  
  print("Counting files ...")
  for mode in os.listdir(f"{cfg.infer_config.src_dir}/images"):
    coco_annotation = deepcopy(COCO_ANNOT)
    detected_more_than_one_face = []
    old_coco = COCO(f"{cfg.infer_config.src_dir}/labels/{mode}/annotation.json")
    for emotion in ["anger", "happy", "sad", "embarrass", "pain", "normal", "anxiety"]:
      total_files = 0
      for root, dirs, files in os.walk(f"{cfg.infer_config.src_dir}/images/{mode}/{emotion}"):
        total_files += len(files)
      with tqdm(total=total_files, desc=f"dlib infer {mode} [ {emotion} ]") as progress_bar:
        os.makedirs(f"{dst_dir_path}/images/{mode}/{emotion}", exist_ok=True)
        dlib_det(
          infer_images_path=f"{cfg.infer_config.src_dir}/images/{mode}",
          dat_path=cfg.infer_config.pretrained,
          dst_dir_path=f"{dst_dir_path}/images/{mode}",
          save_coco=coco_annotation,
          old_coco=old_coco,
          emotion=emotion,
          progress_bar=progress_bar,
          detected_more_than_one_face=detected_more_than_one_face
        )

    os.makedirs(f"{dst_dir_path}/labels/{mode}", exist_ok=True)
    print("writing annotation...")
    # with open(f"{dst_dir_path}/labels/{mode}/annotation.json", "w", encoding="cp949") as f:
    #   json.dump(coco_annotation, f)
    with open(f"{dst_dir_path}/labels/{mode}/detected_more_than_one_face.json", "w", encoding="cp949") as f:
      json.dump(detected_more_than_one_face, f)
  print("DONE!")
