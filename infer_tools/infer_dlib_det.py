
from copy import deepcopy
from core.configs import ModelConfig
import cv2
import dlib
import json
import math
import matplotlib.pyplot as plt
import numpy as np
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
    #   "id": 1, # "id" must be int >= 1
    #   "category_id": 1, # "category_id" must be int >= 1
    #   "image_id": 1, # "image_id" must be int >= 1
    #   "bbox": [86, 65, 220, 334] # [x,y,width,height]
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

#Function to create heatmaps by convoluting a 2D gaussian kernel over a (x,y) keypoint.
def gaussian(xL, yL, H, W, sigma=5):
  channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
  channel = np.array(channel, dtype=np.float32)
  channel = np.reshape(channel, newshape=(H, W))
  
  return channel

def keypoints_to_heatmap(keypoints: list, img_width: int, img_height: int):
  #Generate heatmaps for one sample image
  heatmaps = []

  for x, y in keypoints:
    heatmap = gaussian(x, y, img_width, img_height,)
    heatmaps.append(heatmap)
  
  heatmaps_68 = np.array(heatmaps)
  return heatmaps_68

def save_img(index, save_path, emotion, img_name, img):
  if save_path != None:
    if index > 0:
      path, ext = img_name.split(".")
      path += f"_{index}."
      cv2.imwrite(f"{save_path}/{emotion}/{path + ext}", img)
    else:
      print(save_path)
      print(emotion)
      print(img_name)
      cv2.imwrite(f"{save_path}/{emotion}/{img_name}", img)

def dlib_det(
  infer_images_path: str,
  dat_path: str, # shape_predictor_68_face_landmarks.dat # .dat file to predict landmarks # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  gray_save_path: str=None,
  color_save_path: str=None,
  heatmap_save_path: str=None,
  emotion:Literal["anger", "anxiety", "embarrass", "happy", "normal", "pain", "sad"]=None,
  progress_bar=None,
  detected_more_than_one_face: list=None
):
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
      landmarks = predictor(gray, face)
      keypoints = []
      
      # Landmarks
      for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        keypoints.append([x, y])

        # Draw a circle at each landmark point
        if color_save_path != None:
          cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        if gray_save_path != None:
          cv2.circle(gray, (x, y), 2, (255, 0, 0), -1)
      
      # save heatmaps
      if heatmap_save_path != None:
        heatmaps_68 = keypoints_to_heatmap(keypoints, gray.shape[1], gray.shape[0])
        
        fig = plt.figure(frameon=False)
        x = img.shape[1] / fig.dpi
        y = img.shape[0] / fig.dpi
        fig.set_size_inches(x, y)
        
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.imshow(heatmaps_68.sum(axis=0), aspect="auto")
        
        if i > 0:
          path, ext = img_name.split(".")
          path+= f"_{i}."
          fig.savefig(f"{heatmap_save_path}/{emotion}/{path + ext}")
        else:
          fig.savefig(f"{heatmap_save_path}/{emotion}/{img_name}")
        
      # Save the image with landmarks
      if color_save_path != None:
        save_img(i, color_save_path, emotion, img_name, img)
      if gray_save_path != None:
        save_img(i, gray_save_path, emotion, img_name, gray)
      
      # update progress bar postfix
      if idx % 10 == 0:
        progress_bar.set_postfix(score="{0:.2f}".format(scores[i]))
    
    # flush cv2
    cv2.destroyAllWindows()
    progress_bar.update(1)
    raise

def main(cfg:ModelConfig):
  src_image_dir = f"{cfg.infer_config.src_dir}/images"
  _, dst_dir_path = cfg.get_output_path("inference")
  dst_image_dir = f"{dst_dir_path}/images"
  
  print("Counting files ...")
  for mode in os.listdir(src_image_dir):
    # read only dirs not files - train or val or test
    src_mode_dir = f"{src_image_dir}/{mode}"
    if os.path.isfile(src_mode_dir): continue
    
    detected_more_than_one_face = []
    for emotion in os.listdir(src_mode_dir):
      total_files = 0
      for root, dirs, files in os.walk(f"{src_mode_dir}/{emotion}"):
        total_files += len(files)
      
      with tqdm(total=total_files, desc=f"dlib infer {mode} [ {emotion} ]") as progress_bar:
        dst_mode_dir = {}
        for save_mode in ["gray", "color", "heatmap"]:
          if save_mode in cfg.infer_config.save_mode:
            dst_mode_dir[save_mode] = f"{dst_image_dir}/{save_mode}/{mode}"
            os.makedirs(f"{dst_mode_dir[save_mode]}/{emotion}", exist_ok=True)
          else:
            dst_mode_dir[save_mode] = None
        
        dlib_det(
          infer_images_path=src_mode_dir,
          dat_path=cfg.infer_config.pretrained,
          gray_save_path=dst_mode_dir["gray"],
          color_save_path=dst_mode_dir["color"],
          heatmap_save_path=dst_mode_dir["heatmap"],
          emotion=emotion,
          progress_bar=progress_bar,
          detected_more_than_one_face=detected_more_than_one_face
        )

    print("finalizing...")
    with open(f"{dst_mode_dir}/detected_more_than_one_face.json", "w", encoding="cp949") as f:
      json.dump(detected_more_than_one_face, f)
  print("DONE!")
