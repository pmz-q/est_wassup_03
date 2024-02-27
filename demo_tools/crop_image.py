import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box


ROOT_PATH = Path(__file__).parent

def resize_with_padding(cropped_img_array, target_size, padding_scale):
  """ Resize an image to a target size using padding
  cropped_img_array: image array
  target_size: (width, height)
  return: resized image array
  """
  face_pil = Image.fromarray(cropped_img_array)
  original_size = face_pil.size
  # Calculate the aspect ratio
  aspect_ratio = original_size[0] / original_size[1]  # width / height
  target_aspect_ratio = target_size[0] / target_size[1]  # width / height

  # Calculate the new size with padding
  if aspect_ratio > target_aspect_ratio:
      new_height = int(target_size[1] * padding_scale)
      new_width = int(new_height * aspect_ratio)
  else:
      new_width = int(target_size[0] * padding_scale)
      new_height = int(new_width / aspect_ratio)

  resized_image = face_pil.resize((new_width, new_height), Image.BICUBIC) # 1) resize
  padded_image = Image.new("RGB", target_size, (0, 0, 0)) # 2) create an empty black canvas with target size
  paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2) # 3) get a position to paste the resized image
  padded_image.paste(resized_image, paste_position) # 4) paste image to black canvas

  return np.array(padded_image)

def detect_bbox_yolo(filename: str) -> tuple:
  """
  Args:
    filename (str): uploaded filename. the image must be located in demo_tools/input_images.
  Returns:
    (is_successful, result_path)
    is_successful (bool): True if the process was successful. else False
    result_path (str): absoloute path of the saved result image. Error message when the process is not successfuly done.
  """
  src = f"{ROOT_PATH}/input_images/{filename}"
  save_path = f"{ROOT_PATH}/results/cropped/{filename}"
  weights_path = f"{ROOT_PATH}/weights/crop_weight/yolov8n-face.pt"
  target_size = (224, 224)
  padding_scale = 0.8
  
  if not os.path.exists(weights_path): return False, "ERRPR:detect_bbox_yolo:weight file does not exists."
  if not os.path.isfile(weights_path): return False, "ERROR:detect_bbox_yolo:given weight is a directory not a file."
  model = YOLO(weights_path)
  
  try:
    result = model(src, verbose=False)[0] # predict one file only
  except Exception as e:
    return False, str(e)

  if len(result.boxes.xyxy) == 0:
    return False, "ERROR:detect_bbox_yolo:no human face found."
  
  cropped = save_one_box(
    xyxy=result.boxes.xyxy[0],
    im=result.orig_img,
    square=False,
    save=False
  )
  
  padded = resize_with_padding(
    cropped_img_array=cropped,
    target_size=target_size,
    padding_scale=padding_scale
  )
  
  plt.imsave(save_path, padded)

  return True, save_path

if __name__ == "__main__":
  # test code
  returned = detect_bbox_yolo("example.jpg")
  print(returned)