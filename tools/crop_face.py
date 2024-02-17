import numpy as np
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import save_one_box
from PIL import Image

def crop_face_yolo(target_size, bbox_xyxy, origin_array, custom_padding:bool=True, yolo_padding:bool=False):
  """Crop face using bbox information
  Args:
    target_size  (int, int): (width, height) target size of cropped image,
    bbox_results (dict): predicted bbox results for all directories
    custom_padding (bool): if True: custom padding - PIL, add black pixels. if False: no padding, only crop (size will be different)
    yolo_padding (bool, optional): yolo padding - expand facial area
  """
  if custom_padding == True: 

    if yolo_padding == True:  # yolo padding - expand facial areas
      cropped_img_array = save_one_box(bbox_xyxy, origin_array, square=True, save=False)
      # resize 추가
      img_pil = Image.fromarray(cropped_img_array)
      resized_img_pil = img_pil.resize(target_size, Image.BICUBIC)
      result_array = np.array(resized_img_pil)
      print("yolo")

    else: # custom padding with black pixels
      cropped_img_array = save_one_box(bbox_xyxy, origin_array, square=False, save=False)
      face_pil = Image.fromarray(cropped_img_array)
      original_size = face_pil.size

      # Calculate the aspect ratio
      aspect_ratio = original_size[0] / original_size[1]  # width / height
      target_aspect_ratio = target_size[0] / target_size[1]  # width / height

      # Calculate the new size with padding
      if aspect_ratio > target_aspect_ratio:
        new_width = int(target_size[0])
        new_height = int(target_size[0] / aspect_ratio)
      else:
        new_width = int(target_size[1] * aspect_ratio)
        new_height = int(target_size[1])

      resized_image = face_pil.resize((new_width, new_height), Image.BICUBIC) # 1) resize
      padded_image = Image.new("RGB", target_size, (0, 0, 0)) # 2) create an empty black canvas with target size
      paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2) # 3) get a position to paste the resized image
      padded_image.paste(resized_image, paste_position) # 4) paste image to black canvas
      result_array = np.array(padded_image) # 5) convert to numpy array
      print("custom")

  else: # no resizing, only crop
    result_array = save_one_box(bbox_xyxy, origin_array, square=False, save=False)
    print("no resize")
    
  return result_array
  