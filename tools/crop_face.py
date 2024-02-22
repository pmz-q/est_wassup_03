import numpy as np
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import save_one_box
from PIL import Image

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

    return np.array(padded_image) # return image array


def crop_face_yolo(target_size, bbox_xyxy, origin_array, padding_option:str, padding_scale:float=0.8):
  """Crop face using bbox information
  Args:
    target_size  (int, int): (width, height) target size of cropped image,
    bbox_results (dict): predicted bbox results for all directories
    origin_array (np.array): original image array
    padding_option (str): 
      1) custom - PIL, add black pixels. 
      2) yolo - yolo padding - expand facial area
      3) None - only crop, no resizing nor padding (size will be different)
    padding_scale (int): default = 0.8, recommend 0.7 for landmark detection
  """
  
  if padding_option == "custom":  # custom padding with black pixels
    cropped_img_array = save_one_box(bbox_xyxy, origin_array, square=False, save=False)
    result_array = resize_with_padding(cropped_img_array, target_size, padding_scale)

  elif padding_option == "yolo":  # yolo padding - expand facial areas (squared)
    cropped_img_array = save_one_box(bbox_xyxy, origin_array, square=True, save=False)
    # resize 추가
    # img_pil = Image.fromarray(cropped_img_array)
    # resized_img_pil = img_pil.resize(target_size, Image.BICUBIC)
    # result_array = np.array(resized_img_pil)
    result_array = resize_with_padding(cropped_img_array, target_size, padding_scale)
      
  else: # no resizing, only crop
    result_array = save_one_box(bbox_xyxy, origin_array, square=False, save=False)
    
  return result_array
  