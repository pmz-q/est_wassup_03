import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


ROOT_PATH = Path(__file__).parent

def get_emotion(imageArray) -> tuple:
  """
  Args:
    // TODO
  Returns:
    // TODO
  """

  cls_weight_path = f"{ROOT_PATH}/weights/classifier_weight/best.pt"
  
  if not os.path.exists(cls_weight_path): return False, "ERRPR:get_emotion:weight file does not exists."
  if not os.path.isfile(cls_weight_path): return False, "ERROR:get_emotion:given weight is a directory not a file."
  model = YOLO(cls_weight_path)
  
  try:
    result = model(imageArray, verbose=False)[0]
  except Exception as e:
    return False, f"ERROR:get_emotion:{str(e)}", None
  
  save_path = f"{ROOT_PATH}/output_emotion.jpg"
  
  plt.imsave(save_path, cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
  
  return result.plot()


if __name__ == "__main__":
  # test code
  _, _, result = get_emotion("example.jpg")
  print(result)