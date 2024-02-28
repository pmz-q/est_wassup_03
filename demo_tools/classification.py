import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


ROOT_PATH = Path(__file__).parent


def get_emotion(filename) -> tuple:
  """
  Args:
      filename (str): uploaded filename. the image must be located in demo_tools/input_images.
  Returns:
      tuple: (is_successful, result_path, dict_result)
          is_successful (bool): True if the process was successful. else False
          result_path (str): absoloute path of the saved result image. Error message when the process is not successfuly done.
          
  """
  src = f"{ROOT_PATH}/results/cropped/{filename}"
  dst_path = f"{ROOT_PATH}/results/classified/{filename}"
  cls_weight_path = f"{ROOT_PATH}/weights/classifier_weight/yolo-cls-best.pt"
  
  if not os.path.exists(cls_weight_path): return False, "ERRPR:get_emotion:weight file does not exists."
  if not os.path.isfile(cls_weight_path): return False, "ERROR:get_emotion:given weight is a directory not a file."
  model = YOLO(cls_weight_path)
  
  try:
    Image.open(src)
  except IOError:
    return False, f"ERROR:get_emotion:{src} is not an image file.", None
  
  try:
    result = model(src, verbose=False)[0]
  except Exception as e:
    return False, f"ERROR:get_emotion:{str(e)}", None
  
  # save image with probabilities
  result.save(filename=dst_path)
  
  # save the result into dict_result {e_class_name: probability}
  e_names = result.names
  e_top5 = result.probs.top5
  e_top5_conf = result.probs.top5conf
  
  dict_result = {}
  for i in range(5):
    dict_result[e_names[e_top5[i]]] = "{0:.2f}".format(e_top5_conf[i].item())
  
  return True, dst_path, dict_result

if __name__ == "__main__":
  # test code
  _, _, result = get_emotion("example.jpg")
  print(result)