import argparse
from os import makedirs, listdir
from os.path import exists, isfile
from pathlib import Path
import random
import shutil
import warnings

"""
<FROM> 

/anger
    /labeled
      /train
      /validation
    /raw
      /train
      /validation
/anxiety
    /labeled
      /train
      /validation
    /raw
      /train
      /validation
/test_set
    /anger
    /anxiety
    anger.json
    anxiety.json

<TO>

/images
    /train
        /anger
        /happy
    /tst
        /anger
        /happy
/labels
    /train
        /anger
        /happy
    /tst
        /anger
        /happy
"""

EMOTIONS = ("anger", "anxiety", "embarrass", "happy", "normal", "pain", "sad")
DATA_TYPES = ("train", "tst")
ROOT_PATH = Path(__file__).parent # est_wassup_03



def create_features_dirs(dst_data_dir:str):
  new_num = 1
  while exists(dst_data_dir):
    dst_data_dir = dst_data_dir + f"_{new_num}"
    new_num += 1
  makedirs(dst_data_dir)
  
  dir_images = f"{dst_data_dir}/images"
  dir_labels = f"{dst_data_dir}/labels"
  
  # create directories
  # check <TO> structure view above
  makedirs(dir_images, exist_ok=True)
  makedirs(dir_labels, exist_ok=True)
  
  for d_type in DATA_TYPES:
    for emotion in EMOTIONS:
      makedirs(f"{dir_images}/{d_type}/{emotion}", exist_ok=True)
      makedirs(f"{dir_labels}/{d_type}/{emotion}", exist_ok=True)

def move_data_to_features(src_data_dir:str, dst_data_dir:str, selection_ratio:float):
  dir_images = f"{dst_data_dir}/images"
  dir_labels = f"{dst_data_dir}/labels"
  
  for emotion in EMOTIONS:
    dir_emotion = f"{src_data_dir}/{emotion}"
    dir_test_set = f"{src_data_dir}/test_set/{emotion}"
    if not exists(dir_emotion): warnings.warn(f"directory [{dir_emotion}] does not exists")
    elif isfile(dir_emotion): raise warnings.warn(f"[{dir_emotion}] is not a directory. It supposed to be a directory!")
    else:
      dir_img_train = f"{dir_images}/train/{emotion}"
      dir_label_train = f"{dir_labels}/train/{emotion}"
      dir_img_tst = f"{dir_images}/tst/{emotion}"
      dir_label_tst = f"{dir_labels}/tst/{emotion}"
      
      # empty the directory
      if len(listdir(dir_img_train)) != 0: shutil.rmtree(dir_img_train)
      if len(listdir(dir_label_train)) != 0: shutil.rmtree(dir_label_train)
      if len(listdir(dir_img_tst)) != 0: shutil.rmtree(dir_img_tst)
      if len(listdir(dir_label_tst)) != 0: shutil.rmtree(dir_label_tst)
      
      # select full*ratio number of images
      list_of_imgs = listdir(f"{dir_emotion}/raw/train")
      list_of_imgs = list(filter(lambda x: not x.endswith(".zip"), list_of_imgs))
      selected_imgs = random.sample(list_of_imgs, int(len(list_of_imgs) * selection_ratio))
      for img in selected_imgs:
        src_img = f"{dir_emotion}/raw/train/{img}"
        dst_img = f"{dir_images}/{img}"
        shutil.copy(src_img, dst_img)
      
      # copy to the directory
      # shutil.copytree(f"{dir_emotion}/raw/train", dir_img_train, dirs_exist_ok=True)  
      shutil.copytree(f"{dir_emotion}/labeled/train", dir_label_train, dirs_exist_ok=True)
      shutil.copyfile(f"{dir_test_set}.json", f"{dir_label_tst}/{emotion}.json")
      shutil.copytree(dir_test_set, dir_img_tst, dirs_exist_ok=True)
  
def main(cfg):
  src_data_dir = cfg.src_data_path
  dst_data_dir = cfg.dst_data_path
  selection_ratio = cfg.selection_ratio
  
  if not exists(src_data_dir): raise FileNotFoundError(f"directory [{src_data_dir}] does not exists")
  elif isfile(src_data_dir): raise FileNotFoundError(f"[{src_data_dir}] is not a directory. It supposed to be a directory!")
  else:
    # create est_wassup_03/features
    create_features_dirs(dst_data_dir)
    
    # move data to features
    move_data_to_features(src_data_dir, dst_data_dir, selection_ratio)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--selection-ratio", type=float, default=0.8)
  parser.add_argument("--src-data-path", type=str, default="/home/KDT-admin/data")
  parser.add_argument("--dst-data-path", type=str, default="/home/KDT-admin/work/selected-images")
  
  config = parser.parse_args()
  main(config)