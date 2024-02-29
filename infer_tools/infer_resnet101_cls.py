import os
from core.configs import Resnet101Config
from torchvision.io import read_image
from core.models import ResNet101Cls
from tqdm import tqdm
import pandas as pd


def resnet101_cls(cfg: Resnet101Config):
  """
  Code reference: https://pytorch.org/vision/stable/models.html
  """
  src_dir = cfg.infer_config.src_dir
  dst_dir = cfg.get_output_path("inference")[1]
  weights = cfg.infer_config.pretrained
  
  # 1. Initialize model with the best avaiable weight
  model = ResNet101Cls(weights=weights, num_classes=7)
  model.eval()
  
  # 2. Initialize the inference transforms
  preprocess = weights.transforms()
  
  # TODO: CROP THEM FIST THEN DO INFERENCE
  os.makedirs(dst_dir, exist_ok=True)
  print(dst_dir)
  
  result = {
    "image_filename": [],
    "class_id": [],
    "score": [],
    "category_name": [],
  }
  for root, dirs, files in os.walk(src_dir):
    for file in tqdm(files):
      try:
        img_path = f"{root}/{file}"
        img = read_image(img_path)
      except IOError:
        # not an image file
        continue
      
      batch = preprocess(img).unsqueeze(0)
      
      prediction = model(batch).squeeze(0).softmax(0)
      class_id = prediction.argmax().item()
      score = prediction[class_id].item()
      category_name = weights.meta["categories"][class_id]
      
      result["image_filename"].append(file)
      result["class_id"].append(class_id)
      result["score"].append(score)
      result["category_name"].append(category_name)
  pd.DataFrame(result).to_csv(f"{dst_dir}/results.csv")
      