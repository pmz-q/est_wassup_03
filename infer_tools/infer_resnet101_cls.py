import os
from core.configs import Resnet101Config
from torchvision.io import read_image
from torchvision.models import resnet101, ResNet101_Weights
from rich.progress import track


def resnet101_cls(cfg: Resnet101Config):
  """
  Code reference: https://pytorch.org/vision/stable/models.html
  """
  src_dir = cfg.infer_config.src_dir
  dst_dir = cfg.get_output_path("inference")[1]
  pretrained = cfg.infer_config.pretrained
  
  # 1. Initialize model with the best avaiable weight
  weights = ResNet101_Weights[pretrained]
  model = resnet101(weights=weights)
  model.eval()
  
  # 2. Initialize the inference transforms
  preprocess = weights.transforms()
  
  # TODO: CROP THEM FIST THEN DO INFERENCE
  os.makedirs(dst_dir, exist_ok=True)
  print(dst_dir)
  
  for root, dirs, files in os.walk(src_dir):
    for file in track(files):
      try:
        img_path = f"{root}/{file}"
        img = read_image(img_path)
      except IOError:
        # not an image file
        continue
  
      # 3. Apply inference preprocessing transforms
      batch = preprocess(img).unsqueeze(0)
      
      # 4. Inference with the model
      prediction = model(batch).squeeze(0).softmax(0)
      class_id = prediction.argmax().item()
      score = prediction[class_id].item()
      category_name = weights.meta["categories"][class_id]
      
      # TODO: 5. Save the classification result
      result = {
        "image_filename": file,
        "class_id": class_id,
        "score": score,
        "category_name": category_name,
      }
      
      print(result)
      