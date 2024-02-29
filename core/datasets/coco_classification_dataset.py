import os.path
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from torchvision import transforms
from typing import Optional, Callable, List, Any, Tuple

class CocoClassificationDataset(VisionDataset):
  def __init__(
    self,
    src_root_path: str,
    ann_path: str,
    transform: Optional[Callable]=None
  ):
    """
    Args:
      src_root_path (str): Root directory where images are downloaded to.
      ann_path (str): Path to json annotation file.
      transform (Optional[Callable]): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.PILToTensor``
    """
    super().__init__(src_root_path, transform=transform)
    self.transforms = transforms.Compose([
      # transforms.Resize((224 , 224)),
      transforms.ToTensor()
    ])
        
    self.coco = COCO(ann_path)
    self.ids = list(sorted(self.coco.imgs.keys()))

  def _load_image(self, id: int) -> Image.Image:
    path = self.coco.loadImgs(id)[0]["file_name"]
    return self.transforms(Image.open(os.path.join(self.root, path)).convert("RGB"))

  def _load_target(self, id: int) -> List[Any]:
    """
    assumes one annotation per one image
    """
    return self.coco.loadAnns(self.coco.getAnnIds(id))[0]

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    """
    Returns:
        Tuple[Any, Any]: image, category_id
    """
    id = self.ids[index]
    image = self._load_image(id)
    target = self._load_target(id)
    
    return image, target["category_id"]

  def __len__(self) -> int:
        return len(self.ids)