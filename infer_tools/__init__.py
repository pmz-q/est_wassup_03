from .infer_yolo import yolo_det
from .infer_yolo_cls import yolo_cls
from .infer_dlib_det import main as dlib_det


__all__ = ["yolo_det", "yolo_cls", "dlib_det"]