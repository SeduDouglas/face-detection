import cv2
import cvzone
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device


def load(weights, dnn=False, data=ROOT / 'data/coco.yaml', fp16=False):
  device = select_device('cpu')
  model = DetectMultiBackend(weights, dnn=False, device=device, data=ROOT / 'data/coco.yaml', fp16=False)
  img_size = check_img_size((640, 640), s=model.stride)
  model.warmup(imgsz=(1 if model.pt or model.triton else 1, 3, *img_size))  # warmup

  return model
