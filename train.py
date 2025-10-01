from ultralytics import YOLO
import cv2
import numpy as np
import ultralytics.utils.patches as patches

def safe_imread(f, flags=cv2.IMREAD_COLOR):
    if isinstance(f, (bytes, bytearray)):
        f = np.frombuffer(f, np.uint8)
    if isinstance(f, np.ndarray):
        return cv2.imdecode(f, flags)
    return cv2.imread(str(f), flags)  # fallback to path-based

patches.imread = safe_imread

import torch, os
if torch.cuda.is_available():
    device = 0             # first CUDA GPU; for multi-GPU use "0,1" (string)
    print(f"Using CUDA → {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA not available; using CPU")

model = YOLO("yolo11n.pt")

results = model.train(
    data="data.yaml",
    imgsz=1280,
    epochs=100,
    batch=8,          # raise this on GPU if memory allows
    device=device,    # now actually uses CUDA
    workers=8,
    amp=True,         # mixed precision for speed on CUDA
)
path = model.export(format="onnx")
print("Exportgit add -A ed:", path)