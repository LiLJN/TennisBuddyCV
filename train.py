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

model = YOLO('yolo11s.pt')

device = "cpu"
try:
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass

results = model.train(
    data='data.yaml',
    imgsz=1280,
    epochs=100,
    batch=2,
    device=device,
)

path = model.export(format="onnx")