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
# if torch.cuda.is_available():
#     device = 0             # first CUDA GPU; for multi-GPU use "0,1" (string)
#     print(f"Using CUDA → {torch.cuda.get_device_name(0)}")
# else:
#     device = "cpu"
#     print("CUDA not available; using CPU")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    num_gpus = torch.cuda.device_count()
    device = ",".join(str(i) for i in range(num_gpus))  # e.g., "0,1,2,3,4,5,6,7"
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    print(f"Using {num_gpus} CUDA GPUs → {gpu_names}")
else:
    device = "cpu"
    print("CUDA not available; using CPU")

model = YOLO("yolov8s.pt")

results = model.train(
    data="data.yaml",
    imgsz=1280,
    epochs=250,
    batch=16,          # raise this on GPU if memory allows
    device=device,    # now actually uses CUDA
    workers=8,
    amp=True,         # mixed precision for speed on CUDA
)
path = model.export(format="onnx")
print("Exportgit add -A ed:", path)