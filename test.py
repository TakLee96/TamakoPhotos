image_path = "C:\\Users\\jiaha\\Developer\\TamakoPhotos\\photos\\010.JPG"

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
model = YOLO(model_path)  # Keep model on CPU to avoid CUDA issues

results = model.predict(image_path, save=False)  # Force CPU inference
