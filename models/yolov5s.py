# Quick script to download YOLOv5
import torch

# This downloads the model automatically
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
torch.save(model.state_dict(), 'models/yolov5s.pt')
print("Model downloaded successfully!")