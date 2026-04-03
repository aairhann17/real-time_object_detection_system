"""
yolov5s.py – Bootstrap script to download and cache YOLOv5-small weights.

Loads the pre-trained YOLOv5-small model from the Ultralytics torch.hub
repository (downloading it on first run) and serialises just the state-dict
to ``models/yolov5s.pt`` for later use without a network connection.

Usage:
    python models/yolov5s.py
"""
import torch

# Load YOLOv5-small from torch.hub; this triggers an automatic download on
# the first run and stores the weights in the torch hub cache directory
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Persist only the model state-dict (parameters + buffers) to disk.
# Saving the state-dict rather than the full model avoids pickle compatibility
# issues when loading across different PyTorch versions.
torch.save(model.state_dict(), 'models/yolov5s.pt')
print("Model downloaded successfully!")