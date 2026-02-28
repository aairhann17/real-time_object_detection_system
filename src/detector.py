import torch
import cv2
import time
import numpy as np
from pathlib import Path

class ObjectDetector:
    def __init__(self, model_name='yolov5s', device='cpu'):
        """
        Initialize object detector
        
        Args:
            model_name: Name of YOLOv5 model (yolov5s, yolov5m, yolov5l)
            device: 'cpu' or 'cuda' for GPU
        """
        self.device = device
        print(f"Loading model on {device}...")
        
        # Load pre-trained YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Model loaded successfully on {device}")
    
    def detect_image(self, image_path):
        """
        Detect objects in a single image
        
        Returns:
            results: Detection results
            inference_time: Time taken for inference (ms)
        """
        # Read image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Start timing
        start_time_ns = time.perf_counter_ns()
        
        # Run inference
        results = self.model(img_rgb)
        
        # Calculate inference time
        inference_time = (time.perf_counter_ns() - start_time_ns) / 1_000_000  # Convert ns to ms
        
        return results, inference_time
    
    def detect_video(self, video_path, output_path=None):
        """
        Detect objects in video frames
        
        Returns:
            fps: Average frames per second
            total_frames: Total frames processed
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_times = []
        frame_count = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Measure inference time
            start_time_ns = time.perf_counter_ns()
            results = self.model(frame_rgb)
            inference_time = (time.perf_counter_ns() - start_time_ns) / 1_000_000
            
            frame_times.append(inference_time)
            frame_count += 1
            
            # Render results on frame
            rendered_frame = np.array(results.render()[0])
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
            
            if writer:
                writer.write(rendered_frame)
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                avg_time = np.mean(frame_times[-30:])
                print(f"Processed {frame_count}/{total_frames} frames | "
                      f"Avg inference: {avg_time:.2f}ms")
        
        cap.release()
        if writer:
            writer.release()
        
        avg_time_ms = max(float(np.mean(frame_times)), 1e-6)
        avg_fps = 1000 / avg_time_ms  # Convert ms to FPS
        
        return {
            'avg_fps': avg_fps,
            'total_frames': frame_count,
            'avg_inference_time': avg_time_ms,
            'min_inference_time': np.min(frame_times),
            'max_inference_time': np.max(frame_times)
        }