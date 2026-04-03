"""
detector.py – Core object-detection wrapper around YOLOv5.

Exposes the ObjectDetector class which loads a pre-trained YOLOv5 model via
torch.hub and provides convenience methods for running inference on single
images and on video files.
"""
import torch
import cv2
import time
import numpy as np
from pathlib import Path

class ObjectDetector:
    """YOLOv5-based object detector that supports both CPU and CUDA inference.

    Wraps torch.hub YOLOv5 models and adds timing instrumentation so callers
    can measure per-inference latency.  Supports still-image and video-file
    processing.

    Attributes:
        device (str): Compute backend in use, either 'cpu' or 'cuda'.
        model: The loaded YOLOv5 torch.hub model in evaluation mode.
    """
    def __init__(self, model_name='yolov5s', device='cpu'):
        """Initialise the detector by loading a YOLOv5 model.

        The model is downloaded from the Ultralytics torch.hub repository on
        first use and cached locally by PyTorch.

        Args:
            model_name (str): YOLOv5 variant to load.  Supported values are
                'yolov5s' (small/fast), 'yolov5m' (medium), and 'yolov5l'
                (large/accurate).  Defaults to 'yolov5s'.
            device (str): Compute device for inference.  Use 'cpu' for
                CPU-only inference or 'cuda' for GPU acceleration.
                Defaults to 'cpu'.
        """
        self.device = device
        print(f"Loading model on {device}...")
        
        # Download (or load from cache) the pre-trained YOLOv5 weights
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        # Move model parameters to the target device (CPU or GPU)
        self.model.to(device)
        self.model.eval()  # Disable dropout/BatchNorm training behaviour
        
        print(f"Model loaded successfully on {device}")
    
    def detect_image(self, image_path):
        """Run object detection on a single image file.

        Reads the image from disk, converts it from BGR (OpenCV default) to
        RGB (YOLOv5 expectation), then runs a forward pass through the model
        while measuring wall-clock inference time with nanosecond precision.

        Args:
            image_path (str | Path): Path to the input image file.

        Returns:
            tuple:
                results: YOLOv5 Detections object containing bounding boxes,
                    class labels, and confidence scores.
                inference_time (float): Wall-clock time for the forward pass
                    in milliseconds.
        """
        # Load image from disk using OpenCV (returns BGR channel order)
        img = cv2.imread(str(image_path))
        # YOLOv5 expects RGB input, so convert from OpenCV's BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Record start time with nanosecond precision for accurate short-duration measurement
        start_time_ns = time.perf_counter_ns()
        
        # Forward pass through the YOLOv5 model
        results = self.model(img_rgb)
        
        # Compute elapsed time and convert from nanoseconds to milliseconds
        inference_time = (time.perf_counter_ns() - start_time_ns) / 1_000_000
        
        return results, inference_time
    
    def detect_video(self, video_path, output_path=None):
        """Run object detection across every frame of a video file.

        Iterates over frames using OpenCV, runs inference on each, optionally
        draws bounding boxes and writes the annotated frames to a new video
        file.

        Args:
            video_path (str | Path): Path to the input video file.
            output_path (str | Path | None): Destination path for the
                annotated output video.  If ``None``, no output file is
                written.  Defaults to ``None``.

        Returns:
            dict: Aggregate statistics for the processed video:
                - ``avg_fps`` (float): Average frames per second derived from
                  mean per-frame inference time.
                - ``total_frames`` (int): Number of frames processed.
                - ``avg_inference_time`` (float): Mean per-frame inference
                  time in milliseconds.
                - ``min_inference_time`` (float): Fastest single-frame
                  inference time in milliseconds.
                - ``max_inference_time`` (float): Slowest single-frame
                  inference time in milliseconds.
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Read source video properties needed to configure the output writer
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Optionally prepare a VideoWriter to save the annotated output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V codec for .mp4 output
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_times = []  # Collects per-frame inference durations for statistics
        frame_count = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # End of stream or read error – stop processing
                break
            
            # YOLOv5 expects RGB input; OpenCV delivers BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Time each forward pass independently
            start_time_ns = time.perf_counter_ns()
            results = self.model(frame_rgb)
            inference_time = (time.perf_counter_ns() - start_time_ns) / 1_000_000
            
            frame_times.append(inference_time)
            frame_count += 1
            
            # Render bounding boxes onto the frame and convert back to BGR for writing
            rendered_frame = np.array(results.render()[0])
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
            
            if writer:
                writer.write(rendered_frame)
            
            # Print a rolling progress update every 30 frames to avoid log spam
            if frame_count % 30 == 0:
                avg_time = np.mean(frame_times[-30:])
                print(f"Processed {frame_count}/{total_frames} frames | "
                      f"Avg inference: {avg_time:.2f}ms")
        
        # Release file handles
        cap.release()
        if writer:
            writer.release()
        
        # Guard against empty frame_times (e.g. unreadable file) to avoid division by zero
        avg_time_ms = max(float(np.mean(frame_times)), 1e-6)
        avg_fps = 1000 / avg_time_ms  # Convert mean ms/frame to frames per second
        
        return {
            'avg_fps': avg_fps,
            'total_frames': frame_count,
            'avg_inference_time': avg_time_ms,
            'min_inference_time': np.min(frame_times),
            'max_inference_time': np.max(frame_times)
        }