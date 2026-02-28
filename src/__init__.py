"""
Real-time Object Detection System
GPU-accelerated object detection using YOLOv5
"""

from src.detector import ObjectDetector
from src.benchmark import PerformanceBenchmark

__all__ = [
    'ObjectDetector',
    'PerformanceBenchmark',
]

__version__ = '1.0.0'