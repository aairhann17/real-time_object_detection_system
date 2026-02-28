"""Unit tests for ObjectDetector class."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.detector import ObjectDetector


class DummyResults:
    def __init__(self, frame):
        self._frame = frame

    def render(self):
        return [self._frame]


class DummyModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, image):
        return DummyResults(image)


@pytest.fixture(autouse=True)
def patch_torch_hub_load(monkeypatch):
    def _fake_load(*args, **kwargs):
        return DummyModel()

    monkeypatch.setattr("torch.hub.load", _fake_load)


class TestObjectDetector:
    def test_detector_initialization_cpu(self):
        detector = ObjectDetector(model_name="yolov5s", device="cpu")
        assert detector.device == "cpu"
        assert detector.model is not None

    def test_detect_image_with_dummy_image(self):
        detector = ObjectDetector(model_name="yolov5s", device="cpu")
        img = Image.new("RGB", (640, 480), color="red")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            results, inference_time = detector.detect_image(tmp_path)
            assert inference_time > 0
            assert results is not None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_detect_video_returns_stats(self):
        detector = ObjectDetector(model_name="yolov5s", device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            video_path = temp_dir_path / "input.mp4"
            output_path = temp_dir_path / "output.mp4"

            width, height = 320, 240
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (width, height),
            )
            for _ in range(8):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            stats = detector.detect_video(video_path, str(output_path))

            assert stats["total_frames"] == 8
            assert stats["avg_fps"] > 0
            assert stats["avg_inference_time"] > 0
            assert output_path.exists()
