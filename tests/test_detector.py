"""
test_detector.py – Unit tests for the ObjectDetector class.

All tests use lightweight stub ("Dummy") objects in place of the real
torch.hub YOLOv5 model so that the test suite can run without network access,
a GPU, or large model weights downloaded.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.detector import ObjectDetector


class DummyResults:
    """Minimal stand-in for a YOLOv5 Detections result object.

    Implements the ``render()`` method expected by the detector so that test
    code can verify the output pipeline without running real inference.

    Attributes:
        _frame (np.ndarray): The image array returned by ``render()``.
    """

    def __init__(self, frame):
        """Store the frame to be returned by render().

        Args:
            frame (np.ndarray): Image array representing the detection output.
        """
        self._frame = frame

    def render(self):
        """Return the stored frame as a single-element list, mimicking YOLOv5.

        Returns:
            list: A list containing the stored frame array.
        """
        return [self._frame]


class DummyModel:
    """Minimal stand-in for a torch.hub YOLOv5 model.

    Provides the ``to()``, ``eval()``, and ``__call__()`` interface expected by
    :class:`~src.detector.ObjectDetector` so that tests can instantiate the
    detector without loading real model weights.
    """

    def to(self, device):
        """No-op device placement; returns self to allow method chaining.

        Args:
            device (str): Ignored compute device string.

        Returns:
            DummyModel: This instance.
        """
        return self

    def eval(self):
        """No-op evaluation-mode switch; returns self to allow method chaining.

        Returns:
            DummyModel: This instance.
        """
        return self

    def __call__(self, image):
        """Simulate a forward pass by echoing the input image as detections.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            DummyResults: A results object that wraps the input image.
        """
        return DummyResults(image)


@pytest.fixture(autouse=True)
def patch_torch_hub_load(monkeypatch):
    """Autouse fixture that replaces torch.hub.load with a DummyModel factory.

    Applied automatically to every test in this module so that no test
    accidentally triggers a real model download or requires a network connection.

    Args:
        monkeypatch: pytest's built-in monkeypatching fixture.
    """
    def _fake_load(*args, **kwargs):
        """Return a DummyModel regardless of the requested model name."""
        return DummyModel()

    monkeypatch.setattr("torch.hub.load", _fake_load)


class TestObjectDetector:
    """Tests for :class:`~src.detector.ObjectDetector`."""

    def test_detector_initialization_cpu(self):
        """Detector should store the requested device and hold a model reference."""
        detector = ObjectDetector(model_name="yolov5s", device="cpu")
        assert detector.device == "cpu"
        assert detector.model is not None

    def test_detect_image_with_dummy_image(self):
        """detect_image should return a positive inference time and non-None results."""
        detector = ObjectDetector(model_name="yolov5s", device="cpu")
        # Create a minimal 640×480 red image to use as test input
        img = Image.new("RGB", (640, 480), color="red")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            results, inference_time = detector.detect_image(tmp_path)
            # Inference time must be strictly positive
            assert inference_time > 0
            # Results object must be truthy (non-None)
            assert results is not None
        finally:
            # Always clean up the temporary file even if an assertion fails
            Path(tmp_path).unlink(missing_ok=True)

    def test_detect_video_returns_stats(self):
        """detect_video should process all frames and return a valid stats dictionary."""
        detector = ObjectDetector(model_name="yolov5s", device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            video_path = temp_dir_path / "input.mp4"
            output_path = temp_dir_path / "output.mp4"

            # Build a synthetic 8-frame video of solid black 320×240 frames
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

            # All 8 written frames should have been processed
            assert stats["total_frames"] == 8
            # Derived FPS must be positive
            assert stats["avg_fps"] > 0
            # Inference time must be recorded
            assert stats["avg_inference_time"] > 0
            # The annotated output file should have been created
            assert output_path.exists()
