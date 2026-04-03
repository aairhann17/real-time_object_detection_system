"""
test_benchmark.py – Unit tests for the PerformanceBenchmark class.

Uses a lightweight FakeDetector stub to avoid loading real YOLOv5 weights,
and monkeypatches ``torch.cuda.is_available`` to exercise both the CPU-only
and CPU+GPU code paths without requiring physical GPU hardware.
"""
from pathlib import Path

from src.benchmark import PerformanceBenchmark


class FakeDetector:
    """Stub detector that returns deterministic, predictable latency values.

    Replaces :class:`~src.detector.ObjectDetector` during tests so that the
    benchmark logic can be exercised without loading real model weights or
    performing actual inference.

    CPU latencies cycle through the range ``[50, 54]`` ms and GPU latencies
    through ``[10, 12]`` ms, ensuring the GPU always appears faster.

    Attributes:
        device (str): Compute device this fake detector was created for.
        _counter (int): Call counter used to produce varied latency values.
    """

    def __init__(self, model_name="yolov5s", device="cpu"):
        """Initialise the stub with a device identifier and a call counter.

        Args:
            model_name (str): Ignored; present for interface compatibility.
            device (str): Device tag stored for latency lookup ('cpu'/'cuda').
        """
        self.device = device
        self._counter = 0  # Incremented on every detect_image call

    def detect_image(self, image_path):
        """Return a fake (None, latency) pair without performing real inference.

        Latency values cycle deterministically to simulate realistic variation:
          - CPU: values in [50, 54] ms
          - GPU: values in [10, 12] ms

        Args:
            image_path (str | Path): Ignored input path (kept for interface
                compatibility with ObjectDetector).

        Returns:
            tuple: ``(None, float)`` where the float is the simulated latency
                in milliseconds.
        """
        self._counter += 1
        if self.device == "cpu":
            return None, 50.0 + float(self._counter % 5)  # Cycles 50–54 ms
        return None, 10.0 + float(self._counter % 3)      # Cycles 10–12 ms


def test_benchmark_cpu_only(monkeypatch):
    """Benchmark should collect CPU stats and omit 'cuda' when CUDA is unavailable.

    Monkeypatches ObjectDetector and cuda availability so the benchmark runs
    quickly with fake data instead of real inference.
    """
    # Replace the real detector and CUDA check with lightweight stubs
    monkeypatch.setattr("src.benchmark.ObjectDetector", FakeDetector)
    monkeypatch.setattr("src.benchmark.torch.cuda.is_available", lambda: False)

    benchmark = PerformanceBenchmark("data/sample_images/test.jpg")
    results = benchmark.run_benchmark(num_iterations=20)

    # CPU results must be present
    assert "cpu" in results
    # CUDA results must be absent when no GPU is available
    assert "cuda" not in results
    # p95 must be at least as large as the median (basic sanity check)
    assert results["cpu"]["p95"] >= results["cpu"]["median"]


def test_benchmark_cpu_gpu_speedup(monkeypatch, tmp_path):
    """Benchmark should report a GPU speedup >1.0 when CUDA is simulated as available.

    Also verifies that save_results() writes a valid JSON file to the given path.

    Args:
        monkeypatch: pytest fixture for runtime attribute patching.
        tmp_path (Path): pytest-provided temporary directory for output files.
    """
    # Simulate a machine with both CPU and GPU available
    monkeypatch.setattr("src.benchmark.ObjectDetector", FakeDetector)
    monkeypatch.setattr("src.benchmark.torch.cuda.is_available", lambda: True)

    benchmark = PerformanceBenchmark("data/sample_images/test.jpg")
    results = benchmark.run_benchmark(num_iterations=20)

    # Both device entries must be present
    assert "cpu" in results
    assert "cuda" in results
    # The speedup dictionary must be computed when both devices are available
    assert "speedup" in results
    # FakeDetector produces CPU latencies ~5× higher than GPU latencies, so
    # mean_x and p95_x must both be greater than 1.0
    assert results["speedup"]["mean_x"] > 1.0
    assert results["speedup"]["p95_x"] > 1.0

    # Verify that save_results() creates a valid file at the specified path
    output_path = tmp_path / "benchmark_results.json"
    benchmark.save_results(output_path)
    assert Path(output_path).exists()