from pathlib import Path

from src.benchmark import PerformanceBenchmark


class FakeDetector:
    def __init__(self, model_name="yolov5s", device="cpu"):
        self.device = device
        self._counter = 0

    def detect_image(self, image_path):
        self._counter += 1
        if self.device == "cpu":
            return None, 50.0 + float(self._counter % 5)
        return None, 10.0 + float(self._counter % 3)


def test_benchmark_cpu_only(monkeypatch):
    monkeypatch.setattr("src.benchmark.ObjectDetector", FakeDetector)
    monkeypatch.setattr("src.benchmark.torch.cuda.is_available", lambda: False)

    benchmark = PerformanceBenchmark("data/sample_images/test.jpg")
    results = benchmark.run_benchmark(num_iterations=20)

    assert "cpu" in results
    assert "cuda" not in results
    assert results["cpu"]["p95"] >= results["cpu"]["median"]


def test_benchmark_cpu_gpu_speedup(monkeypatch, tmp_path):
    monkeypatch.setattr("src.benchmark.ObjectDetector", FakeDetector)
    monkeypatch.setattr("src.benchmark.torch.cuda.is_available", lambda: True)

    benchmark = PerformanceBenchmark("data/sample_images/test.jpg")
    results = benchmark.run_benchmark(num_iterations=20)

    assert "cpu" in results
    assert "cuda" in results
    assert "speedup" in results
    assert results["speedup"]["mean_x"] > 1.0
    assert results["speedup"]["p95_x"] > 1.0

    output_path = tmp_path / "benchmark_results.json"
    benchmark.save_results(output_path)
    assert Path(output_path).exists()