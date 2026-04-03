"""
benchmark.py – CPU vs GPU inference latency benchmarking for YOLOv5 detectors.

Runs a configurable number of inference iterations on a fixed test image using
both the CPU and (when available) a CUDA GPU, then reports detailed latency
statistics and a GPU speedup multiplier.

Typical usage:
    python -m src.benchmark
    # or from Python:
    from src.benchmark import PerformanceBenchmark
    bm = PerformanceBenchmark("data/sample_images/test.jpg")
    bm.run_benchmark(num_iterations=100)
    bm.save_results("results/benchmark_results.json")
"""
import torch
from src.detector import ObjectDetector
from src.utils import summarize_latencies, compute_speedup, save_json

class PerformanceBenchmark:
    """Measures and compares YOLOv5 inference latency on CPU and GPU.

    For each available compute device the benchmark:
      1. Runs a short warm-up phase to prime the JIT compiler and GPU caches.
      2. Executes *num_iterations* timed inference calls.
      3. Summarises the collected latencies (mean, median, std, p95, p99, …).

    Results are stored in `self.results` and can optionally be persisted to a
    JSON file via :meth:`save_results`.

    Attributes:
        test_image (str | Path): Path to the image used as input for every
            benchmark iteration.
        results (dict): Populated by :meth:`run_benchmark` with per-device
            latency statistics and, when both devices are available, a
            ``'speedup'`` entry.
    """
    def __init__(self, test_image_path):
        """Initialise the benchmark with a test image path.

        Args:
            test_image_path (str | Path): Path to the image that will be used
                as input for every inference iteration.
        """
        self.test_image = test_image_path
        self.results = {}  # Populated by run_benchmark()
    
    def run_benchmark(self, num_iterations=100):
        """Run inference benchmarks on all available compute devices.

        Iterates over the set of available devices (always 'cpu', plus 'cuda'
        when a compatible GPU is present).  For each device a fresh
        ObjectDetector is created, 5 warm-up iterations are discarded, then
        *num_iterations* timed calls are collected.  A GPU-vs-CPU speedup
        ratio is computed when both devices are available.

        Args:
            num_iterations (int): Number of timed inference calls per device.
                Defaults to 100.

        Returns:
            dict: Benchmark results keyed by device name ('cpu', 'cuda') and,
                when applicable, by 'speedup'.  Each device entry contains the
                keys returned by :func:`~src.utils.summarize_latencies`.
        """
        devices = ['cpu']
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            devices.append('cuda')
            print("GPU (CUDA) detected! Running GPU benchmarks...")
        else:
            print("No GPU detected. Running CPU-only benchmarks.")
        
        for device in devices:
            print(f"\n{'='*50}")
            print(f"Benchmarking on {device.upper()}")
            print(f"{'='*50}")
            
            # Create a dedicated detector for this device
            detector = ObjectDetector(model_name='yolov5s', device=device)
            
            inference_times = []  # Stores per-iteration latencies in milliseconds
            
            # Warm-up runs: the first few iterations are typically slower due to
            # lazy JIT compilation and GPU cache warm-up; discard their timings
            print("Running warm-up iterations...")
            for i in range(5):
                _, _ = detector.detect_image(self.test_image)
            
            # Main benchmark loop: collect *num_iterations* timed samples
            print(f"Running {num_iterations} iterations...")
            for i in range(num_iterations):
                _, inf_time = detector.detect_image(self.test_image)
                inference_times.append(inf_time)
                
                # Print intermediate progress every 20 iterations
                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{num_iterations} iterations")
            
            # Aggregate statistics (mean, median, std, min, max, p95, p99)
            self.results[device] = summarize_latencies(inference_times)

        # Compute GPU speedup ratios when both CPU and CUDA results are present
        if 'cuda' in self.results and 'cpu' in self.results:
            self.results['speedup'] = {
                'mean_x': compute_speedup(self.results['cpu']['mean'], self.results['cuda']['mean']),
                'p95_x': compute_speedup(self.results['cpu']['p95'], self.results['cuda']['p95'])
            }
        
        self.print_results()
        return self.results
    
    def print_results(self):
        """Print a formatted summary of benchmark results to stdout.

        Displays per-device latency statistics in a fixed-width table and,
        when applicable, the GPU speedup ratio relative to CPU.
        """
        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        
        for device in ['cpu', 'cuda']:
            if device not in self.results:
                continue
            metrics = self.results[device]
            print(f"\n{device.upper()} Performance:")
            print(f"  Mean inference time:   {metrics['mean']:.2f} ms")
            print(f"  Median inference time: {metrics['median']:.2f} ms")
            print(f"  Std deviation:         {metrics['std']:.2f} ms")
            print(f"  Min inference time:    {metrics['min']:.2f} ms")
            print(f"  Max inference time:    {metrics['max']:.2f} ms")
            print(f"  95th percentile:       {metrics['p95']:.2f} ms")
            print(f"  99th percentile:       {metrics['p99']:.2f} ms")
        
        # Print GPU speedup summary only when a CUDA device was benchmarked
        if 'cuda' in self.results and 'cpu' in self.results:
            speedup = self.results['speedup']['mean_x']
            print(f"\n{'='*70}")
            print(f"GPU SPEEDUP: {speedup:.2f}x faster than CPU")
            print(f"{'='*70}")
        else:
            print("\nGPU speedup not available (CUDA device not detected).")
    
    def save_results(self, output_path='benchmark_results.json'):
        """Persist the benchmark results dictionary to a JSON file.

        Args:
            output_path (str | Path): Destination file path.  Parent
                directories are created automatically if they do not exist.
                Defaults to 'benchmark_results.json'.
        """
        output_file = save_json(self.results, output_path)
        print(f"\nResults saved to {output_file}")

# Script entry-point: run the benchmark with default settings when executed directly
if __name__ == "__main__":
    # Path to a representative test image used for all iterations
    test_image = "data/sample_images/test.jpg"
    
    benchmark = PerformanceBenchmark(test_image)
    benchmark.run_benchmark(num_iterations=100)
    benchmark.save_results()