import torch
from src.detector import ObjectDetector
from src.utils import summarize_latencies, compute_speedup, save_json

class PerformanceBenchmark:
    def __init__(self, test_image_path):
        self.test_image = test_image_path
        self.results = {}
    
    def run_benchmark(self, num_iterations=100):
        """
        Compare CPU vs GPU performance
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
            
            # Initialize detector
            detector = ObjectDetector(model_name='yolov5s', device=device)
            
            inference_times = []
            
            # Warm-up runs (first few runs are slower)
            print("Running warm-up iterations...")
            for i in range(5):
                _, _ = detector.detect_image(self.test_image)
            
            # Actual benchmark
            print(f"Running {num_iterations} iterations...")
            for i in range(num_iterations):
                _, inf_time = detector.detect_image(self.test_image)
                inference_times.append(inf_time)
                
                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{num_iterations} iterations")
            
            # Calculate statistics
            self.results[device] = summarize_latencies(inference_times)

        if 'cuda' in self.results and 'cpu' in self.results:
            self.results['speedup'] = {
                'mean_x': compute_speedup(self.results['cpu']['mean'], self.results['cuda']['mean']),
                'p95_x': compute_speedup(self.results['cpu']['p95'], self.results['cuda']['p95'])
            }
        
        self.print_results()
        return self.results
    
    def print_results(self):
        """Print formatted benchmark results"""
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
        
        # Calculate speedup if GPU results exist
        if 'cuda' in self.results and 'cpu' in self.results:
            speedup = self.results['speedup']['mean_x']
            print(f"\n{'='*70}")
            print(f"GPU SPEEDUP: {speedup:.2f}x faster than CPU")
            print(f"{'='*70}")
        else:
            print("\nGPU speedup not available (CUDA device not detected).")
    
    def save_results(self, output_path='benchmark_results.json'):
        """Save results to JSON file"""
        output_file = save_json(self.results, output_path)
        print(f"\nResults saved to {output_file}")

# Run benchmark
if __name__ == "__main__":
    # You'll need a test image
    test_image = "data/sample_images/test.jpg"
    
    benchmark = PerformanceBenchmark(test_image)
    benchmark.run_benchmark(num_iterations=100)
    benchmark.save_results()