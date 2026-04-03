"""
utils.py – Shared utility helpers for the real-time object detection system.

Provides:
  - Directory creation helpers.
  - Latency summarisation (aggregation of timing lists into statistical metrics).
  - JSON serialisation for experiment results.
  - GPU speedup ratio computation.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def ensure_directory(path: str | Path) -> Path:
	"""Create a directory (and any missing parents) if it does not already exist.

	Args:
		path (str | Path): Target directory path to create.

	Returns:
		Path: The resolved ``Path`` object for the created (or pre-existing)
			directory.
	"""
	directory = Path(path)
	directory.mkdir(parents=True, exist_ok=True)
	return directory


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
	"""Compute descriptive statistics for a list of latency measurements.

	Converts the input list to a 64-bit NumPy array to ensure consistent
	numerical precision, then calculates a standard set of distribution
	metrics useful for performance reporting.

	Args:
		latencies_ms (List[float]): Non-empty sequence of inference times in
			milliseconds.

	Returns:
		Dict[str, float]: Dictionary containing:
			- ``mean``   – arithmetic mean.
			- ``median`` – 50th-percentile value.
			- ``std``    – population standard deviation.
			- ``min``    – minimum observed value.
			- ``max``    – maximum observed value.
			- ``p95``    – 95th-percentile value.
			- ``p99``    – 99th-percentile value.

	Raises:
		ValueError: If *latencies_ms* is empty.
	"""
	if not latencies_ms:
		raise ValueError("latencies_ms must contain at least one value")

	# Cast to float64 for consistent high-precision NumPy arithmetic
	values = np.asarray(latencies_ms, dtype=np.float64)
	return {
		"mean": float(np.mean(values)),
		"median": float(np.median(values)),
		"std": float(np.std(values)),
		"min": float(np.min(values)),
		"max": float(np.max(values)),
		"p95": float(np.percentile(values, 95)),  # 95th percentile latency
		"p99": float(np.percentile(values, 99)),  # 99th percentile latency
	}


def save_json(data: Dict, output_path: str | Path) -> Path:
	"""Serialise a dictionary to a pretty-printed JSON file.

	Creates any missing parent directories before writing so callers do not
	need to pre-create the output folder.

	Args:
		data (Dict): JSON-serialisable dictionary to write.
		output_path (str | Path): Destination file path (including filename).

	Returns:
		Path: The resolved ``Path`` of the written file.
	"""
	output_file = Path(output_path)
	# Ensure the parent directory exists before attempting to write
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2)  # indent=2 for human-readable formatting
	return output_file


def compute_speedup(cpu_ms: float, gpu_ms: float) -> float:
	"""Calculate the GPU speedup ratio relative to CPU for a given metric.

	Returns how many times faster the GPU is compared to the CPU for the same
	inference workload.  A value greater than 1.0 means the GPU is faster.

	Args:
		cpu_ms (float): CPU latency in milliseconds (must be > 0).
		gpu_ms (float): GPU latency in milliseconds (must be > 0).

	Returns:
		float: Speedup ratio (cpu_ms / gpu_ms).  A result of 5.0 means the
			GPU is five times faster than the CPU.

	Raises:
		ValueError: If either argument is not strictly positive.
	"""
	if cpu_ms <= 0 or gpu_ms <= 0:
		raise ValueError("cpu_ms and gpu_ms must be greater than 0")
	# Higher ratio → greater GPU advantage
	return cpu_ms / gpu_ms
