import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def ensure_directory(path: str | Path) -> Path:
	directory = Path(path)
	directory.mkdir(parents=True, exist_ok=True)
	return directory


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
	if not latencies_ms:
		raise ValueError("latencies_ms must contain at least one value")

	values = np.asarray(latencies_ms, dtype=np.float64)
	return {
		"mean": float(np.mean(values)),
		"median": float(np.median(values)),
		"std": float(np.std(values)),
		"min": float(np.min(values)),
		"max": float(np.max(values)),
		"p95": float(np.percentile(values, 95)),
		"p99": float(np.percentile(values, 99)),
	}


def save_json(data: Dict, output_path: str | Path) -> Path:
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2)
	return output_file


def compute_speedup(cpu_ms: float, gpu_ms: float) -> float:
	if cpu_ms <= 0 or gpu_ms <= 0:
		raise ValueError("cpu_ms and gpu_ms must be greater than 0")
	return cpu_ms / gpu_ms
