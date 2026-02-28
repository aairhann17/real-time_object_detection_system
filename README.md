# Real-Time Object Detection System (YOLOv5 + Streamlit)

A Streamlit app for image/video object detection with optional CPU vs GPU benchmarking.

## Project Structure

- `app.py`: Streamlit interface (image, video, benchmark tabs)
- `src/detector.py`: YOLOv5 inference for image/video
- `src/benchmark.py`: CPU/GPU benchmark runner (100-iteration compatible)
- `src/fine_tune_roboflow.py`: Roboflow dataset download + YOLOv5 fine-tuning pipeline
- `tests/`: unit tests for detector and benchmark behavior

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a requirements file yet, install minimally:

```bash
pip install torch torchvision torchaudio opencv-python streamlit numpy pytest roboflow yolov5
```

## Run the App

```bash
streamlit run app.py
```

## Benchmark Results (Real Run)

Environment used for these numbers:

- Python: `3.11.5`
- Torch: `2.9.1+cpu`
- CUDA available: `False`
- Iterations: `100`
- Test image: `data/sample_images/test.jpg`

Measured on CPU (from `benchmark_results.json`):

- Mean inference: **54.54 ms**
- Median inference: **54.22 ms**
- 95th percentile: **58.93 ms**
- 99th percentile: **60.42 ms**

### GPU Speedup Status

This machine has no CUDA device (`torch.cuda.is_available() == False`), so a real CPU-vs-GPU speedup metric could not be measured on this host.

As soon as you run the same benchmark on a CUDA-enabled machine, `src/benchmark.py` now records:

- `speedup.mean_x = cpu_mean / gpu_mean`
- `speedup.p95_x = cpu_p95 / gpu_p95`

Run command:

```bash
python -m src.benchmark
```

## Roboflow Fine-Tuning (Custom Dataset)

`src/fine_tune_roboflow.py` automates:

1. Downloading a Roboflow dataset export in YOLOv5 format
2. Cloning YOLOv5 training repo (if missing)
3. Running `train.py`
4. Extracting metrics (`precision`, `recall`, `mAP50`, `mAP50_95`) into `runs/roboflow_finetune/metrics.json`

### Run Fine-Tuning

```bash
set ROBOFLOW_API_KEY=your_key_here
python -m src.fine_tune_roboflow --workspace your-workspace --project your-project --version 1 --epochs 10 --img 640 --batch 8
```

### Current Status in This Workspace

- Roboflow access attempted: **blocked** (no valid `ROBOFLOW_API_KEY` available in environment)
- Dataset/images count: **pending key + dataset selection**
- Achieved mAP: **pending training run**

Once key + dataset are provided, rerun command above and copy resulting values from `runs/roboflow_finetune/metrics.json` into this README.

## Demo Media

- Detection result image (generated from real inference): `data/sample_images/detection_output.jpg`
- Optional Streamlit screenshot/GIF: add under `assets/` and reference here

## Tests

```bash
pytest -v
```

Included coverage:

- Detector initialization and image inference timing
- Video detection stats/output generation
- Benchmark CPU-only and CPU+GPU speedup calculation paths

## Notes

- `venv/` is now ignored via `.gitignore` and should not be committed.
- Suggested GitHub topics: `object-detection`, `yolov5`, `computer-vision`, `pytorch`, `cuda`
