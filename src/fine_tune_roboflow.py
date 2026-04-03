"""
fine_tune_roboflow.py – Fine-tune YOLOv5 using a Roboflow-hosted dataset.

This script orchestrates the end-to-end fine-tuning workflow:
  1. Downloads a labelled dataset from Roboflow in YOLOv5 format.
  2. Clones the Ultralytics YOLOv5 repository if it is not already present.
  3. Launches the YOLOv5 training loop via a subprocess call to ``train.py``.
  4. Extracts key evaluation metrics from the ``results.csv`` produced by
     the training run and writes them to ``metrics.json``.

Environment variables:
    ROBOFLOW_API_KEY: Your Roboflow API key (required).  Set this before
        running the script; the script will raise ``EnvironmentError`` if
        it is absent.

Usage:
    python -m src.fine_tune_roboflow \\
        --workspace <workspace-slug> \\
        --project   <project-slug>   \\
        --version   <dataset-version>
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from roboflow import Roboflow


def download_dataset(api_key: str, workspace: str, project: str, version: int, output_dir: Path) -> Path:
    """Download a Roboflow dataset in YOLOv5 format.

    Authenticates with the Roboflow API, navigates the workspace → project →
    version hierarchy, then downloads the annotations and images in
    YOLOv5-compatible format to *output_dir*.

    Args:
        api_key (str): Roboflow API key for authentication.
        workspace (str): Roboflow workspace slug (visible in the project URL).
        project (str): Roboflow project slug.
        version (int): Dataset version number to download.
        output_dir (Path): Local directory where the dataset will be saved.

    Returns:
        Path: Path to the directory containing the downloaded dataset,
            including the ``data.yaml`` configuration file required by
            YOLOv5 training.
    """
    # Authenticate and navigate the Roboflow project/version hierarchy
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download("yolov5", location=str(output_dir))  # Download in YOLOv5 annotation format
    )
    return Path(dataset.location)


def run_training(yolov5_repo: Path, dataset_yaml: Path, epochs: int, img_size: int, batch_size: int) -> None:
    """Launch YOLOv5 training via a subprocess call to ``train.py``.

    Constructs the training command from the supplied parameters and runs it
    inside the cloned YOLOv5 repository directory.  Training output and
    weights are written to ``../runs/roboflow_finetune`` relative to the
    repository root.

    Args:
        yolov5_repo (Path): Path to the locally cloned ``ultralytics/yolov5``
            repository.
        dataset_yaml (Path): Path to the dataset ``data.yaml`` file downloaded
            by :func:`download_dataset`.
        epochs (int): Number of training epochs.
        img_size (int): Input image size (width and height, in pixels).
        batch_size (int): Mini-batch size for gradient updates.

    Raises:
        subprocess.CalledProcessError: If the training subprocess exits with a
            non-zero return code.
    """
    # Build the argument list for the YOLOv5 train.py script
    command = [
        sys.executable,   # Use the same Python interpreter that launched this script
        "train.py",
        "--img",
        str(img_size),
        "--batch",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--data",
        str(dataset_yaml),
        "--weights",
        "yolov5s.pt",     # Start from pre-trained YOLOv5-small weights
        "--project",
        "../runs",
        "--name",
        "roboflow_finetune",
        "--exist-ok",     # Allow overwriting a previous run with the same name
    ]
    # Execute training as a child process; check=True raises on failure
    subprocess.run(command, cwd=str(yolov5_repo), check=True)


def extract_metrics(results_csv: Path) -> dict:
    """Parse YOLOv5 training metrics from the ``results.csv`` log file.

    Reads the CSV produced by ``train.py``, extracts values from the final
    row (which corresponds to the last training epoch), and returns the four
    standard COCO-style evaluation metrics.

    Args:
        results_csv (Path): Path to the ``results.csv`` file written by the
            YOLOv5 training run.

    Returns:
        dict: Dictionary with the following float-valued keys:
            - ``precision``  – precision at the default IoU threshold.
            - ``recall``     – recall at the default IoU threshold.
            - ``mAP50``      – mean Average Precision at IoU 0.5.
            - ``mAP50_95``   – mean Average Precision across IoU 0.5:0.95.

    Raises:
        FileNotFoundError: If *results_csv* does not exist.
    """
    if not results_csv.exists():
        raise FileNotFoundError(f"Training results file not found: {results_csv}")

    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    # First line contains column headers; strip surrounding whitespace from each
    headers = [column.strip() for column in lines[0].split(",")]
    # Last line contains metrics for the final training epoch
    values = [value.strip() for value in lines[-1].split(",")]
    # Zip headers and values into a lookup dictionary for named access
    row = dict(zip(headers, values))

    return {
        "precision": float(row.get("metrics/precision", 0.0)),
        "recall": float(row.get("metrics/recall", 0.0)),
        "mAP50": float(row.get("metrics/mAP_0.5", 0.0)),
        "mAP50_95": float(row.get("metrics/mAP_0.5:0.95", 0.0)),
    }


def main() -> None:
    """Entry point for the fine-tuning CLI.

    Parses command-line arguments, validates the environment, then orchestrates
    dataset download → training → metrics extraction in sequence.  The final
    metrics are written to ``runs/roboflow_finetune/metrics.json`` and printed
    to stdout as JSON.
    """
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv5 using a Roboflow dataset.")
    # Required positional-style arguments identifying the Roboflow dataset
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", required=True, type=int, help="Roboflow dataset version")
    # Optional training hyper-parameters with sensible defaults
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    # File-system paths (overridable for custom layouts)
    parser.add_argument("--dataset-dir", default="data/roboflow_dataset")
    parser.add_argument("--yolov5-repo", default="data/yolov5")
    args = parser.parse_args()

    # Require the API key to be provided via environment variable for security
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set ROBOFLOW_API_KEY before running this script.")

    dataset_dir = Path(args.dataset_dir)
    yolov5_repo = Path(args.yolov5_repo)

    # Clone the YOLOv5 repository if it has not already been downloaded
    if not yolov5_repo.exists():
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5", str(yolov5_repo)], check=True)

    # Step 1: Download the Roboflow dataset in YOLOv5 annotation format
    download_location = download_dataset(
        api_key=api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        output_dir=dataset_dir,
    )

    # Step 2: Launch the YOLOv5 training loop
    dataset_yaml = download_location / "data.yaml"  # Config file generated by Roboflow
    run_training(
        yolov5_repo=yolov5_repo,
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        img_size=args.img,
        batch_size=args.batch,
    )

    # Step 3: Parse and save evaluation metrics from the training log
    results_csv = Path("runs/roboflow_finetune/results.csv")
    metrics = extract_metrics(results_csv)

    # Write metrics to JSON alongside the training artefacts and print to stdout
    output_path = Path("runs/roboflow_finetune/metrics.json")
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()