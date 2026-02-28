import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from roboflow import Roboflow


def download_dataset(api_key: str, workspace: str, project: str, version: int, output_dir: Path) -> Path:
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(workspace)
        .project(project)
        .version(version)
        .download("yolov5", location=str(output_dir))
    )
    return Path(dataset.location)


def run_training(yolov5_repo: Path, dataset_yaml: Path, epochs: int, img_size: int, batch_size: int) -> None:
    command = [
        sys.executable,
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
        "yolov5s.pt",
        "--project",
        "../runs",
        "--name",
        "roboflow_finetune",
        "--exist-ok",
    ]
    subprocess.run(command, cwd=str(yolov5_repo), check=True)


def extract_metrics(results_csv: Path) -> dict:
    if not results_csv.exists():
        raise FileNotFoundError(f"Training results file not found: {results_csv}")

    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    headers = [column.strip() for column in lines[0].split(",")]
    values = [value.strip() for value in lines[-1].split(",")]
    row = dict(zip(headers, values))

    return {
        "precision": float(row.get("metrics/precision", 0.0)),
        "recall": float(row.get("metrics/recall", 0.0)),
        "mAP50": float(row.get("metrics/mAP_0.5", 0.0)),
        "mAP50_95": float(row.get("metrics/mAP_0.5:0.95", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv5 using a Roboflow dataset.")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", required=True, type=int, help="Roboflow dataset version")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dataset-dir", default="data/roboflow_dataset")
    parser.add_argument("--yolov5-repo", default="data/yolov5")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("Please set ROBOFLOW_API_KEY before running this script.")

    dataset_dir = Path(args.dataset_dir)
    yolov5_repo = Path(args.yolov5_repo)

    if not yolov5_repo.exists():
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5", str(yolov5_repo)], check=True)

    download_location = download_dataset(
        api_key=api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        output_dir=dataset_dir,
    )

    dataset_yaml = download_location / "data.yaml"
    run_training(
        yolov5_repo=yolov5_repo,
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        img_size=args.img,
        batch_size=args.batch,
    )

    results_csv = Path("runs/roboflow_finetune/results.csv")
    metrics = extract_metrics(results_csv)

    output_path = Path("runs/roboflow_finetune/metrics.json")
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()