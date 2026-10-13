from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
REVISION_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts import benchmark_20_methods_8_datasets as metrics


MANIFEST = REVISION_ROOT / "manifests" / "frontier_staged_manifest.csv"
RESULT_ROOT = REVISION_ROOT / "raw_results" / "frontier"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def runtime_from_log(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\(([0-9.]+)s/image\)", text)
    if match:
        return 1000.0 * float(match.group(1))
    # Reference-backend runners log one measured latency per image, e.g.
    # ``HRMamba 287/287 1436.0 ms 256x144``.
    per_image = [float(value) for value in re.findall(r"\b[0-9]+/[0-9]+\s+([0-9.]+)\s+ms\b", text)]
    return float(np.mean(per_image)) if per_image else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path)
    parser.add_argument("--runtime-note", default="")
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--auxiliary-checkpoint-sha256", default="")
    parser.add_argument("--sampling-steps", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    runtime_ms = runtime_from_log(args.runtime_log)
    for index, item in enumerate(read_csv(MANIFEST), 1):
        output_path = args.output_dir / f"{item['staged_token']}.png"
        if not output_path.exists():
            rows.append(
                {
                    "method_id": args.method_id,
                    "method": args.method_name,
                    "family": args.family,
                    "dataset": item["dataset"],
                    "sample_key": item["sample_key"],
                    "cluster_id": item["cluster_id"],
                    "status": "missing_output",
                    "error": str(output_path),
                }
            )
            continue
        output = metrics.read_bgr(output_path)
        source = metrics.read_bgr(Path(item["staged_input"]))
        if output.shape[:2] != source.shape[:2]:
            output = cv2.resize(
                output, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_CUBIC
            )
        reference = (
            metrics.read_bgr(Path(item["staged_reference"]))
            if item["staged_reference"]
            else None
        )
        row: dict[str, object] = {
            "method_id": args.method_id,
            "method": args.method_name,
            "family": args.family,
            "authority": "official_repository_and_official_checkpoint",
            "implementation_class": "official_repository_and_official_checkpoint",
            "device": "GPU",
            "dataset": item["dataset"],
            "sample_key": item["sample_key"],
            "cluster_id": item["cluster_id"],
            "input": item["staged_input"],
            "output": str(output_path),
            "reference": item["staged_reference"],
            "width": output.shape[1],
            "height": output.shape[0],
            "runtime_ms": runtime_ms if runtime_ms is not None else "",
            "sampling_steps": args.sampling_steps if args.sampling_steps else "",
            "checkpoint_sha256": args.checkpoint_sha256,
            "auxiliary_checkpoint_sha256": args.auxiliary_checkpoint_sha256,
            "status": "ok",
            "error": "",
        }
        row.update(
            metrics.reference_metrics(output, reference)
            if reference is not None
            else {"psnr": "", "ssim": "", "mae": "", "delta_e2000": ""}
        )
        row.update(metrics.no_reference_metrics(output))
        rows.append(row)
        if index % 25 == 0 or index == 287:
            print(f"evaluated {index}/287", flush=True)

    successful = [row for row in rows if row["status"] == "ok"]
    summary = metrics.aggregate(successful)
    for row in summary:
        row["method_id"] = args.method_id
        row["family"] = args.family
    stem = args.method_id.lower()
    write_csv(RESULT_ROOT / f"{stem}_per_image.csv", rows)
    write_csv(RESULT_ROOT / f"{stem}_summary.csv", summary)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method_id": args.method_id,
        "method": args.method_name,
        "family": args.family,
        "manifest": str(MANIFEST),
        "expected_images": 287,
        "successful_images": len(successful),
        "missing_or_failed_images": len(rows) - len(successful),
        "runtime_ms_per_image_from_log": runtime_ms,
        "runtime_note": args.runtime_note,
        "sampling_steps": args.sampling_steps,
        "checkpoint_sha256": args.checkpoint_sha256,
        "auxiliary_checkpoint_sha256": args.auxiliary_checkpoint_sha256,
        "metric_code": str(WORKSPACE / "scripts" / "benchmark_20_methods_8_datasets.py"),
    }
    (RESULT_ROOT / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
