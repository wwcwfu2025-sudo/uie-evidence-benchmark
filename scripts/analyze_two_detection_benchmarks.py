from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


WORKSPACE = Path(__file__).resolve().parents[2]
REVISION_ROOT = Path(__file__).resolve().parents[1]
TRASHCAN_SUMMARY = WORKSPACE / "benchmark_assets" / "90_trashcan_detection_metrics.csv"
TRASHCAN_BOOTSTRAP = WORKSPACE / "benchmark_assets" / "101_trashcan_detection_paired_bootstrap.csv"
URPC_SUMMARY = REVISION_ROOT / "raw_results" / "detection_urpc2020" / "urpc2020_yolov12n_summary.csv"
URPC_BOOTSTRAP = REVISION_ROOT / "raw_results" / "detection_urpc2020" / "urpc2020_yolov12n_paired_bootstrap.csv"
OUT = REVISION_ROOT / "analysis"


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


def normalize_summary(
    rows: list[dict[str, str]], dataset: str, detector: str
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "dataset": dataset,
                "detector": detector,
                "method_id": row["method_id"],
                "method": row["method"],
                "n_images": int(row["n_images"]),
                "n_ground_truth": int(row["n_ground_truth"]),
                "bbox_map_50_95": float(row["bbox_map_50_95"]),
                "bbox_map_50": float(row["bbox_map_50"]),
                "bbox_map_75": float(row["bbox_map_75"]),
                "f1_conf025_iou050": float(row["f1_conf025_iou050"]),
                "delta_map_50_95_pp_vs_raw": float(row["delta_map_50_95_pp_vs_raw"]),
                "delta_f1_pp_vs_raw": 100.0
                * (
                    float(row["f1_conf025_iou050"])
                    - float(next(item for item in rows if item["method_id"] == "C000")["f1_conf025_iou050"])
                ),
            }
        )
    return result


def main() -> None:
    combined = normalize_summary(
        read_csv(TRASHCAN_SUMMARY), "TrashCan validation", "YOLOv8m-seg (bbox head)"
    ) + normalize_summary(
        read_csv(URPC_SUMMARY), "URPC2020 test", "IEUOD YOLOv12n"
    )
    write_csv(OUT / "two_detection_benchmarks_summary.csv", combined)

    bootstrap_rows: list[dict[str, object]] = []
    for dataset, detector, path in (
        ("TrashCan validation", "YOLOv8m-seg (bbox head)", TRASHCAN_BOOTSTRAP),
        ("URPC2020 test", "IEUOD YOLOv12n", URPC_BOOTSTRAP),
    ):
        for row in read_csv(path):
            bootstrap_rows.append(
                {
                    "dataset": dataset,
                    "detector": detector,
                    **row,
                }
            )
    write_csv(OUT / "two_detection_benchmarks_paired_f1_ci.csv", bootstrap_rows)

    names = {row["method_id"]: row["method"] for row in combined}
    method_ids = sorted(names)
    ranks: dict[tuple[str, str], float] = {}
    for dataset in sorted({row["dataset"] for row in combined}):
        selected = [row for row in combined if row["dataset"] == dataset]
        values = np.asarray([float(row["bbox_map_50_95"]) for row in selected])
        dataset_ranks = rankdata(-values, method="average")
        for row, rank in zip(selected, dataset_ranks):
            ranks[(dataset, str(row["method_id"]))] = float(rank)

    stability: list[dict[str, object]] = []
    datasets = sorted({row["dataset"] for row in combined})
    for method_id in method_ids:
        selected = [row for row in combined if row["method_id"] == method_id]
        map_deltas = [float(row["delta_map_50_95_pp_vs_raw"]) for row in selected]
        f1_deltas = [float(row["delta_f1_pp_vs_raw"]) for row in selected]
        method_ranks = [ranks[(dataset, method_id)] for dataset in datasets]
        stability.append(
            {
                "method_id": method_id,
                "method": names[method_id],
                "n_dataset_detector_pairs": len(selected),
                "mean_map_delta_pp_vs_raw": float(np.mean(map_deltas)),
                "best_map_delta_pp_vs_raw": float(np.max(map_deltas)),
                "worst_map_delta_pp_vs_raw": float(np.min(map_deltas)),
                "map_improvement_pair_count": int(np.sum(np.asarray(map_deltas) > 0)),
                "map_degradation_pair_count": int(np.sum(np.asarray(map_deltas) < 0)),
                "mean_f1_delta_pp_vs_raw": float(np.mean(f1_deltas)),
                "mean_map_rank": float(np.mean(method_ranks)),
                "rank_range": float(np.max(method_ranks) - np.min(method_ranks)),
            }
        )
    write_csv(OUT / "two_detection_benchmarks_stability.csv", stability)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "design": "two datasets and two detector architectures; detector parameters held constant within each benchmark",
        "dataset_detector_pairs": [
            "TrashCan validation + YOLOv8m-seg bbox head",
            "URPC2020 test + IEUOD YOLOv12n",
        ],
        "methods_including_raw": len(method_ids),
        "interpretation": (
            "Cross-benchmark consistency is descriptive because there are only two dataset-detector pairs; "
            "paired within-benchmark uncertainty is reported separately."
        ),
    }
    (OUT / "two_detection_benchmarks_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
