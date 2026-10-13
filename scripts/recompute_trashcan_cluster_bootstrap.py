"""Recompute the archived 5,000-resample TrashCan video-cluster intervals.

This is separate from the original 2,000-resample image-level analysis. The
estimand is the image-weighted mean paired per-image F1 change versus RAW;
videos, not individual frames, are sampled with replacement. The original
method/stratum order, random seed, and sampling algorithm are retained.

Set UIE_ANALYSIS_OUTPUT to an external directory to leave release files intact.
The output is <output>/detection/trashcan_video_cluster_bootstrap.csv.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "raw_results" / "trashcan_quality_detection_per_image.csv"
OUT = Path(os.environ.get("UIE_ANALYSIS_OUTPUT", str(ROOT / "analysis"))).expanduser().resolve()
OUTPUT = OUT / "detection" / "trashcan_video_cluster_bootstrap.csv"
METHOD_IDS = ["C020", "D001", "D002", "D007", "D044", "D046", "D048"]
SEED = 20260722
REPS = 5000


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def cluster_draws(records: list[dict[str, object]], rng: np.random.Generator) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[str(row["video_id"])].append(float(row["delta_f1"]))
    video_ids = sorted(grouped)
    if not video_ids:
        raise ValueError("Cannot bootstrap an empty video stratum")
    sums = np.asarray([sum(grouped[item]) for item in video_ids], dtype=np.float64)
    counts = np.asarray([len(grouped[item]) for item in video_ids], dtype=np.float64)
    draws = np.empty(REPS, dtype=np.float64)
    for index in range(REPS):
        chosen = rng.integers(0, len(video_ids), len(video_ids))
        draws[index] = sums[chosen].sum() / counts[chosen].sum()
    return draws


def main() -> None:
    rows = read_rows()
    raw_by_name: dict[str, dict[str, object]] = {}
    method_by_name: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    seen: set[tuple[str, str]] = set()
    for source in rows:
        key = (source["method_id"], source["file_name"])
        if key in seen:
            raise ValueError(f"Duplicate method/image record: {key}")
        seen.add(key)
        match = re.search(r"(vid_\d+)", source["file_name"])
        if not match:
            raise RuntimeError(f"Cannot extract video id from {source['file_name']}")
        row = {
            "file_name": source["file_name"],
            "video_id": match.group(1),
            "method": source["method"],
            "per_image_f1": float(source["per_image_f1"]),
            "mean_luma": float(source["mean_luma"]),
        }
        if not np.all(np.isfinite([row["per_image_f1"], row["mean_luma"]])):
            raise ValueError(f"Non-finite F1 or luminance: {key}")
        if source["method_id"] == "C000":
            raw_by_name[source["file_name"]] = row
        else:
            method_by_name[source["method_id"]][source["file_name"]] = row

    if not raw_by_name:
        raise ValueError("No RAW image records were found")
    for method_id in METHOD_IDS:
        if set(method_by_name[method_id]) != set(raw_by_name):
            raise ValueError(f"Image pairing differs from RAW for {method_id}")

    raw_luma = np.asarray([float(item["mean_luma"]) for item in raw_by_name.values()], dtype=np.float64)
    q1, q2 = np.quantile(raw_luma, [1.0 / 3.0, 2.0 / 3.0])
    rng = np.random.default_rng(SEED)
    output_rows: list[dict[str, object]] = []
    for method_id in METHOD_IDS:
        records: list[dict[str, object]] = []
        for file_name, raw in raw_by_name.items():
            method = method_by_name[method_id][file_name]
            luma = float(raw["mean_luma"])
            stratum = "dark" if luma <= q1 else "middle" if luma <= q2 else "bright"
            records.append(
                {
                    "video_id": raw["video_id"],
                    "stratum": stratum,
                    "method": method["method"],
                    "delta_f1": float(method["per_image_f1"]) - float(raw["per_image_f1"]),
                }
            )
        for stratum in ("all", "dark", "middle", "bright"):
            work = records if stratum == "all" else [item for item in records if item["stratum"] == stratum]
            draws = cluster_draws(work, rng)
            deltas = np.asarray([float(item["delta_f1"]) for item in work], dtype=np.float64)
            output_rows.append(
                {
                    "method_id": method_id,
                    "method": work[0]["method"],
                    "stratum": stratum,
                    "n_frames": len(work),
                    "n_video_clusters": len({str(item["video_id"]) for item in work}),
                    "delta_f1_mean": float(deltas.mean()),
                    "delta_f1_cluster_bootstrap_mean": float(draws.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "improved_fraction": float(np.mean(deltas > 0)),
                    "bootstrap_reps": REPS,
                    "seed": SEED,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
