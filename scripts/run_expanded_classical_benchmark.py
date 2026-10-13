from __future__ import annotations

import csv
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
REVISION_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts import benchmark_20_methods_8_datasets as base


MANIFEST = REVISION_ROOT / "manifests" / "expanded_quality_manifest.csv"
STATE_DIR = REVISION_ROOT / "raw_results" / "classical_state"
OUT_PER_IMAGE = REVISION_ROOT / "raw_results" / "expanded_classical_per_image.csv"
OUT_SUMMARY = REVISION_ROOT / "raw_results" / "expanded_classical_summary.csv"
OUT_METADATA = REVISION_ROOT / "raw_results" / "expanded_classical_metadata.json"


METHODS: list[tuple[str, str, Callable[[np.ndarray], np.ndarray], str]] = [
    ("C000", "RAW", lambda image: image.copy(), "control"),
    ("C001", "GrayWorld", base.gray_world, "standardized_reimplementation"),
    ("C002", "MaxRGB", base.max_rgb, "standardized_reimplementation"),
    ("C003", "ShadesOfGray_p6", base.shades_of_gray, "standardized_reimplementation"),
    ("C004", "GrayEdge_p6", base.gray_edge, "standardized_reimplementation"),
    ("C005", "WhitePatch_p99", base.white_patch_percentile, "standardized_reimplementation"),
    ("C006", "RGB_HE", base.rgb_hist_equalization, "standardized_reimplementation"),
    ("C007", "YCrCb_HE", base.ycrcb_hist_equalization, "standardized_reimplementation"),
    ("C008", "LAB_CLAHE", base.lab_clahe, "standardized_reimplementation"),
    ("C009", "HSV_CLAHE", base.hsv_clahe, "standardized_reimplementation"),
    ("C010", "Gamma_0p7", base.fixed_gamma, "standardized_reimplementation"),
    ("C011", "AdaptiveGamma", base.adaptive_gamma, "standardized_reimplementation"),
    ("C012", "UnsharpMask", base.unsharp_mask, "standardized_reimplementation"),
    ("C013", "Homomorphic", base.homomorphic_filter, "standardized_reimplementation"),
    ("C014", "SSR", base.ssr, "standardized_reimplementation"),
    ("C015", "MSR", base.msr, "standardized_reimplementation"),
    ("C016", "MSRCR", base.msrcr, "standardized_reimplementation"),
    ("C017", "DCP", base.dcp, "standardized_reimplementation"),
    ("C018", "UDCP", base.udcp, "standardized_reimplementation"),
    ("C019", "RCP", base.rcp, "standardized_reimplementation"),
    (
        "C020",
        "Fusion_Ancuti_inspired",
        base.fusion_ancuti_inspired,
        "standardized_reimplementation_not_original_code",
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_method(
    method_id: str,
    name: str,
    function: Callable[[np.ndarray], np.ndarray],
    implementation_class: str,
    manifest: list[dict[str, str]],
    long_side: int,
) -> list[dict[str, object]]:
    state_path = STATE_DIR / f"{method_id}.csv"
    prior = read_csv(state_path)
    completed = {(row["dataset"], row["sample_key"]) for row in prior}
    rows: list[dict[str, object]] = list(prior)
    for index, item in enumerate(manifest, 1):
        key = (item["dataset"], item["sample_key"])
        if key in completed:
            continue
        image = base.resize_long_side(base.read_bgr(Path(item["input"])), long_side)
        reference = (
            base.resize_long_side(base.read_bgr(Path(item["reference"])), long_side)
            if item["reference"]
            else None
        )
        if reference is not None and reference.shape[:2] != image.shape[:2]:
            reference = cv2.resize(
                reference,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        row: dict[str, object] = {
            "method_id": method_id,
            "method": name,
            "implementation_class": implementation_class,
            "device": "CPU",
            "dataset": item["dataset"],
            "sample_key": item["sample_key"],
            "cluster_id": item["cluster_id"],
            "input": item["input"],
            "reference": item["reference"],
            "width": image.shape[1],
            "height": image.shape[0],
            "status": "ok",
            "error": "",
        }
        try:
            start = time.perf_counter()
            output = base.clip_u8(function(image))
            row["runtime_ms"] = (time.perf_counter() - start) * 1000.0
            if output.shape[:2] != image.shape[:2]:
                output = cv2.resize(
                    output,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            row.update(
                base.reference_metrics(output, reference)
                if reference is not None
                else {"psnr": "", "ssim": "", "mae": "", "delta_e2000": ""}
            )
            row.update(base.no_reference_metrics(output))
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_ms": "",
                    "psnr": "",
                    "ssim": "",
                    "mae": "",
                    "delta_e2000": "",
                    "uciqe": "",
                    "uiqm": "",
                    "entropy": "",
                    "mean_luma": "",
                    "laplacian_variance": "",
                    "channel_mean_range": "",
                    "dark_fraction": "",
                    "bright_fraction": "",
                }
            )
        rows.append(row)
        if index % 10 == 0 or index == len(manifest):
            write_csv(state_path, rows)
            print(f"{method_id} {index}/{len(manifest)} {row['status']}", flush=True)
    return rows


def main(long_side: int = 512) -> None:
    manifest = read_csv(MANIFEST)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    all_rows: list[dict[str, object]] = []
    for method_id, name, function, implementation_class in METHODS:
        all_rows.extend(
            run_method(method_id, name, function, implementation_class, manifest, long_side)
        )
    write_csv(OUT_PER_IMAGE, all_rows)
    summary_source = [
        {key: value for key, value in row.items() if key != "method_id"} for row in all_rows
    ]
    summary = base.aggregate(summary_source)
    for row in summary:
        row["method_id"] = next(
            method_id for method_id, name, _, _ in METHODS if name == row["method"]
        )
    write_csv(OUT_SUMMARY, summary)
    failures = [row for row in all_rows if row.get("status") != "ok"]
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(MANIFEST),
        "method_count_including_raw": len(METHODS),
        "image_count": len(manifest),
        "expected_rows": len(METHODS) * len(manifest),
        "actual_rows": len(all_rows),
        "failure_count": len(failures),
        "long_side": long_side,
        "timing_boundary": (
            "enhancement only; excludes file I/O, common resize, metric computation, and output writing"
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "wall_seconds": time.perf_counter() - start_all,
    }
    OUT_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_PER_IMAGE)
    print(OUT_SUMMARY)
    print(OUT_METADATA)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

