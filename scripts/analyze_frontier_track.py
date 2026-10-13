from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "frontier_stratified_manifest_public.csv"
CLASSICAL = ROOT / "raw_results" / "expanded_classical_per_image.csv"
DEEP = ROOT / "raw_results" / "72_method_library_deep28_8datasets_per_image.csv"
FRONTIER = ROOT / "raw_results"
# UIE_ANALYSIS_OUTPUT denotes the shared analysis root, not the track directory.
OUT = Path(os.environ.get("UIE_ANALYSIS_OUTPUT", ROOT / "analysis")).expanduser().resolve() / "frontier_track"

# RAW, one standardized traditional baseline, and six released deep models.
# F001/F002 are appended when their evaluations are complete.
COMPARATORS = ["C000", "C020", "D001", "D007", "D044", "D046", "D047", "D048"]
METRICS = ["psnr", "ssim", "delta_e2000", "uciqe", "uiqm"]
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "delta_e2000": False, "uciqe": True, "uiqm": True}


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


def number(value: str | object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, repeats: int = 2000) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    chunk = 250
    samples: list[np.ndarray] = []
    for start in range(0, repeats, chunk):
        count = min(chunk, repeats - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        samples.append(values[indices].mean(axis=1))
    boot = np.concatenate(samples)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    for required in (MANIFEST, CLASSICAL, DEEP, FRONTIER / "f001_per_image.csv", FRONTIER / "f002_per_image.csv"):
        if not required.is_file():
            raise FileNotFoundError(f"Required public-package input is missing: {required}")
    manifest = read_csv(MANIFEST)
    selected = {(row["dataset"], row["sample_key"]) for row in manifest}
    paired = {(row["dataset"], row["sample_key"]) for row in manifest if row["has_reference"] == "1"}

    rows: list[dict[str, str]] = []
    for path in (CLASSICAL, DEEP):
        for row in read_csv(path):
            if row.get("method_id") not in COMPARATORS:
                continue
            if (row.get("dataset", ""), row.get("sample_key", "")) not in selected:
                continue
            if row.get("status") != "ok":
                continue
            rows.append(row)

    frontier_files = sorted(FRONTIER.glob("f*_per_image.csv")) if FRONTIER.exists() else []
    for path in frontier_files:
        for row in read_csv(path):
            if row.get("status") == "ok":
                rows.append(row)

    by_method: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method_id"], []).append(row)

    expected = len(manifest)
    incomplete = {method: len(items) for method, items in by_method.items() if len(items) != expected}
    if incomplete:
        raise RuntimeError(f"Incomplete frontier rows: {incomplete}; expected {expected}")

    overall: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    for method_id, items in sorted(by_method.items()):
        name = items[0]["method"]
        base: dict[str, object] = {"method_id": method_id, "method": name, "n": len(items), "paired_n": sum((r["dataset"], r["sample_key"]) in paired for r in items)}
        for metric in METRICS:
            base[f"{metric}_mean"] = mean([number(row.get(metric, "")) for row in items])
        base["runtime_ms_mean"] = mean([number(row.get("runtime_ms", "")) for row in items])
        overall.append(base)

        for dataset in sorted({row["dataset"] for row in items}):
            subset = [row for row in items if row["dataset"] == dataset]
            record: dict[str, object] = {"method_id": method_id, "method": name, "dataset": dataset, "n": len(subset)}
            for metric in METRICS:
                record[f"{metric}_mean"] = mean([number(row.get(metric, "")) for row in subset])
            dataset_rows.append(record)

    raw = {(row["dataset"], row["sample_key"]): row for row in by_method["C000"]}
    bootstrap: list[dict[str, object]] = []
    for method_id, items in sorted(by_method.items()):
        if method_id == "C000":
            continue
        for metric in METRICS:
            deltas: list[float] = []
            for row in items:
                key = (row["dataset"], row["sample_key"])
                value = number(row.get(metric, ""))
                raw_value = number(raw[key].get(metric, ""))
                if np.isfinite(value) and np.isfinite(raw_value):
                    delta = value - raw_value
                    if not HIGHER_IS_BETTER[metric]:
                        delta = -delta
                    deltas.append(delta)
            estimate, low, high = bootstrap_mean_ci(
                np.asarray(deltas, dtype=np.float64),
                seed=20261012 + sum(ord(c) for c in f"{method_id}:{metric}"),
            )
            bootstrap.append(
                {
                    "method_id": method_id,
                    "method": items[0]["method"],
                    "metric": metric,
                    "n": len(deltas),
                    "favorable_delta_vs_raw": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "direction_note": "positive_is_favorable",
                }
            )

    ranks: list[dict[str, object]] = []
    datasets = sorted({row["dataset"] for row in rows})
    for metric in METRICS:
        eligible_datasets = ["LSUI", "UIEB"] if metric in {"psnr", "ssim", "delta_e2000"} else datasets
        method_ranks: dict[str, list[float]] = {method: [] for method in by_method}
        for dataset in eligible_datasets:
            candidates: list[tuple[str, float]] = []
            for method_id in by_method:
                match = next((row for row in dataset_rows if row["method_id"] == method_id and row["dataset"] == dataset), None)
                if match is None:
                    continue
                value = number(match.get(f"{metric}_mean", ""))
                if np.isfinite(value):
                    candidates.append((method_id, value))
            candidates.sort(key=lambda item: item[1], reverse=HIGHER_IS_BETTER[metric])
            for rank, (method_id, _) in enumerate(candidates, 1):
                method_ranks[method_id].append(float(rank))
        for method_id, values in method_ranks.items():
            if not values:
                continue
            ranks.append(
                {
                    "method_id": method_id,
                    "method": by_method[method_id][0]["method"],
                    "metric": metric,
                    "datasets": len(values),
                    "mean_rank": float(np.mean(values)),
                    "rank_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "best_rank": float(np.min(values)),
                    "worst_rank": float(np.max(values)),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "frontier_overall_summary.csv", overall)
    write_csv(OUT / "frontier_dataset_summary.csv", dataset_rows)
    write_csv(OUT / "frontier_bootstrap_vs_raw.csv", bootstrap)
    write_csv(OUT / "frontier_cross_dataset_ranks.csv", ranks)
    write_csv(OUT / "frontier_combined_per_image.csv", rows)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "selected_images": expected,
        "paired_images": len(paired),
        "datasets": datasets,
        "methods": {method_id: items[0]["method"] for method_id, items in sorted(by_method.items())},
        "bootstrap_repeats": 2000,
        "bootstrap_unit": "image; one observation per selected source image",
        "frontier_files": [path.relative_to(ROOT).as_posix() for path in frontier_files],
        "interpretation_limit": "Official checkpoints are evaluated as released-model transfer; this is not standardized retraining.",
    }
    (OUT / "frontier_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
