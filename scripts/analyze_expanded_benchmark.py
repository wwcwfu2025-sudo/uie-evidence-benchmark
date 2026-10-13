from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr


REVISION_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REVISION_ROOT / "manifests" / "expanded_quality_manifest_public.csv"
CLASSICAL_FINAL = REVISION_ROOT / "raw_results" / "expanded_classical_per_image.csv"
CLASSICAL_STATE = REVISION_ROOT / "raw_results" / "classical_state"
DEEP_FINAL = REVISION_ROOT / "raw_results" / "72_method_library_deep28_8datasets_per_image.csv"
DEEP_STATE = REVISION_ROOT / "benchmark_runs" / "expanded_deep" / "state"
# Override the analysis root to verify the archive without overwriting it.
OUT = Path(os.environ.get("UIE_ANALYSIS_OUTPUT", REVISION_ROOT / "analysis")).expanduser().resolve()

METRICS = {
    "psnr": {"higher": True, "scope": "paired"},
    "ssim": {"higher": True, "scope": "paired"},
    "delta_e2000": {"higher": False, "scope": "paired"},
    "uciqe": {"higher": True, "scope": "all"},
    "uiqm": {"higher": True, "scope": "all"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_state_or_final(final_path: Path, state_dir: Path) -> list[dict[str, str]]:
    if final_path.exists():
        return read_csv(final_path)
    rows: list[dict[str, str]] = []
    for path in sorted(state_dir.glob("*.csv")):
        rows.extend(read_csv(path))
    return rows


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


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def attach_clusters(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    manifest = read_csv(MANIFEST)
    cluster = {(row["dataset"], row["sample_key"]): row["cluster_id"] for row in manifest}
    for row in rows:
        row["cluster_id"] = cluster.get(
            (row.get("dataset", ""), row.get("sample_key", "")),
            row.get("cluster_id", row.get("sample_key", "")),
        )
    return rows


def successful_rows() -> list[dict[str, str]]:
    classical = read_state_or_final(CLASSICAL_FINAL, CLASSICAL_STATE)
    deep = read_state_or_final(DEEP_FINAL, DEEP_STATE)
    rows = attach_clusters(classical + deep)
    return [row for row in rows if row.get("status") == "ok"]


def grouped_values(
    rows: Iterable[dict[str, str]], metric: str
) -> dict[tuple[str, str], list[float]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(metric))
        if value is not None:
            groups[(row["method_id"], row["dataset"])].append(value)
    return groups


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    names = {row["method_id"]: row["method"] for row in rows}
    method_ids = sorted(names)
    for method_id in method_ids:
        selected = [row for row in rows if row["method_id"] == method_id]
        item: dict[str, object] = {
            "method_id": method_id,
            "method": names[method_id],
            "n_images": len(selected),
            "n_datasets": len({row["dataset"] for row in selected}),
        }
        for metric in METRICS:
            per_dataset: list[float] = []
            all_values: list[float] = []
            for dataset in sorted({row["dataset"] for row in selected}):
                values = [
                    value
                    for row in selected
                    if row["dataset"] == dataset
                    for value in [as_float(row.get(metric))]
                    if value is not None
                ]
                if values:
                    per_dataset.append(float(np.mean(values)))
                    all_values.extend(values)
            item[f"{metric}_dataset_balanced_mean"] = (
                float(np.mean(per_dataset)) if per_dataset else ""
            )
            item[f"{metric}_pooled_mean"] = float(np.mean(all_values)) if all_values else ""
        result.append(item)
    return result


def paired_index(
    rows: list[dict[str, str]], method_id: str, metric: str
) -> dict[tuple[str, str], tuple[str, float]]:
    result: dict[tuple[str, str], tuple[str, float]] = {}
    for row in rows:
        if row["method_id"] != method_id:
            continue
        value = as_float(row.get(metric))
        if value is not None:
            result[(row["dataset"], row["sample_key"])] = (row["cluster_id"], value)
    return result


def stratified_cluster_bootstrap(
    pairs: list[tuple[str, str, str, float, float]],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float]:
    # tuple fields: dataset, sample_key, cluster_id, method_value, raw_value
    by_dataset: dict[str, list[tuple[str, str, str, float, float]]] = defaultdict(list)
    for pair in pairs:
        by_dataset[pair[0]].append(pair)

    observed_dataset_deltas = []
    observed_method = []
    observed_raw = []
    for values in by_dataset.values():
        observed_method.append(float(np.mean([item[3] for item in values])))
        observed_raw.append(float(np.mean([item[4] for item in values])))
        observed_dataset_deltas.append(float(np.mean([item[3] - item[4] for item in values])))
    observed_delta = float(np.mean(observed_dataset_deltas))

    # Vectorized cluster bootstrap.  For a sampled cluster, its image-level
    # differences remain together; cluster sums and sizes preserve the exact
    # weighting of the earlier explicit expansion without Python-level loops
    # over bootstrap replicates.
    bootstrap_by_dataset: list[np.ndarray] = []
    for values in by_dataset.values():
        by_cluster: dict[str, list[tuple[str, str, str, float, float]]] = defaultdict(list)
        for item in values:
            by_cluster[item[2]].append(item)
        cluster_ids = sorted(by_cluster)
        sums = np.asarray(
            [sum(item[3] - item[4] for item in by_cluster[cluster_id]) for cluster_id in cluster_ids],
            dtype=np.float64,
        )
        counts = np.asarray(
            [len(by_cluster[cluster_id]) for cluster_id in cluster_ids], dtype=np.float64
        )
        indices = rng.integers(0, len(cluster_ids), size=(iterations, len(cluster_ids)))
        sampled_sums = np.sum(sums[indices], axis=1)
        sampled_counts = np.sum(counts[indices], axis=1)
        bootstrap_by_dataset.append(sampled_sums / sampled_counts)
    bootstrap = np.mean(np.stack(bootstrap_by_dataset, axis=1), axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return (
        float(np.mean(observed_method)),
        float(np.mean(observed_raw)),
        observed_delta,
        float(low),
        float(high),
    )


def bootstrap_table(
    rows: list[dict[str, str]], iterations: int = 2000, seed: int = 20261012
) -> list[dict[str, object]]:
    names = {row["method_id"]: row["method"] for row in rows}
    method_ids = [method_id for method_id in sorted(names) if method_id != "C000"]
    result: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for metric, properties in METRICS.items():
        raw = paired_index(rows, "C000", metric)
        for method_id in method_ids:
            method = paired_index(rows, method_id, metric)
            keys = sorted(set(raw) & set(method))
            pairs = [
                (key[0], key[1], method[key][0], method[key][1], raw[key][1])
                for key in keys
            ]
            if not pairs:
                continue
            method_mean, raw_mean, delta, low, high = stratified_cluster_bootstrap(
                pairs, iterations, rng
            )
            favorable = delta if properties["higher"] else -delta
            favorable_low = low if properties["higher"] else -high
            favorable_high = high if properties["higher"] else -low
            result.append(
                {
                    "metric": metric,
                    "scope": properties["scope"],
                    "method_id": method_id,
                    "method": names[method_id],
                    "n_paired": len(pairs),
                    "n_datasets": len({item[0] for item in pairs}),
                    "method_dataset_balanced_mean": method_mean,
                    "raw_dataset_balanced_mean": raw_mean,
                    "raw_scale_delta": delta,
                    "raw_scale_delta_ci95_low": low,
                    "raw_scale_delta_ci95_high": high,
                    "favorable_delta": favorable,
                    "favorable_delta_ci95_low": favorable_low,
                    "favorable_delta_ci95_high": favorable_high,
                    "statistically_resolved_favorable": favorable_low > 0,
                    "statistically_resolved_unfavorable": favorable_high < 0,
                    "bootstrap_iterations": iterations,
                    "bootstrap_seed": seed,
                }
            )
    return result


def rank_stability(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    names = {row["method_id"]: row["method"] for row in rows}
    result: list[dict[str, object]] = []
    for metric, properties in METRICS.items():
        groups = grouped_values(rows, metric)
        ranks_by_method: dict[str, list[float]] = defaultdict(list)
        for dataset in sorted({dataset for _, dataset in groups}):
            methods = sorted(method for method, source in groups if source == dataset)
            if len(methods) < 2:
                continue
            means = np.asarray([np.mean(groups[(method, dataset)]) for method in methods])
            ranks = rankdata(-means if properties["higher"] else means, method="average")
            for method, rank in zip(methods, ranks):
                ranks_by_method[method].append(float(rank))
        for method_id, ranks in sorted(ranks_by_method.items()):
            result.append(
                {
                    "metric": metric,
                    "scope": properties["scope"],
                    "method_id": method_id,
                    "method": names[method_id],
                    "n_ranked_datasets": len(ranks),
                    "mean_rank": float(np.mean(ranks)),
                    "rank_std": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
                    "best_rank": float(np.min(ranks)),
                    "worst_rank": float(np.max(ranks)),
                    "top5_frequency": float(np.mean(np.asarray(ranks) <= 5.0)),
                }
            )
    return result


def metric_alignment(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            value = as_float(row.get(metric))
            if value is not None:
                groups[(row["method_id"], row["dataset"])][metric].append(value)
    records = []
    for key, values in groups.items():
        record: dict[str, object] = {"method_id": key[0], "dataset": key[1]}
        for metric, metric_values in values.items():
            record[metric] = float(np.mean(metric_values))
        records.append(record)

    result: list[dict[str, object]] = []
    for full_reference in ("psnr", "ssim", "delta_e2000"):
        for no_reference in ("uciqe", "uiqm"):
            paired = [
                (float(row[full_reference]), float(row[no_reference]))
                for row in records
                if full_reference in row and no_reference in row
            ]
            if len(paired) < 3:
                continue
            coefficient, p_value = spearmanr(
                [item[0] for item in paired], [item[1] for item in paired]
            )
            result.append(
                {
                    "level": "method_by_dataset_mean",
                    "full_reference_metric": full_reference,
                    "no_reference_metric": no_reference,
                    "n_method_dataset_pairs": len(paired),
                    "spearman_rho": float(coefficient),
                    "p_value": float(p_value),
                }
            )
    return result


def main() -> None:
    for required in (MANIFEST, CLASSICAL_FINAL, DEEP_FINAL):
        if not required.is_file():
            raise FileNotFoundError(f"Required public-package input is missing: {required}")
    rows = successful_rows()
    methods = sorted({row["method_id"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    expected = len(methods) * len(read_csv(MANIFEST))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "expanded_overall_summary.csv", aggregate(rows))
    write_csv(OUT / "expanded_paired_bootstrap_vs_raw.csv", bootstrap_table(rows))
    write_csv(OUT / "expanded_cross_dataset_rank_stability.csv", rank_stability(rows))
    write_csv(OUT / "expanded_metric_alignment.csv", metric_alignment(rows))
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "successful_rows": len(rows),
        "expected_rows_for_available_methods": expected,
        "available_method_ids": methods,
        "method_count_including_raw": len(methods),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "manifest_images": len(read_csv(MANIFEST)),
        "paired_reference_images": sum(1 for row in read_csv(MANIFEST) if row["has_reference"] == "1"),
        "bootstrap": (
            "Paired against RAW; cluster resampling within each dataset; dataset means are then "
            "equally weighted. Confidence intervals are percentile 95% intervals."
        ),
        "rank_stability": "Ranks are computed independently in each eligible dataset.",
        "warning": (
            "UIQM/UCIQE are reported as diagnostic measures rather than perceptual ground truth; "
            "their alignment with full-reference metrics is tested separately."
        ),
    }
    (OUT / "expanded_analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
