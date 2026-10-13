from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
CLASSICAL = ROOT / "raw_results" / "expanded_classical_per_image.csv"
DEEP = ROOT / "raw_results" / "deep" / "72_method_library_deep28_8datasets_per_image.csv"
OVERALL = ANALYSIS / "expanded_overall_summary.csv"
BOOTSTRAP = ANALYSIS / "expanded_paired_bootstrap_vs_raw.csv"
OUT = ANALYSIS / "manuscript_tables"

DEEP_IDS = [
    "D001", "D002", "D003", "D005", "D006", "D007", "D009", "D010", "D011", "D024",
    "D031", "D035", "D036", "D044", "D045", "D046", "D047", "D048", "D049", "D050",
]
M_ID = {"C000": "M00", **{f"C{index:03d}": f"M{index:02d}" for index in range(1, 21)}}
M_ID.update({method_id: f"M{index:02d}" for index, method_id in enumerate(DEEP_IDS, 21)})
REPRESENTATIVES = ["C020", "D001", "D002", "D005", "D007", "D024", "D031", "D044", "D045", "D046", "D047", "D048"]
DISPLAY_NAMES = {
    "C003": "Shades-of-Gray (p=6)",
    "C004": "Gray-Edge (p=6)",
    "C005": "White-Patch (p99)",
    "C006": "RGB-HE",
    "C007": "YCrCb-HE",
    "C008": "LAB-CLAHE",
    "C009": "HSV-CLAHE",
    "C010": "Gamma-0.7",
    "C020": "Fusion-Ancuti-inspired",
}


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


def value(item: dict[str, str], key: str) -> float:
    try:
        return float(item[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def main() -> None:
    overall = {row["method_id"]: row for row in read_csv(OVERALL)}
    per_image = read_csv(CLASSICAL) + read_csv(DEEP)
    by_method: dict[str, list[dict[str, str]]] = {}
    for row in per_image:
        if row.get("status") == "ok":
            by_method.setdefault(row["method_id"], []).append(row)

    table4: list[dict[str, object]] = []
    for method_id in sorted(M_ID, key=lambda key: int(M_ID[key][1:])):
        row = overall[method_id]
        runtimes = np.asarray([value(item, "runtime_ms") for item in by_method[method_id]], dtype=np.float64)
        runtimes = runtimes[np.isfinite(runtimes)]
        table4.append(
            {
                "method_id": method_id,
                "display_id": M_ID[method_id],
                "method": DISPLAY_NAMES.get(method_id, row["method"]),
                "paired_n": 800,
                "all_n": 1253,
                "psnr": value(row, "psnr_dataset_balanced_mean"),
                "ssim": value(row, "ssim_dataset_balanced_mean"),
                "mae": float(np.nanmean([value(item, "mae") for item in by_method[method_id]])),
                "delta_e2000": value(row, "delta_e2000_dataset_balanced_mean"),
                "uciqe_dataset_balanced": value(row, "uciqe_dataset_balanced_mean"),
                "uiqm_dataset_balanced": value(row, "uiqm_dataset_balanced_mean"),
                "runtime_ms_pooled": float(runtimes.mean()),
            }
        )

    bootstrap = read_csv(BOOTSTRAP)
    boot_lookup = {(row["method_id"], row["metric"]): row for row in bootstrap}
    datasets = sorted({row["dataset"] for row in per_image})
    method_ids = sorted(M_ID, key=lambda key: int(M_ID[key][1:]))
    borda_by_method: dict[str, list[float]] = {method_id: [] for method_id in method_ids}
    dataset_borda_vectors: list[np.ndarray] = []
    for dataset in datasets:
        uciqe = np.asarray(
            [np.mean([value(row, "uciqe") for row in by_method[method_id] if row["dataset"] == dataset]) for method_id in method_ids],
            dtype=np.float64,
        )
        uiqm = np.asarray(
            [np.mean([value(row, "uiqm") for row in by_method[method_id] if row["dataset"] == dataset]) for method_id in method_ids],
            dtype=np.float64,
        )
        borda = (rank_desc(uciqe) + rank_desc(uiqm)) / 2.0
        dataset_borda_vectors.append(borda)
        for method_id, rank in zip(method_ids, borda):
            borda_by_method[method_id].append(float(rank))

    table6: list[dict[str, object]] = []
    for method_id in REPRESENTATIVES:
        psnr = boot_lookup[(method_id, "psnr")]
        ssim = boot_lookup[(method_id, "ssim")]
        ranks = borda_by_method[method_id]
        table6.append(
            {
                "display_id": M_ID[method_id],
                "method_id": method_id,
                "method": DISPLAY_NAMES.get(method_id, overall[method_id]["method"]),
                "delta_psnr": value(psnr, "raw_scale_delta"),
                "delta_psnr_ci_low": value(psnr, "raw_scale_delta_ci95_low"),
                "delta_psnr_ci_high": value(psnr, "raw_scale_delta_ci95_high"),
                "delta_ssim": value(ssim, "raw_scale_delta"),
                "delta_ssim_ci_low": value(ssim, "raw_scale_delta_ci95_low"),
                "delta_ssim_ci_high": value(ssim, "raw_scale_delta_ci95_high"),
                "borda_mean": float(np.mean(ranks)),
                "borda_best": float(np.min(ranks)),
                "borda_worst": float(np.max(ranks)),
            }
        )

    rank_matrix = np.vstack(dataset_borda_vectors)
    # Kendall's W over datasets (raters) and methods (objects), no ties correction;
    # ties are rare and averaged ranks are retained in the released table.
    rank_sums = rank_matrix.sum(axis=0)
    m, n = rank_matrix.shape
    kendall_w = float(12.0 * np.sum((rank_sums - rank_sums.mean()) ** 2) / (m * m * (n**3 - n)))
    correlations: list[float] = []
    for left in range(m):
        for right in range(left + 1, m):
            correlations.append(float(np.corrcoef(rank_matrix[left], rank_matrix[right])[0, 1]))

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "table4_expanded_41_items.csv", table4)
    write_csv(OUT / "table6_expanded_stability.csv", table6)
    metadata = {
        "table4": {
            "items_including_raw": len(table4),
            "enhancement_algorithms": len(table4) - 1,
            "paired_images": 800,
            "all_images": 1253,
            "datasets": 8,
            "aggregation": "Equal-weight mean of dataset means; runtime is pooled per-image mean.",
        },
        "table6": {
            "representative_methods": len(table6),
            "bootstrap_repeats": 2000,
            "bootstrap_seed": 20261012,
            "borda_definition": "Within each dataset, average descending UCIQE and UIQM ranks over all 41 items; then summarize across eight datasets.",
            "kendall_w_no_tie_correction": kendall_w,
            "pairwise_dataset_spearman_like_pearson_on_rank_vectors_median": float(np.median(correlations)),
            "pairwise_dataset_rank_correlation_range": [float(np.min(correlations)), float(np.max(correlations))],
        },
    }
    (OUT / "manuscript_table_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
