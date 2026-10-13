from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REVISION_ROOT / "manifests" / "expanded_quality_manifest.csv"
OUTPUT = REVISION_ROOT / "manifests" / "frontier_stratified_manifest.csv"
METADATA = REVISION_ROOT / "manifests" / "frontier_stratified_manifest_metadata.json"
SEED = "TCSVT-31464-2026-frontier-v1"
MAXIMUM_PER_DATASET = 40


def score(row: dict[str, str]) -> str:
    payload = "|".join((SEED, row["dataset"], row["sample_key"], row["input_sha256"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    with SOURCE.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    selected: list[dict[str, str]] = []
    for dataset in sorted(groups):
        candidates = sorted(groups[dataset], key=score)
        for row in candidates[:MAXIMUM_PER_DATASET]:
            item = dict(row)
            item["frontier_selection_score"] = score(row)
            selected.append(item)
    fields = list(selected[0].keys())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    dataset_counts = Counter(row["dataset"] for row in selected)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(SOURCE),
        "output_manifest": str(OUTPUT),
        "selection": (
            "Within each data source, sort by SHA-256(seed|dataset|sample_key|input_sha256) "
            "and retain at most 40 images."
        ),
        "seed": SEED,
        "maximum_per_dataset": MAXIMUM_PER_DATASET,
        "image_count": len(selected),
        "paired_reference_count": sum(bool(row["reference"]) for row in selected),
        "dataset_counts": dict(sorted(dataset_counts.items())),
    }
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
