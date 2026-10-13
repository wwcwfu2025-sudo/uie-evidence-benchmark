from __future__ import annotations

import os

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


WORKSPACE = Path(__file__).resolve().parents[2]
REVISION_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("UIE_DATA_ROOT", "datasets"))
OUT_DIR = REVISION_ROOT / "manifests"
OUT_CSV = OUT_DIR / "expanded_quality_manifest.csv"
OUT_META = OUT_DIR / "expanded_quality_manifest_metadata.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED = "TCSVT-31464-2026-expanded-v1"


def image_files(root: Path, suffixes: Optional[set[str]] = None) -> list[Path]:
    allowed = suffixes or IMAGE_SUFFIXES
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: path.as_posix().lower(),
    )


def stable_score(dataset: str, path: Path) -> str:
    relative = path.relative_to(DATA_ROOT).as_posix()
    return hashlib.sha256(f"{SEED}|{dataset}|{relative}".encode("utf-8")).hexdigest()


def select(paths: Iterable[Path], dataset: str, maximum: int) -> list[Path]:
    ordered = sorted(paths, key=lambda path: (stable_score(dataset, path), path.as_posix().lower()))
    if maximum <= 0 or len(ordered) <= maximum:
        return ordered
    return ordered[:maximum]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_id(dataset: str, path: Path) -> str:
    if dataset == "Submerged3D":
        return path.parent.parent.name
    if dataset == "UID2021":
        return path.parent.name
    return path.stem


def build() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    used_keys: set[tuple[str, str]] = set()

    def add(
        dataset: str,
        inputs: Iterable[Path],
        reference_dir: Optional[Path] = None,
        maximum: int = 0,
    ) -> None:
        for path in select(inputs, dataset, maximum):
            sample_key = path.stem
            if (dataset, sample_key) in used_keys:
                sample_key = f"{sample_key}__{stable_score(dataset, path)[:8]}"
            used_keys.add((dataset, sample_key))
            reference = ""
            if reference_dir is not None:
                candidate = reference_dir / path.name
                if candidate.exists():
                    reference = str(candidate)
            rows.append(
                {
                    "dataset": dataset,
                    "sample_key": sample_key,
                    "cluster_id": cluster_id(dataset, path),
                    "input": str(path),
                    "reference": reference,
                    "has_reference": "1" if reference else "0",
                    "selection_score": stable_score(dataset, path),
                    "input_sha256": sha256(path),
                }
            )

    add(
        "UIEB",
        image_files(DATA_ROOT / "UIEB" / "extracted" / "raw-890"),
        DATA_ROOT / "UIEB" / "extracted" / "reference-890",
        maximum=400,
    )
    add(
        "LSUI",
        image_files(DATA_ROOT / "LSUI" / "extracted" / "LSUI" / "input"),
        DATA_ROOT / "LSUI" / "extracted" / "LSUI" / "GT",
        maximum=400,
    )
    add(
        "U45",
        image_files(
            DATA_ROOT
            / "U45"
            / "extracted"
            / "underwater-test-dataset-U45--master"
            / "upload"
            / "U45"
            / "U45"
        ),
    )
    sub3d = DATA_ROOT / "Submerged3D" / "extracted" / "Submerged3D"
    add("Submerged3D", [path for path in image_files(sub3d) if path.parent.name == "input"])
    add(
        "TankImage-I",
        image_files(DATA_ROOT / "TankImage-I" / "extracted" / "TankImage-I-main", {".bmp"}),
    )
    cc_root = (
        DATA_ROOT
        / "ColorCheck7-OceanDark"
        / "extracted"
        / "two-No-reference-image-dataset-main"
    )
    add("ColorCheck7", image_files(cc_root / "Color-Check7"))
    add("OceanDark", image_files(cc_root / "OceanDark"))
    uid_root = DATA_ROOT / "UID2021" / "extracted"
    add("UID2021", [path for path in image_files(uid_root) if path.name.startswith("SI_")])
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build()
    keys = [(row["dataset"], row["sample_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate dataset/sample_key values in expanded manifest")
    fields = list(rows[0])
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["dataset"] for row in rows)
    reference_counts = Counter(row["dataset"] for row in rows if row["has_reference"] == "1")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "selection": (
            "All images are retained for datasets with <=400 eligible inputs; UIEB and LSUI use "
            "the 400 lowest SHA-256 scores of dataset name plus relative path under a fixed seed. "
            "UID2021 uses only the 60 source images prefixed SI_ and excludes algorithm-generated versions."
        ),
        "dataset_counts": dict(sorted(counts.items())),
        "reference_counts": dict(sorted(reference_counts.items())),
        "total_images": len(rows),
        "total_reference_images": sum(reference_counts.values()),
        "source_data_root": str(DATA_ROOT),
        "manifest": str(OUT_CSV),
    }
    OUT_META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_CSV)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

