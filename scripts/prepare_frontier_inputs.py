from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


REVISION_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REVISION_ROOT / "manifests" / "frontier_stratified_manifest.csv"
OUTPUT = REVISION_ROOT / "manifests" / "frontier_staged_manifest.csv"
STAGING = REVISION_ROOT / "benchmark_runs" / "frontier_track" / "staged"
LONG_SIDE = 512


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def resize_long_side(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, LONG_SIDE / float(max(height, width)))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"cannot encode {path}")
    encoded.tofile(str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with SOURCE.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        token = f"{index:04d}_{row['dataset'].replace('-', '_')}"
        input_path = STAGING / "input" / f"{token}.png"
        reference_path = STAGING / "reference" / f"{token}.png" if row["reference"] else None
        if not input_path.exists():
            save_png(input_path, resize_long_side(read_bgr(Path(row["input"]))))
        if reference_path is not None and not reference_path.exists():
            reference = resize_long_side(read_bgr(Path(row["reference"])))
            input_image = read_bgr(input_path)
            if reference.shape[:2] != input_image.shape[:2]:
                reference = cv2.resize(
                    reference,
                    (input_image.shape[1], input_image.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            save_png(reference_path, reference)
        staged = dict(row)
        staged.update(
            {
                "frontier_index": index,
                "staged_token": token,
                "staged_input": str(input_path),
                "staged_reference": str(reference_path) if reference_path is not None else "",
                "staged_input_sha256": sha256(input_path),
            }
        )
        output_rows.append(staged)
        if index % 25 == 0 or index == len(rows):
            print(f"staged {index}/{len(rows)}", flush=True)
    fields = list(output_rows[0].keys())
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(SOURCE),
        "staged_manifest": str(OUTPUT),
        "image_count": len(output_rows),
        "long_side_cap": LONG_SIDE,
        "format": "PNG",
        "reason": (
            "Common deterministic input staging for frontier models with isolated environments; "
            "references are geometry-aligned but never supplied to inference."
        ),
    }
    (OUTPUT.parent / "frontier_staged_manifest_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
