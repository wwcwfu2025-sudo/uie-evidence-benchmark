from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


REVISION_ROOT = Path(__file__).resolve().parents[1]
REPO = REVISION_ROOT / "external_methods" / "W2WDiff-main"
PARTS = REPO / "weights" / "download_parts"
OUTPUTS = {
    "ema_model_130000.pth": "ema_model_130000.pth.part-",
    "w2wdiff_vae.ckpt": "w2wdiff_vae.ckpt.part-",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(destination: Path, prefix: str) -> dict[str, object]:
    parts = sorted(PARTS.glob(prefix + "*"))
    expected_names = [f"{prefix}{index:02d}" for index in range(5)]
    if [part.name for part in parts] != expected_names:
        raise RuntimeError(
            f"incomplete split checkpoint for {destination.name}: "
            f"found {[part.name for part in parts]}, expected {expected_names}"
        )
    if any(part.stat().st_size <= 0 for part in parts):
        raise RuntimeError(f"zero-byte split checkpoint part for {destination.name}")
    expected_size = sum(part.stat().st_size for part in parts)
    if destination.exists() and destination.stat().st_size == expected_size:
        return {
            "output": str(destination),
            "bytes": expected_size,
            "sha256": sha256(destination),
            "parts": [str(part) for part in parts],
            "status": "existing_size_verified",
        }
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
        target.flush()
        os.fsync(target.fileno())
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"merged size mismatch for {destination.name}: "
            f"{temporary.stat().st_size} != {expected_size}"
        )
    temporary.replace(destination)
    return {
        "output": str(destination),
        "bytes": expected_size,
        "sha256": sha256(destination),
        "parts": [
            {"path": str(part), "bytes": part.stat().st_size, "sha256": sha256(part)}
            for part in parts
        ],
        "status": "merged_and_size_verified",
    }


def main() -> None:
    records = []
    for filename, prefix in OUTPUTS.items():
        records.append(merge(REPO / "weights" / filename, prefix))
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Official W2WDiff Google Drive folder linked by the official repository",
        "records": records,
    }
    path = REPO / "weights" / "merge_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
