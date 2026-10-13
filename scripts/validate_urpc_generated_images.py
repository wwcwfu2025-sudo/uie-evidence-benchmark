from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "URPC2020" / "URPC2020" / "test" / "images"
GENERATED = ROOT / "benchmark_runs" / "urpc2020_yolov12n_detection" / "images"
METHODS = ["C000", "C020", "D001", "D002", "D007", "D044", "D046", "D048"]
REPORT = ROOT / "analysis" / "urpc2020_generated_images_validation.json"


def read(path: Path) -> np.ndarray | None:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        return None


def main() -> None:
    sources = sorted(SOURCE.glob("*.jpg"))
    payload: dict[str, object] = {"source_images": len(sources), "methods": {}}
    failed = False
    source_shapes: dict[str, tuple[int, ...] | None] = {}
    for source_path in sources:
        source = read(source_path)
        source_shapes[source_path.stem] = None if source is None else source.shape
    for method in METHODS:
        output_dir = GENERATED / method
        missing: list[str] = []
        decode_failures: list[str] = []
        shape_mismatches: list[dict[str, object]] = []
        for source_path in sources:
            output_path = output_dir / f"{source_path.stem}.png"
            if not output_path.exists():
                missing.append(output_path.name)
                continue
            output = read(output_path)
            source_shape = source_shapes[source_path.stem]
            if source_shape is None or output is None:
                decode_failures.append(source_path.stem)
            elif source_shape != output.shape:
                shape_mismatches.append(
                    {
                        "stem": source_path.stem,
                        "source_shape": list(source_shape),
                        "output_shape": list(output.shape),
                    }
                )
        extras = sorted(
            path.name
            for path in output_dir.glob("*.png")
            if not (SOURCE / f"{path.stem}.jpg").exists()
        )
        payload["methods"][method] = {
            "outputs": len(list(output_dir.glob("*.png"))),
            "missing": missing,
            "extras": extras,
            "decode_failures": decode_failures,
            "shape_mismatches": shape_mismatches,
        }
        if missing or extras or decode_failures or shape_mismatches:
            failed = True
    report = json.dumps(payload, ensure_ascii=False, indent=2)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report + "\n", encoding="utf-8")
    print(report)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
