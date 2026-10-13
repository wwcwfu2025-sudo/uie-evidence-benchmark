from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.input_dir.glob("*.png"))
    source_names = {path.name for path in sources}
    output_names = {path.name for path in args.output_dir.glob("*.png")}
    decode_failures: list[str] = []
    shape_mismatches: list[dict[str, object]] = []
    for source_path in sources:
        output_path = args.output_dir / source_path.name
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if source is None or output is None:
            decode_failures.append(source_path.name)
        elif source.shape != output.shape:
            shape_mismatches.append(
                {
                    "file": source_path.name,
                    "input_shape": list(source.shape),
                    "output_shape": list(output.shape),
                }
            )
    payload = {
        "input_files": len(sources),
        "output_files": len(output_names),
        "missing_outputs": sorted(source_names - output_names),
        "unexpected_outputs": sorted(output_names - source_names),
        "decode_failures": decode_failures,
        "shape_mismatches": shape_mismatches,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["missing_outputs"] or payload["unexpected_outputs"] or decode_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
