from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from .adapters import build_adapter
from .core import AdapterError, read_bgr, write_image
from .registry import registry_by_id, load_registry


def command_list(only_deep: bool, status: str) -> None:
    items = load_registry()
    if only_deep:
        items = [item for item in items if item.is_deep]
    if status:
        items = [item for item in items if item.status == status]
    for item in items:
        print("\t".join([item.method_id, item.name, item.family, item.status, item.authority]))
    print("TOTAL=%d; DEEP=%d" % (len(items), sum(item.is_deep for item in items)))


def command_run(method_id: str, input_path: Path, output_path: Path, device: str, metadata_path: Path) -> None:
    specs = registry_by_id()
    if method_id not in specs:
        raise AdapterError("Unknown method_id: %s" % method_id)
    spec = specs[method_id]
    adapter = build_adapter(spec, device)
    image = read_bgr(input_path)
    start_load = time.perf_counter()
    adapter.load()
    load_seconds = time.perf_counter() - start_load
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = adapter.enhance(image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    write_image(output_path, output)
    metadata = adapter.metadata()
    metadata.update(
        {
            "input": str(input_path),
            "output": str(output_path),
            "input_width": image.shape[1],
            "input_height": image.shape[0],
            "output_width": output.shape[1],
            "output_height": output.shape[0],
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    print(metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="uie-method-library")
    sub = parser.add_subparsers(dest="command", required=True)
    list_parser = sub.add_parser("list")
    list_parser.add_argument("--deep", action="store_true")
    list_parser.add_argument("--status", default="")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--method", required=True)
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        command_list(args.deep, args.status)
    else:
        command_run(args.method, args.input, args.output, args.device, args.metadata)


if __name__ == "__main__":
    main()
