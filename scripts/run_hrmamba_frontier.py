from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image  # load before Torch/OpenCV on Windows
import cv2
import numpy as np
import torch
import torch.nn.functional as F


REVISION_ROOT = Path(__file__).resolve().parents[1]
REPO = REVISION_ROOT / "external_methods" / "HRMamba-main"
MANIFEST = REVISION_ROOT / "manifests" / "frontier_staged_manifest.csv"
OUTPUT = REVISION_ROOT / "benchmark_runs" / "frontier_track" / "HRMamba"
METADATA = REVISION_ROOT / "raw_results" / "frontier" / "hrmamba_inference_metadata.json"
CHECKPOINT = REPO / "pretrained_models" / "HRMamba-UIEB.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"cannot encode {path}")
    encoded.tofile(str(path))


def cap_long_side(image: np.ndarray, maximum: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= maximum:
        return image
    scale = maximum / float(max(height, width))
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--long-side", type=int, default=256)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from net.HRMamba import HRMamba

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_images:
        rows = rows[: args.max_images]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("HRMamba reference inference requires a CUDA GPU")
    model = HRMamba().to(device)
    state = torch.load(str(CHECKPOINT), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        destination = OUTPUT / f"{row['staged_token']}.png"
        if destination.exists() and not args.no_resume:
            records.append({"staged_token": row["staged_token"], "status": "existing"})
            continue
        source = read_bgr(Path(row["staged_input"]))
        original_height, original_width = source.shape[:2]
        working = cap_long_side(source, args.long_side)
        rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        tensor = tensor.unsqueeze(0).to(device) / 255.0
        pad_height = (32 - tensor.shape[-2] % 32) % 32
        pad_width = (32 - tensor.shape[-1] % 32) % 32
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(tensor)
        torch.cuda.synchronize(device)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        output = output[..., : working.shape[0], : working.shape[1]]
        output = output.squeeze(0).clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        output = cv2.cvtColor(np.round(output * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if output.shape[:2] != (original_height, original_width):
            output = cv2.resize(
                output, (original_width, original_height), interpolation=cv2.INTER_CUBIC
            )
        write_png(destination, output)
        records.append(
            {
                "staged_token": row["staged_token"],
                "status": "ok",
                "runtime_ms_reference_backend": elapsed_ms,
                "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
                "working_width": working.shape[1],
                "working_height": working.shape[0],
                "pad_width": pad_width,
                "pad_height": pad_height,
            }
        )
        print(
            f"HRMamba {index}/{len(rows)} {elapsed_ms:.1f} ms "
            f"{working.shape[1]}x{working.shape[0]}",
            flush=True,
        )

    successful = [row for row in records if row["status"] == "ok"]
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "HRMamba",
        "official_repository": "https://github.com/YeFan-web/HRMamba",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_variant": "UIEB",
        "manifest": str(MANIFEST),
        "requested_images": len(rows),
        "newly_processed_images": len(successful),
        "long_side_cap": args.long_side,
        "padding": "reflect to next multiple of 32; crop after inference",
        "output_geometry": "resized back to staged input geometry",
        "backend": (
            "mamba-ssm 1.1.1 official parameterization with its pure-PyTorch "
            "selective_scan_ref; compiled CUDA kernels unavailable on Windows host"
        ),
        "latency_status": (
            "reference-backend diagnostic only; excluded from deployment latency comparison"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "python": sys.version,
        "platform": platform.platform(),
        "records": records,
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
