from __future__ import annotations

import os

import argparse
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from PIL import Image  # Import Pillow and Torch before OpenCV on Windows to avoid duplicate OpenMP loading.
import torch
import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
REVISION_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

DATA_ROOT = REVISION_ROOT / "datasets" / "URPC2020" / "URPC2020"
TEST_IMAGES = DATA_ROOT / "test" / "images"
TEST_LABELS = DATA_ROOT / "test" / "labels"
IEUOD_ROOT = Path(os.environ.get("IEUOD_ROOT", "third_party/IEUOD"))
DETECTOR_WEIGHT = IEUOD_ROOT / "model" / "ours" / "raw.pt"
RUN_ROOT = REVISION_ROOT / "benchmark_runs" / "urpc2020_yolov12n_detection"
GENERATED_ROOT = RUN_ROOT / "images"
PREDICTION_ROOT = RUN_ROOT / "predictions"
RAW_RESULT_DIR = REVISION_ROOT / "raw_results" / "detection_urpc2020"
DEFAULT_METHODS = ["C000", "C020", "D001", "D002", "D007", "D044", "D046", "D048"]
DISPLAY_NAMES = {
    "C000": "RAW",
    "C020": "Fusion",
    "D001": "FUnIE-GAN",
    "D002": "P2CNet",
    "D007": "UIEC2-Net",
    "D044": "TACL",
    "D046": "Histoformer",
    "D048": "AutoEnhancer",
}
CLASS_NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}
ENHANCEMENT_LONG_SIDE = 512


def read_bgr(path: Path) -> np.ndarray:
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    encoded.tofile(str(path))


def clip_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def resize_long_side(image: np.ndarray, maximum: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= maximum:
        return image
    scale = maximum / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def gray_world(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    means = values.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    return clip_u8(values * (target / (means + 1e-6)))


def lab_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def fusion_ancuti_inspired(image: np.ndarray) -> np.ndarray:
    """Deterministic two-input fusion used in the quality benchmark."""
    corrected = gray_world(image)
    contrast_enhanced = lab_clahe(image)

    def weight(candidate: np.ndarray) -> np.ndarray:
        values = candidate.astype(np.float32) / 255.0
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        saturation = np.std(values, axis=2)
        saliency = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 5.0))
        return contrast + saturation + saliency + 1e-4

    weight_corrected = weight(corrected)
    weight_contrast = weight(contrast_enhanced)
    total = weight_corrected + weight_contrast
    fused = (
        corrected.astype(np.float32) * (weight_corrected / total)[:, :, None]
        + contrast_enhanced.astype(np.float32) * (weight_contrast / total)[:, :, None]
    )
    return clip_u8(fused)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def test_items(max_images: int = 0) -> list[dict[str, object]]:
    paths = sorted(TEST_IMAGES.glob("*.jpg"), key=lambda path: path.name)
    if max_images:
        paths = paths[:max_images]
    items: list[dict[str, object]] = []
    for index, path in enumerate(paths, 1):
        image = read_bgr(path)
        height, width = image.shape[:2]
        label_path = TEST_LABELS / f"{path.stem}.txt"
        ground_truth: list[dict[str, object]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            category = int(float(fields[0]))
            xc, yc, bw, bh = map(float, fields[1:5])
            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height
            ground_truth.append(
                {
                    "category_id": category,
                    "xyxy": [x1, y1, x2, y2],
                    "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                }
            )
        items.append(
            {
                "image_id": index,
                "stem": path.stem,
                "path": path,
                "width": width,
                "height": height,
                "ground_truth": ground_truth,
            }
        )
    return items


def prepare_images(method_ids: Sequence[str], items: Sequence[dict[str, object]], resume: bool) -> None:
    # Enhancement adapters are deliberately imported only in the generation
    # process.  The detector process uses a separate Torch/YOLO environment and
    # must not preload SciPy/scikit-image OpenMP runtimes through this module.
    from uie_method_library.adapters import build_adapter
    from uie_method_library.registry import registry_by_id

    registry = registry_by_id()
    for method_id in method_ids:
        output_dir = GENERATED_ROOT / method_id
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume and all((output_dir / f"{item['stem']}.png").exists() for item in items):
            print(f"[generate] {method_id} complete ({len(items)} images)", flush=True)
            continue
        adapter = None
        if method_id not in {"C000", "C020"}:
            spec = registry[method_id]
            if not spec.status.startswith("ready_verified"):
                raise RuntimeError(f"{method_id} is not benchmark-ready: {spec.status}")
            adapter = build_adapter(spec, "auto")
            adapter.load()
        start = time.perf_counter()
        for index, item in enumerate(items, 1):
            destination = output_dir / f"{item['stem']}.png"
            if resume and destination.exists():
                continue
            image = read_bgr(Path(item["path"]))
            original_shape = image.shape[:2]
            working_image = image if method_id == "C000" else resize_long_side(
                image, ENHANCEMENT_LONG_SIDE
            )
            if method_id == "C020":
                output = fusion_ancuti_inspired(working_image)
            elif adapter is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                output = adapter.enhance(working_image)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            else:
                output = image
            if output.shape[:2] != original_shape:
                output = cv2.resize(
                    output,
                    (original_shape[1], original_shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            write_image(destination, np.clip(output, 0, 255).astype(np.uint8))
            if index % 100 == 0 or index == len(items):
                print(f"[generate] {method_id} {index}/{len(items)}", flush=True)
        print(
            f"[generate] {method_id} finished in {time.perf_counter() - start:.1f}s",
            flush=True,
        )
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_detector(method_ids: Sequence[str], items: Sequence[dict[str, object]], resume: bool) -> None:
    if str(IEUOD_ROOT) not in sys.path:
        sys.path.insert(0, str(IEUOD_ROOT))
    from ultralytics import YOLO

    model = YOLO(str(DETECTOR_WEIGHT))
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != CLASS_NAMES:
        raise RuntimeError(f"detector class order mismatch: {names}")
    image_id_by_stem = {str(item["stem"]): int(item["image_id"]) for item in items}
    for method_id in method_ids:
        prediction_path = PREDICTION_ROOT / f"{method_id}.json"
        if resume and prediction_path.exists():
            payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            if len(payload.get("images", [])) == len(items):
                print(f"[detect] {method_id} complete ({len(items)} images)", flush=True)
                continue
        predictions: list[dict[str, object]] = []
        processed: list[int] = []
        start = time.perf_counter()
        results = model.predict(
            source=str(GENERATED_ROOT / method_id),
            imgsz=640,
            conf=0.001,
            iou=0.70,
            max_det=300,
            device=0 if torch.cuda.is_available() else "cpu",
            batch=4,
            stream=True,
            verbose=False,
        )
        for index, result in enumerate(results, 1):
            image_id = image_id_by_stem[Path(result.path).stem]
            processed.append(image_id)
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                for box, score, category in zip(boxes, scores, classes):
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(category),
                            "score": float(score),
                            "xyxy": [float(value) for value in box],
                        }
                    )
            if index % 100 == 0 or index == len(items):
                print(f"[detect] {method_id} {index}/{len(items)}", flush=True)
        payload = {
            "method_id": method_id,
            "images": processed,
            "predictions": predictions,
            "elapsed_seconds": time.perf_counter() - start,
            "inference": {"imgsz": 640, "conf": 0.001, "nms_iou": 0.70, "max_det": 300},
        }
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def box_iou(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    box_array = np.asarray(box, dtype=np.float32)
    xx1 = np.maximum(box_array[0], boxes[:, 0])
    yy1 = np.maximum(box_array[1], boxes[:, 1])
    xx2 = np.minimum(box_array[2], boxes[:, 2])
    yy2 = np.minimum(box_array[3], boxes[:, 3])
    intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = max(0.0, box_array[2] - box_array[0]) * max(0.0, box_array[3] - box_array[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if len(recalls) == 0:
        return 0.0
    values = []
    for level in np.linspace(0.0, 1.0, 101):
        mask = recalls >= level
        values.append(float(np.max(precisions[mask])) if np.any(mask) else 0.0)
    return float(np.mean(values))


def evaluate_class(
    predictions: list[dict[str, object]],
    ground_truth: dict[int, list[dict[str, object]]],
    category_id: int,
    iou_threshold: float,
) -> tuple[float, int]:
    gt_boxes: dict[int, np.ndarray] = {}
    total = 0
    for image_id, annotations in ground_truth.items():
        boxes = np.asarray(
            [item["xyxy"] for item in annotations if item["category_id"] == category_id],
            dtype=np.float32,
        ).reshape((-1, 4))
        gt_boxes[image_id] = boxes
        total += len(boxes)
    candidates = sorted(
        [item for item in predictions if item["category_id"] == category_id],
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_boxes.items()}
    true_positive = np.zeros(len(candidates), dtype=np.float32)
    false_positive = np.zeros(len(candidates), dtype=np.float32)
    for index, prediction in enumerate(candidates):
        image_id = int(prediction["image_id"])
        boxes = gt_boxes.get(image_id, np.zeros((0, 4), dtype=np.float32))
        ious = box_iou(prediction["xyxy"], boxes)
        hit = next(
            (
                candidate
                for candidate in np.argsort(-ious)
                if ious[candidate] >= iou_threshold and not matched[image_id][candidate]
            ),
            None,
        )
        if hit is None:
            false_positive[index] = 1.0
        else:
            matched[image_id][hit] = True
            true_positive[index] = 1.0
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recalls = tp / max(total, 1)
    precisions = tp / np.maximum(tp + fp, 1e-12)
    return average_precision(recalls, precisions), total


def fixed_counts(
    predictions: list[dict[str, object]],
    ground_truth: dict[int, list[dict[str, object]]],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> tuple[int, int, int, dict[int, dict[str, int]]]:
    per_image = {
        image_id: {"tp": 0, "fp": 0, "fn": len(annotations)}
        for image_id, annotations in ground_truth.items()
    }
    for category_id in CLASS_NAMES:
        for image_id, annotations in ground_truth.items():
            boxes = np.asarray(
                [item["xyxy"] for item in annotations if item["category_id"] == category_id],
                dtype=np.float32,
            ).reshape((-1, 4))
            candidates = sorted(
                [
                    item
                    for item in predictions
                    if item["image_id"] == image_id
                    and item["category_id"] == category_id
                    and float(item["score"]) >= score_threshold
                ],
                key=lambda item: float(item["score"]),
                reverse=True,
            )
            matched = np.zeros(len(boxes), dtype=bool)
            for prediction in candidates:
                ious = box_iou(prediction["xyxy"], boxes)
                hit = next(
                    (
                        candidate
                        for candidate in np.argsort(-ious)
                        if ious[candidate] >= iou_threshold and not matched[candidate]
                    ),
                    None,
                )
                if hit is None:
                    per_image[image_id]["fp"] += 1
                else:
                    matched[hit] = True
                    per_image[image_id]["tp"] += 1
            per_image[image_id]["fn"] -= int(np.sum(matched))
    tp = sum(item["tp"] for item in per_image.values())
    fp = sum(item["fp"] for item in per_image.values())
    fn = sum(item["fn"] for item in per_image.values())
    return tp, fp, fn, per_image


def evaluate(method_ids: Sequence[str], items: Sequence[dict[str, object]]) -> None:
    ground_truth = {int(item["image_id"]): item["ground_truth"] for item in items}
    summary_rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []
    for method_id in method_ids:
        payload = json.loads((PREDICTION_ROOT / f"{method_id}.json").read_text(encoding="utf-8"))
        predictions = payload["predictions"]
        by_threshold: dict[float, list[float]] = defaultdict(list)
        ap50_per_class: dict[int, float] = {}
        for threshold in np.arange(0.50, 0.951, 0.05):
            threshold = round(float(threshold), 2)
            for category_id in CLASS_NAMES:
                ap, count = evaluate_class(predictions, ground_truth, category_id, threshold)
                if count:
                    by_threshold[threshold].append(ap)
                    if threshold == 0.50:
                        ap50_per_class[category_id] = ap
        map_50_95 = float(np.mean([value for values in by_threshold.values() for value in values]))
        map_50 = float(np.mean(by_threshold[0.50]))
        map_75 = float(np.mean(by_threshold[0.75]))
        tp, fp, fn, per_image = fixed_counts(predictions, ground_truth)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        summary_rows.append(
            {
                "method_id": method_id,
                "method": DISPLAY_NAMES[method_id],
                "n_images": len(items),
                "n_ground_truth": sum(len(value) for value in ground_truth.values()),
                "bbox_map_50_95": map_50_95,
                "bbox_map_50": map_50,
                "bbox_map_75": map_75,
                "precision_conf025_iou050": precision,
                "recall_conf025_iou050": recall,
                "f1_conf025_iou050": f1,
                "tp_conf025_iou050": tp,
                "fp_conf025_iou050": fp,
                "fn_conf025_iou050": fn,
                "detector_seconds": float(payload["elapsed_seconds"]),
                "detector_ms_per_image": 1000.0 * float(payload["elapsed_seconds"]) / len(items),
                "ap50_per_class_json": json.dumps(ap50_per_class, ensure_ascii=False),
            }
        )
        item_by_id = {int(item["image_id"]): item for item in items}
        for image_id, counts in sorted(per_image.items()):
            denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
            per_image_rows.append(
                {
                    "method_id": method_id,
                    "method": DISPLAY_NAMES[method_id],
                    "image_id": image_id,
                    "file_name": f"{item_by_id[image_id]['stem']}.jpg",
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "f1": 2 * counts["tp"] / denominator if denominator else 1.0,
                }
            )
    raw = next(row for row in summary_rows if row["method_id"] == "C000")
    for row in summary_rows:
        row["delta_map_50_95_pp_vs_raw"] = 100.0 * (
            float(row["bbox_map_50_95"]) - float(raw["bbox_map_50_95"])
        )
        row["delta_f1_pp_vs_raw"] = 100.0 * (
            float(row["f1_conf025_iou050"]) - float(raw["f1_conf025_iou050"])
        )

    per_image_lookup = {
        (str(row["method_id"]), int(row["image_id"])): float(row["f1"])
        for row in per_image_rows
    }
    raw_by_image = {
        image_id: value
        for (method_id, image_id), value in per_image_lookup.items()
        if method_id == "C000"
    }
    rng = np.random.default_rng(20261012)
    bootstrap_rows: list[dict[str, object]] = []
    for method_id in method_ids:
        if method_id == "C000":
            continue
        image_ids = sorted(
            image_id
            for image_id in raw_by_image
            if (method_id, image_id) in per_image_lookup
        )
        differences = np.asarray(
            [per_image_lookup[(method_id, image_id)] - raw_by_image[image_id] for image_id in image_ids],
            dtype=np.float64,
        )
        replicates = np.empty(2000, dtype=np.float64)
        for replicate in range(len(replicates)):
            indices = rng.integers(0, len(differences), size=len(differences))
            replicates[replicate] = float(np.mean(differences[indices]))
        low, high = np.percentile(replicates, [2.5, 97.5])
        bootstrap_rows.append(
            {
                "method_id": method_id,
                "method": DISPLAY_NAMES[method_id],
                "n_images": len(image_ids),
                "raw_mean_per_image_f1": float(np.mean([raw_by_image[value] for value in image_ids])),
                "enhanced_mean_per_image_f1": float(
                    np.mean([per_image_lookup[(method_id, value)] for value in image_ids])
                ),
                "delta_mean_per_image_f1": float(np.mean(differences)),
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "improved_images_percent": 100.0 * float(np.mean(differences > 0)),
                "worsened_images_percent": 100.0 * float(np.mean(differences < 0)),
                "unchanged_images_percent": 100.0 * float(np.mean(differences == 0)),
                "bootstrap_replicates": len(replicates),
                "seed": 20261012,
            }
        )
    write_csv(RAW_RESULT_DIR / "urpc2020_yolov12n_summary.csv", summary_rows)
    write_csv(RAW_RESULT_DIR / "urpc2020_yolov12n_per_image.csv", per_image_rows)
    write_csv(RAW_RESULT_DIR / "urpc2020_yolov12n_paired_bootstrap.csv", bootstrap_rows)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "detector parameters held constant; identical weights and inference settings for RAW and enhanced inputs",
        "dataset": "URPC2020 official Kaggle package, test split",
        "dataset_root": str(DATA_ROOT),
        "dataset_archive_sha256": sha256(REVISION_ROOT / "datasets" / "URPC2020" / "urpc2020.zip"),
        "images": len(items),
        "annotations": sum(len(value) for value in ground_truth.values()),
        "classes": CLASS_NAMES,
        "methods": list(method_ids),
        "detector": "IEUOD repository YOLOv12n raw-input checkpoint",
        "detector_weight": str(DETECTOR_WEIGHT),
        "detector_weight_sha256": sha256(DETECTOR_WEIGHT),
        "inference": {"imgsz": 640, "conf_for_ap": 0.001, "nms_iou": 0.70, "max_det": 300},
        "fixed_operating_point": {"confidence": 0.25, "matching_iou": 0.50},
        "ap_implementation": (
            "class-wise greedy matching and 101-point interpolated AP at IoU 0.50:0.05:0.95"
        ),
        "uncertainty": (
            "paired image-level bootstrap of per-image F1 differences versus RAW; "
            "2000 replicates; seed 20261012"
        ),
        "geometry": "All enhancement outputs are resized back to the exact original width and height before detection.",
        "enhancement_long_side_cap": ENHANCEMENT_LONG_SIDE,
        "python": sys.version,
        "platform": platform.platform(),
    }
    RAW_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_RESULT_DIR / "urpc2020_yolov12n_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "detect", "evaluate", "all"), default="all")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    items = test_items(args.max_images)
    resume = not args.no_resume
    if args.stage in {"generate", "all"}:
        prepare_images(args.methods, items, resume)
    if args.stage in {"detect", "all"}:
        run_detector(args.methods, items, resume)
    if args.stage in {"evaluate", "all"}:
        evaluate(args.methods, items)


if __name__ == "__main__":
    main()

