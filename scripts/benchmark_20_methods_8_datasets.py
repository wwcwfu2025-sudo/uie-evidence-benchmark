from __future__ import annotations

import os

import argparse
import csv
import json
import math
import platform
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("UIE_DATA_ROOT", "datasets"))
FUNIE_REPO = Path(os.environ.get("FUNIE_REPO", "third_party/FUnIE-GAN/PyTorch"))
OUT_DIR = ROOT / "benchmark_runs" / "benchmark_20_methods_8_datasets"
ASSET_DIR = ROOT / "benchmark_assets"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_bgr(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("decode failed: %s" % path)
    return image


def write_jpg(path: Path, image: np.ndarray, quality: int = 93) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("encode failed: %s" % path)
    path.write_bytes(encoded.tobytes())


def resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, float(long_side) / float(max(h, w)))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def clip_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def percentile_stretch(channel: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(channel, [low, high])
    if hi <= lo + 1e-6:
        return channel.copy()
    return np.clip((channel.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def gray_world(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    means = x.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    return clip_u8(x * (target / (means + 1e-6)))


def max_rgb(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    maxima = np.percentile(x.reshape(-1, 3), 99.9, axis=0)
    return clip_u8(x * (255.0 / (maxima + 1e-6)))


def shades_of_gray(image: np.ndarray, p: float = 6.0) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    illuminant = np.power(np.mean(np.power(x + 1e-8, p), axis=(0, 1)), 1.0 / p)
    target = np.mean(illuminant)
    return clip_u8(x * (target / (illuminant + 1e-8)) * 255.0)


def gray_edge(image: np.ndarray, p: float = 6.0) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    grads = []
    for channel in cv2.split(x):
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        grads.append(np.power(np.mean(np.power(mag + 1e-8, p)), 1.0 / p))
    illuminant = np.asarray(grads, dtype=np.float32)
    target = float(np.mean(illuminant))
    return clip_u8(x * (target / (illuminant + 1e-8)) * 255.0)


def white_patch_percentile(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    high = np.percentile(x.reshape(-1, 3), 99.0, axis=0)
    return clip_u8(x * (255.0 / (high + 1e-6)))


def rgb_hist_equalization(image: np.ndarray) -> np.ndarray:
    return cv2.merge([cv2.equalizeHist(channel) for channel in cv2.split(image)])


def ycrcb_hist_equalization(image: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def lab_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def hsv_clahe(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gamma_transform(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def fixed_gamma(image: np.ndarray) -> np.ndarray:
    return gamma_transform(image, 0.7)


def adaptive_gamma(image: np.ndarray) -> np.ndarray:
    mean_luma = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() / 255.0)
    gamma = np.clip(math.log(0.5 + 1e-6) / math.log(mean_luma + 1e-6), 0.45, 1.8) if mean_luma > 0.01 else 0.45
    return gamma_transform(image, float(gamma))


def unsharp_mask(image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.addWeighted(image, 1.6, blur, -0.6, 0)


def homomorphic_filter(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    log_v = np.log1p(value)
    illumination = gaussian_filter(log_v, sigma=max(3.0, min(image.shape[:2]) / 35.0))
    reflectance = log_v - illumination
    restored = np.expm1(0.75 * illumination + 1.6 * reflectance)
    restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = restored
    return cv2.cvtColor(clip_u8(hsv), cv2.COLOR_HSV2BGR)


def retinex_channels(image: np.ndarray, sigmas: Sequence[float]) -> np.ndarray:
    x = image.astype(np.float32) + 1.0
    result = np.zeros_like(x, dtype=np.float32)
    for sigma in sigmas:
        for c in range(3):
            blur = cv2.GaussianBlur(x[:, :, c], (0, 0), sigmaX=sigma, sigmaY=sigma)
            result[:, :, c] += np.log(x[:, :, c]) - np.log(blur + 1e-6)
    result /= float(len(sigmas))
    output = np.empty_like(result)
    for c in range(3):
        lo, hi = np.percentile(result[:, :, c], [1.0, 99.0])
        output[:, :, c] = np.clip((result[:, :, c] - lo) * 255.0 / (hi - lo + 1e-6), 0, 255)
    return output


def ssr(image: np.ndarray) -> np.ndarray:
    return clip_u8(retinex_channels(image, [30.0]))


def msr(image: np.ndarray) -> np.ndarray:
    return clip_u8(retinex_channels(image, [15.0, 80.0, 250.0]))


def msrcr(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) + 1.0
    ret = retinex_channels(image, [15.0, 80.0, 250.0]) / 255.0
    color_restore = np.log(125.0 * x) - np.log(np.sum(x, axis=2, keepdims=True) + 1e-6)
    result = ret * color_restore
    output = np.empty_like(result)
    for c in range(3):
        lo, hi = np.percentile(result[:, :, c], [1.0, 99.0])
        output[:, :, c] = np.clip((result[:, :, c] - lo) * 255.0 / (hi - lo + 1e-6), 0, 255)
    return clip_u8(output)


def dark_channel(x: np.ndarray, channels: Sequence[int], patch: int = 15) -> np.ndarray:
    selected = np.min(x[:, :, list(channels)], axis=2)
    return cv2.erode(selected, cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))


def atmospheric_light(x: np.ndarray, dark: np.ndarray) -> np.ndarray:
    count = max(1, int(dark.size * 0.001))
    indices = np.argpartition(dark.ravel(), -count)[-count:]
    pixels = x.reshape(-1, 3)[indices]
    return pixels[np.argmax(np.sum(pixels, axis=1))]


def prior_restore(image: np.ndarray, mode: str) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    if mode == "dcp":
        proxy = x
        channels = (0, 1, 2)
    elif mode == "udcp":
        proxy = x
        channels = (0, 1)
    else:
        proxy = np.dstack((x[:, :, 0], x[:, :, 1], 1.0 - x[:, :, 2]))
        channels = (0, 1, 2)
    dark = dark_channel(proxy, channels)
    atmosphere = np.maximum(atmospheric_light(x, dark), 0.08)
    normalized = proxy / atmosphere.reshape(1, 1, 3)
    transmission = 1.0 - 0.92 * dark_channel(normalized, channels)
    transmission = cv2.bilateralFilter(transmission.astype(np.float32), 9, 0.08, 15)
    transmission = np.clip(transmission, 0.12, 1.0)
    restored = (x - atmosphere.reshape(1, 1, 3)) / transmission[:, :, None] + atmosphere.reshape(1, 1, 3)
    if mode == "rcp":
        restored[:, :, 2] *= 1.08
    return clip_u8(restored * 255.0)


def dcp(image: np.ndarray) -> np.ndarray:
    return prior_restore(image, "dcp")


def udcp(image: np.ndarray) -> np.ndarray:
    return prior_restore(image, "udcp")


def rcp(image: np.ndarray) -> np.ndarray:
    return prior_restore(image, "rcp")


def fusion_ancuti_inspired(image: np.ndarray) -> np.ndarray:
    a = gray_world(image)
    b = lab_clahe(image)

    def weight(x: np.ndarray) -> np.ndarray:
        xf = x.astype(np.float32) / 255.0
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        saturation = np.std(xf, axis=2)
        saliency = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 5.0))
        return contrast + saturation + saliency + 1e-4

    wa, wb = weight(a), weight(b)
    total = wa + wb
    fused = a.astype(np.float32) * (wa / total)[:, :, None] + b.astype(np.float32) * (wb / total)[:, :, None]
    return clip_u8(fused)


class FunieSample:
    def __init__(self, repo: Path) -> None:
        sys.path.insert(0, str(repo))
        from nets.funiegan import GeneratorFunieGAN

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GeneratorFunieGAN().to(self.device)
        weight = repo / "models" / "funie_generator.pth"
        self.model.load_state_dict(torch.load(str(weight), map_location=self.device))
        self.model.eval()
        self.weight = weight
        dummy = torch.zeros((1, 3, 256, 256), device=self.device)
        with torch.no_grad():
            for _ in range(3):
                self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        array = rgb.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            generated = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        out = generated[0].detach().cpu().clamp(-1, 1)
        out = ((out + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)


def uciqe(image: np.ndarray) -> float:
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    a = (lab[:, :, 1] - 128.0) / 127.0
    b = (lab[:, :, 2] - 128.0) / 127.0
    chroma = np.sqrt(a * a + b * b)
    saturation = chroma / np.sqrt(chroma * chroma + l * l + 1e-12)
    contrast = float(np.percentile(l, 99) - np.percentile(l, 1))
    return float(0.4680 * np.std(chroma) + 0.2745 * contrast + 0.2576 * np.mean(saturation))


def eme(channel: np.ndarray, blocks: int = 8) -> float:
    h, w = channel.shape
    bh, bw = max(1, h // blocks), max(1, w // blocks)
    values = []
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            patch = channel[y : y + bh, x : x + bw]
            low, high = float(np.min(patch)), float(np.max(patch))
            values.append(math.log((high + 1.0) / (low + 1.0)))
    return float(2.0 * np.mean(values)) if values else 0.0


def logamee(channel: np.ndarray, blocks: int = 8) -> float:
    h, w = channel.shape
    bh, bw = max(1, h // blocks), max(1, w // blocks)
    values = []
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            patch = channel[y : y + bh, x : x + bw]
            low, high = float(np.min(patch)), float(np.max(patch))
            ratio = (high - low) / (high + low + 1e-6)
            values.append(ratio * math.log(ratio + 1e-6))
    return float(-np.mean(values)) if values else 0.0


def uiqm(image: np.ndarray) -> float:
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = cv2.split(small)
    rg, yb = r - g, (r + g) * 0.5 - b

    def trimmed_stats(x: np.ndarray) -> Tuple[float, float]:
        lo, hi = np.percentile(x, [10, 90])
        selected = x[(x >= lo) & (x <= hi)]
        return float(np.mean(selected)), float(np.var(selected))

    mu_rg, var_rg = trimmed_stats(rg)
    mu_yb, var_yb = trimmed_stats(yb)
    uicm = -0.0268 * math.sqrt(mu_rg * mu_rg + mu_yb * mu_yb) + 0.1586 * math.sqrt(var_rg + var_yb)
    channel_eme = []
    for channel in (b, g, r):
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        channel_eme.append(eme(np.sqrt(gx * gx + gy * gy) * (channel / 255.0)))
    uism = 0.144 * channel_eme[0] + 0.587 * channel_eme[1] + 0.299 * channel_eme[2]
    gray = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    uiconm = logamee(gray)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def no_reference_metrics(image: np.ndarray) -> Dict[str, float]:
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(1.0, float(np.sum(hist)))
    entropy = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))
    means = small.reshape(-1, 3).mean(axis=0) / 255.0
    return {
        "uciqe": uciqe(small),
        "uiqm": uiqm(small),
        "entropy": entropy,
        "mean_luma": float(np.mean(gray) / 255.0),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "channel_mean_range": float(np.max(means) - np.min(means)),
        "dark_fraction": float(np.mean(gray <= 5)),
        "bright_fraction": float(np.mean(gray >= 250)),
    }


def reference_metrics(output: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    if output.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_AREA)
    out_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return {
        "psnr": float(peak_signal_noise_ratio(ref_rgb, out_rgb, data_range=1.0)),
        "ssim": float(structural_similarity(ref_rgb, out_rgb, data_range=1.0, channel_axis=2)),
        "mae": float(np.mean(np.abs(ref_rgb - out_rgb))),
        "delta_e2000": float(np.mean(deltaE_ciede2000(rgb2lab(ref_rgb), rgb2lab(out_rgb)))),
    }


def image_files(root: Path, suffixes: Optional[Sequence[str]] = None) -> List[Path]:
    allowed = set(x.lower() for x in suffixes) if suffixes else EXTENSIONS
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def evenly_spaced(paths: Sequence[Path], maximum: int) -> List[Path]:
    paths = list(paths)
    if maximum <= 0 or len(paths) <= maximum:
        return paths
    indices = np.linspace(0, len(paths) - 1, maximum, dtype=int)
    return [paths[int(i)] for i in indices]


def build_dataset_manifest(max_images: int) -> List[Dict[str, str]]:
    specs = []

    def add(dataset: str, inputs: Sequence[Path], reference_dir: Optional[Path] = None, maximum: Optional[int] = None) -> None:
        chosen = evenly_spaced(inputs, max_images if maximum is None else maximum)
        for path in chosen:
            reference = ""
            if reference_dir is not None:
                candidate = reference_dir / path.name
                if candidate.exists():
                    reference = str(candidate)
            specs.append({"dataset": dataset, "input": str(path), "reference": reference, "sample_key": path.stem})

    uieb_raw = DATA_ROOT / "UIEB" / "extracted" / "raw-890"
    add("UIEB", image_files(uieb_raw), DATA_ROOT / "UIEB" / "extracted" / "reference-890")
    lsui_input = DATA_ROOT / "LSUI" / "extracted" / "LSUI" / "input"
    add("LSUI", image_files(lsui_input), DATA_ROOT / "LSUI" / "extracted" / "LSUI" / "GT")
    add("U45", image_files(DATA_ROOT / "U45" / "extracted" / "underwater-test-dataset-U45--master" / "upload" / "U45" / "U45"))
    sub3d = DATA_ROOT / "Submerged3D" / "extracted" / "Submerged3D"
    add("Submerged3D", [p for p in image_files(sub3d) if p.parent.name == "input"])
    tank = DATA_ROOT / "TankImage-I" / "extracted" / "TankImage-I-main"
    add("TankImage-I", image_files(tank, [".bmp"]))
    cc_root = DATA_ROOT / "ColorCheck7-OceanDark" / "extracted" / "two-No-reference-image-dataset-main"
    add("ColorCheck7", image_files(cc_root / "Color-Check7"), maximum=0)
    add("OceanDark", image_files(cc_root / "OceanDark"))
    uid = DATA_ROOT / "UID2021" / "extracted"
    add("UID2021", sorted(p for p in image_files(uid) if p.name.startswith("SI_")))
    return specs


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[(row["dataset"], row["method"])].append(row)
    metrics = [
        "psnr", "ssim", "mae", "delta_e2000", "uciqe", "uiqm", "entropy", "mean_luma",
        "laplacian_variance", "channel_mean_range", "dark_fraction", "bright_fraction", "runtime_ms",
    ]
    result = []
    for (dataset, method), items in sorted(groups.items()):
        out = {
            "dataset": dataset,
            "method": method,
            "implementation_class": items[0]["implementation_class"],
            "device": items[0]["device"],
            "n_ok": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric, "") not in ("", None)]
            out[metric + "_mean"] = float(np.mean(values)) if values else ""
            out[metric + "_median"] = float(np.median(values)) if values else ""
        times = [float(item["runtime_ms"]) for item in items]
        out["runtime_ms_p95"] = float(np.percentile(times, 95)) if times else ""
        out["fps_from_mean_runtime"] = 1000.0 / float(np.mean(times)) if times and np.mean(times) > 0 else ""
        result.append(out)
    return result


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(max_images: int, long_side: int, save_samples: bool) -> None:
    manifest = build_dataset_manifest(max_images)
    write_csv(ASSET_DIR / "33_benchmark_20x8_fixed_manifest.csv", manifest)
    funie = FunieSample(FUNIE_REPO)
    methods: List[Tuple[str, Callable[[np.ndarray], np.ndarray], str, str]] = [
        ("RAW", lambda x: x.copy(), "control", "CPU"),
        ("GrayWorld", gray_world, "standardized_reimplementation", "CPU"),
        ("MaxRGB", max_rgb, "standardized_reimplementation", "CPU"),
        ("ShadesOfGray_p6", shades_of_gray, "standardized_reimplementation", "CPU"),
        ("GrayEdge_p6", gray_edge, "standardized_reimplementation", "CPU"),
        ("WhitePatch_p99", white_patch_percentile, "standardized_reimplementation", "CPU"),
        ("RGB_HE", rgb_hist_equalization, "standardized_reimplementation", "CPU"),
        ("YCrCb_HE", ycrcb_hist_equalization, "standardized_reimplementation", "CPU"),
        ("LAB_CLAHE", lab_clahe, "standardized_reimplementation", "CPU"),
        ("HSV_CLAHE", hsv_clahe, "standardized_reimplementation", "CPU"),
        ("Gamma_0p7", fixed_gamma, "standardized_reimplementation", "CPU"),
        ("AdaptiveGamma", adaptive_gamma, "standardized_reimplementation", "CPU"),
        ("UnsharpMask", unsharp_mask, "standardized_reimplementation", "CPU"),
        ("Homomorphic", homomorphic_filter, "standardized_reimplementation", "CPU"),
        ("SSR", ssr, "standardized_reimplementation", "CPU"),
        ("MSR", msr, "standardized_reimplementation", "CPU"),
        ("MSRCR", msrcr, "standardized_reimplementation", "CPU"),
        ("DCP", dcp, "standardized_reimplementation", "CPU"),
        ("UDCP", udcp, "standardized_reimplementation", "CPU"),
        ("RCP", rcp, "standardized_reimplementation", "CPU"),
        ("Fusion_Ancuti_inspired", fusion_ancuti_inspired, "standardized_reimplementation_not_original_code", "CPU"),
        ("FUnIEGAN_repo_sample", funie, "official_repository_sample_weight_not_paper_reproduction", str(funie.device)),
    ]
    method_registry = [
        {
            "method": name,
            "implementation_class": kind,
            "device": device,
            "common_input_long_side": long_side,
            "special_model_input": "256x256" if name == "FUnIEGAN_repo_sample" else "",
        }
        for name, _, kind, device in methods
    ]
    write_csv(ASSET_DIR / "34_benchmark_20x8_method_registry.csv", method_registry)

    rows: List[Dict[str, object]] = []
    dataset_counters = defaultdict(int)
    start_all = time.perf_counter()
    for sample_index, item in enumerate(manifest, 1):
        dataset = item["dataset"]
        input_path = Path(item["input"])
        image = resize_long_side(read_bgr(input_path), long_side)
        reference = resize_long_side(read_bgr(Path(item["reference"])), long_side) if item["reference"] else None
        if reference is not None and reference.shape[:2] != image.shape[:2]:
            reference = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        dataset_counters[dataset] += 1
        for name, function, implementation_class, device in methods:
            base = {
                "dataset": dataset,
                "sample_key": item["sample_key"],
                "input": str(input_path),
                "reference": item["reference"],
                "method": name,
                "implementation_class": implementation_class,
                "device": device,
                "width": image.shape[1],
                "height": image.shape[0],
                "status": "ok",
                "error": "",
            }
            try:
                start = time.perf_counter()
                output = function(image)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                runtime_ms = (time.perf_counter() - start) * 1000.0
                if output.shape[:2] != image.shape[:2]:
                    output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
                output = clip_u8(output)
                base["runtime_ms"] = runtime_ms
                base.update(reference_metrics(output, reference) if reference is not None else {"psnr": "", "ssim": "", "mae": "", "delta_e2000": ""})
                base.update(no_reference_metrics(output))
                if save_samples and dataset_counters[dataset] == 1:
                    write_jpg(OUT_DIR / "visual_samples" / dataset / (name + ".jpg"), output)
            except Exception as exc:
                base.update({
                    "status": "failed", "error": "%s: %s" % (type(exc).__name__, exc), "runtime_ms": "",
                    "psnr": "", "ssim": "", "mae": "", "delta_e2000": "", "uciqe": "", "uiqm": "",
                    "entropy": "", "mean_luma": "", "laplacian_variance": "", "channel_mean_range": "",
                    "dark_fraction": "", "bright_fraction": "",
                })
            rows.append(base)
        if sample_index % 5 == 0 or sample_index == len(manifest):
            write_csv(ASSET_DIR / "35_benchmark_20x8_per_image_metrics.csv", rows)
            print("progress %d/%d images; rows=%d; elapsed=%.1fs" % (sample_index, len(manifest), len(rows), time.perf_counter() - start_all), flush=True)

    summary = aggregate(rows)
    write_csv(ASSET_DIR / "35_benchmark_20x8_per_image_metrics.csv", rows)
    write_csv(ASSET_DIR / "36_benchmark_20x8_summary.csv", summary)
    failures = [row for row in rows if row["status"] != "ok"]
    write_csv(ASSET_DIR / "37_benchmark_20x8_failures.csv", failures)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_count": len(set(item["dataset"] for item in manifest)),
        "image_count": len(manifest),
        "enhancement_method_count_excluding_raw": len(methods) - 1,
        "rows": len(rows),
        "failures": len(failures),
        "selection": "deterministic evenly spaced sample, at most %d images per dataset; ColorCheck7 all images" % max_images,
        "common_preprocessing": "BGR decode; preserve aspect ratio; downscale longest side to %d; no upscaling" % long_side,
        "timing_boundary": "method preprocessing + enhancement/inference + method postprocessing; excludes file I/O, common resize, metric computation, and sample saving",
        "no_reference_metric_size": "256x256",
        "full_reference_metrics": ["PSNR", "SSIM", "MAE", "CIEDE2000"],
        "no_reference_metrics": ["UCIQE", "UIQM", "entropy", "mean_luma", "Laplacian_variance", "channel_mean_range", "dark_fraction", "bright_fraction"],
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "funie_weight": str(funie.weight),
        "warning": "Standardized reimplementations are protocol baselines, not claims of reproducing original paper tables. FUnIE result uses the official repository sample weight; repository README identifies TensorFlow/Keras as the paper-reproduction path.",
        "wall_seconds": time.perf_counter() - start_all,
    }
    (ASSET_DIR / "38_benchmark_20x8_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ASSET_DIR / "36_benchmark_20x8_summary.csv")
    print(ASSET_DIR / "38_benchmark_20x8_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--long-side", type=int, default=512)
    parser.add_argument("--save-samples", action="store_true")
    args = parser.parse_args()
    try:
        main(args.max_images, args.long_side, args.save_samples)
    except Exception:
        traceback.print_exc()
        raise
