from __future__ import annotations

import sys
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


def _load_state_dict(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(str(path), map_location=device)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise AdapterError("Unsupported UWEnhancement checkpoint structure: %s" % path)
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def _bytescale(array: np.ndarray) -> np.ndarray:
    """Match the toolbox's scipy.misc.bytescale-style whole-image scaling."""
    low = float(np.nanmin(array))
    high = float(np.nanmax(array))
    if not np.isfinite(low) or not np.isfinite(high):
        raise AdapterError("Network output contains NaN or infinity")
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - low) * (255.0 / (high - low)), 0.0, 255.0).round().astype(np.uint8)


def _simplest_color_balance(rgb_u8: np.ndarray) -> np.ndarray:
    """Port of the official Water-Net SimplestColorBalance.m preprocessing."""
    values = rgb_u8.astype(np.float64)
    channel_sums = values.sum(axis=(0, 1))
    maximum = max(float(channel_sums.max()), 1.0)
    ratios = maximum / np.maximum(channel_sums, 1.0)
    output = np.empty_like(values)
    for channel in range(3):
        saturation = min(0.499, 0.005 * float(ratios[channel]))
        low, high = np.quantile(values[:, :, channel], [saturation, 1.0 - saturation])
        clipped = np.clip(values[:, :, channel], low, high)
        if high <= low:
            output[:, :, channel] = clipped
        else:
            output[:, :, channel] = (clipped - low) * (255.0 / (high - low))
    return np.clip(output, 0, 255).astype(np.uint8)


def _waternet_inputs(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wb = _simplest_color_balance(rgb_u8)
    # MATLAB's legacy srgb2lab representation and OpenCV's uint8 Lab both encode
    # all three channels in [0, 255]. CLAHE defaults are the closest portable
    # equivalent to adapthisteq; this conversion difference is exposed in metadata.
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    ce = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gc = np.clip((rgb_u8.astype(np.float32) / 255.0) ** 0.7 * 255.0, 0, 255).astype(np.uint8)
    return wb, ce, gc


class UwEnhancementToolboxAdapter(MethodAdapter):
    """Adapter for the author-linked UWEnhancement PyTorch model zoo."""

    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("UWEnhancement repository or weight missing")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        # The toolbox imports the third-party ``regex`` package only on Windows,
        # although its configuration reader uses standard ``re.sub`` semantics.
        # Supplying the standard-library module avoids an irrelevant dependency.
        sys.modules.setdefault("regex", re)

        self.device = _resolve_device(self.device_request)
        if self.spec.method_id == "D005":
            from core.Models.UWModels.WaterNet import WaterNet

            self.model = WaterNet()
            self.variant = "WaterNet converted checkpoint"
        elif self.spec.name.startswith("UWCNN"):
            from core.Models.UWModels.UWCNN import UWCNN

            self.model = UWCNN()
            self.variant = self.spec.name.replace("UWCNN-", "UWCNN ").replace("UWCNN", "UWCNN type-1") if self.spec.name == "UWCNN" else self.spec.name
        elif self.spec.method_id == "D007":
            from core.Models.UWModels.UIEC2Net import UIEC2Net

            if self.device.type != "cuda":
                raise AdapterError("The released UIEC2-Net forward pass hard-codes CUDA tensors; use CUDA")
            self.model = UIEC2Net()
            self.variant = "UIEC2-Net released model-zoo checkpoint"
        else:
            raise AdapterError("Unsupported UWEnhancement method: %s" % self.spec.method_id)

        state = _load_state_dict(weight, self.device)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        self.loaded = True

    @staticmethod
    def _tensor(rgb_u8: np.ndarray, device: torch.device) -> torch.Tensor:
        rgb = np.ascontiguousarray(rgb_u8.astype(np.float32) / 255.0)
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        original_h, original_w = bgr.shape[:2]
        scale = min(1.0, 512.0 / max(original_h, original_w))
        work_w = max(8, round(original_w * scale))
        work_h = max(8, round(original_h * scale))
        resized = cv2.resize(
            bgr,
            (work_w, work_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        rgb_u8 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        raw = self._tensor(rgb_u8, self.device)
        with torch.no_grad():
            if self.spec.method_id == "D005":
                wb, ce, gc = _waternet_inputs(rgb_u8)
                output = self.model(
                    raw,
                    self._tensor(wb, self.device),
                    self._tensor(ce, self.device),
                    self._tensor(gc, self.device),
                )
            elif self.spec.name.startswith("UWCNN"):
                output = self.model(raw * 2.0 - 1.0)
            else:
                output = self.model(raw)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        chw = output[0].detach().cpu().numpy()
        if self.spec.name.startswith("UWCNN"):
            rgb_out = _bytescale(((chw + 1.0) * 0.5 * 255.0).transpose(1, 2, 0))
        elif self.spec.method_id == "D005":
            rgb_out = _bytescale((chw * 255.0).transpose(1, 2, 0))
        else:
            rgb_out = np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255).round().astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "variant": getattr(self, "variant", ""),
                "max_internal_long_side": 512,
                "postprocess": "toolbox usebytescale setting reproduced",
            }
        )
        if self.spec.method_id == "D005":
            data["preprocess"] = (
                "official MATLAB SimplestColorBalance and gamma port; "
                "OpenCV uint8 Lab CLAHE used as portable adapthisteq equivalent"
            )
        return data
