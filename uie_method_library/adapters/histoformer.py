from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


def _load_source(name: str, source: Path):
    module_spec = importlib.util.spec_from_file_location(name, str(source))
    if module_spec is None or module_spec.loader is None:
        raise AdapterError("Cannot import source: %s" % source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _histogram_pdf(rgb: np.ndarray) -> np.ndarray:
    rows = []
    for channel in range(3):
        hist = np.histogram(rgb[:, :, channel], bins=256, range=(0, 256))[0].astype(np.float32)
        rows.append(hist / max(1.0, float(hist.sum())))
    return np.stack(rows, axis=0)


def _map_channel(channel: np.ndarray, predicted_pdf: np.ndarray) -> np.ndarray:
    values, inverse, counts = np.unique(channel.ravel(), return_inverse=True, return_counts=True)
    source_quantiles = np.cumsum(counts).astype(np.float64)
    source_quantiles /= source_quantiles[-1]
    target_quantiles = np.cumsum(predicted_pdf.astype(np.float64))
    if not np.isfinite(target_quantiles).all() or abs(target_quantiles[-1]) < 1e-12:
        raise AdapterError("Histoformer predicted an invalid histogram")
    target_quantiles /= target_quantiles[-1]
    target_quantiles = np.maximum.accumulate(target_quantiles)
    mapped = np.interp(source_quantiles, target_quantiles, np.arange(256, dtype=np.float64))
    return np.clip(mapped[inverse].reshape(channel.shape), 0, 255).astype(np.uint8)


class HistoformerAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        histogram_weight = Path(self.spec.weights)
        generator_weight = histogram_weight.parent / "Histoformer-PQR_netG_288_modifyloss.pth"
        if not repo.exists() or not histogram_weight.exists() or not generator_weight.exists():
            raise AdapterError("Histoformer source or one of its two inference checkpoints is missing")
        if not torch.cuda.is_available() and self.device_request == "auto":
            raise AdapterError("Released Histoformer Generator hardcodes CUDA tensors")
        self.device = torch.device("cuda" if self.device_request == "auto" else self.device_request)
        hist_module = _load_source("uie_official_histoformer_model", repo / "model_histoformer.py")
        generator_module = _load_source("uie_official_histoformer_generator", repo / "Generator.py")
        self.histogram_model = hist_module.Histoformer(
            embed_dim=32, token_projection="linear", token_mlp="TwoDCFF"
        ).to(self.device)
        self.generator = generator_module.Generator().to(self.device)
        histogram_checkpoint = torch.load(str(histogram_weight), map_location=self.device)
        generator_checkpoint = torch.load(str(generator_weight), map_location=self.device)
        self.histogram_model.load_state_dict(histogram_checkpoint["state_dict"], strict=True)
        self.generator.load_state_dict(generator_checkpoint["state_dict"], strict=True)
        self.histogram_model.eval()
        self.generator.eval()
        self.generator_weight = str(generator_weight)
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        height, width = bgr.shape[:2]
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        work = cv2.copyMakeBorder(bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        histogram = torch.from_numpy(_histogram_pdf(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predicted = self.histogram_model(histogram)[0].detach().cpu().numpy()
        matched = work.copy()
        matched[:, :, 2] = _map_channel(work[:, :, 2], predicted[0])
        matched[:, :, 1] = _map_channel(work[:, :, 1], predicted[1])
        matched[:, :, 0] = _map_channel(work[:, :, 0], predicted[2])
        tensor = torch.from_numpy(
            np.ascontiguousarray(matched.astype(np.float32).transpose(2, 0, 1) / 255.0)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.generator(tensor)[0].detach().cpu().clamp(0, 1)
        torch.cuda.synchronize()
        bgr_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return bgr_out[:height, :width]

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "inference_stages": "predicted RGB histograms -> per-channel histogram mapping -> official refinement Generator",
                "generator_weight": getattr(self, "generator_weight", ""),
                "geometry_protocol": "reflect-pad right/bottom to multiple of 4, then exact crop",
                "compatibility_note": "adapter supplies missing cv2 dependency in released utils and preserves official BGR refinement order",
            }
        )
        return data
