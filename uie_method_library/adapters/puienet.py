from __future__ import annotations

import importlib
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class PUIENetAdapter(MethodAdapter):
    sample_count = 20

    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("PUIE-Net repository or checkpoint missing")
        for name in ("PUIENet_MC", "utils"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(repo))
        try:
            module = importlib.import_module("PUIENet_MC")
        finally:
            if sys.path and sys.path[0] == str(repo):
                sys.path.pop(0)
        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.model = module.Decoder(device=self.device).to(self.device)
        checkpoint = torch.load(str(weight), map_location=self.device)
        decoder_state = {
            key[len("decoder.") :]: value for key, value in checkpoint.items() if key.startswith("decoder.")
        }
        incompatible = self.model.load_state_dict(decoder_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AdapterError(
                "PUIE-Net decoder checkpoint mismatch; missing=%s unexpected=%s"
                % (incompatible.missing_keys[:5], incompatible.unexpected_keys[:5])
            )
        self.model.eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        height, width = bgr.shape[:2]
        pad_h = (-height) % 8
        pad_w = (-width) % 8
        work = cv2.copyMakeBorder(bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        seed = zlib.crc32(np.ascontiguousarray(bgr).tobytes()) & 0x7FFFFFFF
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            samples = [self.model(tensor, tensor, training=False) for _ in range(self.sample_count)]
            output = torch.stack(samples, dim=0).mean(dim=0)[0].detach().cpu().clamp(0, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)[:height, :width]

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "inference_path": "official MC decoder only; training-only VGG module omitted",
                "uncertainty_protocol": "mean of 20 stochastic samples, as released Test_MC.py",
                "reproducibility": "per-image CRC32 seed",
                "geometry_protocol": "reflect-pad right/bottom to multiple of 8, then exact crop",
            }
        )
        return data
