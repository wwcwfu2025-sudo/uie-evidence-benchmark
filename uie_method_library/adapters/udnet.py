from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class UdnetAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("UDnet repository or weight missing")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from model_utils.UDnet import Decoder

        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.model = Decoder(device=self.device).to(self.device)
        checkpoint = torch.load(str(weight), map_location=self.device)
        decoder_state = {
            key[len("decoder.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("decoder.")
        }
        incompatible = self.model.load_state_dict(decoder_state, strict=False)
        if incompatible.missing_keys:
            raise AdapterError("UDnet checkpoint is missing decoder keys: %s" % incompatible.missing_keys[:5])
        self.ignored_checkpoint_keys = list(incompatible.unexpected_keys)
        self.model.eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        original_h, original_w = bgr.shape[:2]
        scale = min(1.0, 512.0 / max(original_h, original_w))
        work_w = max(8, (round(original_w * scale) // 8) * 8)
        work_h = max(8, (round(original_h * scale) // 8) * 8)
        resized = cv2.resize(
            bgr,
            (work_w, work_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor, tensor, training=False)[0].detach().cpu().clamp(0, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "inference_path": "Decoder only; VGG training-loss module intentionally not instantiated",
                "latent_protocol": "released deterministic inference uses posterior means (sigma multiplier 0)",
                "size_protocol": "aspect-preserving max long side 512; dimensions rounded down to multiple of 8",
                "ignored_checkpoint_keys": getattr(self, "ignored_checkpoint_keys", []),
            }
        )
        return data
