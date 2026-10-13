from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from uie_method_library.core import AdapterError, MethodAdapter


class CLUIENetAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        encoder_weight = Path(self.spec.weights)
        decoder_weight = encoder_weight.parent / "fI_latest.pth"
        if not repo.exists() or not encoder_weight.exists() or not decoder_weight.exists():
            raise AdapterError("CLUIE-Net source or one of the two checkpoints is missing")
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                sys.modules.pop(name, None)
        sys.path.insert(0, str(repo))
        try:
            module = importlib.import_module("models.networks")
        finally:
            if sys.path and sys.path[0] == str(repo):
                sys.path.pop(0)
        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.encoder = module.UNetEncoder().to(self.device)
        self.decoder = module.UNetDecoder().to(self.device)
        self.encoder.load_state_dict(torch.load(str(encoder_weight), map_location=self.device), strict=True)
        self.decoder.load_state_dict(torch.load(str(decoder_weight), map_location=self.device), strict=True)
        self.encoder.eval()
        self.decoder.eval()
        self.decoder_weight = str(decoder_weight)
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        pad_h = (-height) % 16
        pad_w = (-width) % 16
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            encoded, features = self.encoder(tensor)
            output = (self.decoder(encoded, features) + 1.0) * 0.5
            output = output[0, :, :height, :width].detach().cpu().clamp(0, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "inference_stages": "official UNetEncoder fE + UNetDecoder fI",
                "decoder_weight": getattr(self, "decoder_weight", ""),
                "geometry_protocol": "reflect-pad right/bottom to multiple of 16, then exact crop",
            }
        )
        return data
