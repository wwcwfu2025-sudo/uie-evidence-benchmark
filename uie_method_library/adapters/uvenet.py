from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from uie_method_library.core import AdapterError, MethodAdapter


class UveNetAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("UVE-Net repository or weight missing")
        sys.path.insert(0, str(repo))
        if self.spec.method_id == "D003":
            from basicsr.archs.deblur_arch_small import Deblur_samll

            self.model = Deblur_samll(num_feat=12, num_block=15)
            self.variant = "small"
        else:
            from basicsr.archs.deblur_arch import Deblur

            self.model = Deblur(num_feat=24, num_block=15)
            self.variant = "large"
        self.device = torch.device("cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request)
        checkpoint = torch.load(str(weight), map_location=self.device)
        state = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        original_h, original_w = bgr.shape[:2]
        scale = min(1.0, 512.0 / max(original_h, original_w))
        work_w, work_h = max(16, round(original_w * scale)), max(16, round(original_h * scale))
        resized = cv2.resize(bgr, (work_w, work_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).unsqueeze(0).to(self.device)
        pad_h = (16 - work_h % 16) % 16
        pad_w = (16 - work_w % 16) % 16
        batch, frames, channels, _, _ = tensor.shape
        tensor_2d = tensor.reshape(batch * frames, channels, work_h, work_w)
        tensor_2d = F.pad(tensor_2d, (0, pad_w, 0, pad_h), mode="reflect")
        tensor = tensor_2d.reshape(batch, frames, channels, work_h + pad_h, work_w + pad_w)
        with torch.no_grad():
            output, _ = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        output = output[0, 0, :, :work_h, :work_w].detach().cpu().clamp(0, 1)
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update({"variant": getattr(self, "variant", ""), "temporal_protocol": "single-frame sequence (T=1)", "max_internal_long_side": 512})
        return data
