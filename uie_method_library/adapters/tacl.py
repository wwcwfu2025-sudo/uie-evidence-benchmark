from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class TACLAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        source = repo / "models" / "networks.py"
        if not source.exists() or not weight.exists():
            raise AdapterError("TACL source or generator checkpoint missing")
        module_spec = importlib.util.spec_from_file_location("uie_official_tacl_networks", str(source))
        if module_spec is None or module_spec.loader is None:
            raise AdapterError("Cannot import official TACL generator source")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        norm = module.get_norm_layer(norm_type="instance")
        self.model = module.ResnetGenerator(3, 3, 64, norm_layer=norm, use_dropout=False, n_blocks=9).to(self.device)
        self.model.load_state_dict(torch.load(str(weight), map_location=self.device), strict=True)
        self.model.eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        height, width = bgr.shape[:2]
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        work = cv2.copyMakeBorder(bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = ((self.model(tensor)[0] + 1.0) * 0.5).detach().cpu().clamp(0, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)[:height, :width]

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "generator": "official latest_net_G_A, ResNet-9",
                "normalization": "RGB [-1,1]",
                "geometry_protocol": "reflect-pad right/bottom to multiple of 4, then exact crop",
            }
        )
        return data
