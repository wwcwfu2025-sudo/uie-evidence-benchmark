from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class SGUIEAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        source = repo / "models" / "networks.py"
        if not source.exists() or not weight.exists():
            raise AdapterError("SGUIE-Net source or checkpoint missing")
        module_spec = importlib.util.spec_from_file_location("uie_official_sguie_networks", str(source))
        if module_spec is None or module_spec.loader is None:
            raise AdapterError("Cannot import SGUIE-Net source")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.model = module.SGUIENet(3, 3, 64).to(self.device)
        self.model.load_state_dict(torch.load(str(weight), map_location=self.device), strict=True)
        self.model.eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        original_h, original_w = bgr.shape[:2]
        # Match the released TestOptions dimensions and raw-image normalization.
        resized = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        raw = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        # The released test path requires externally supplied semantic masks but
        # ships no segmentation predictor. For mask-free datasets, expose a
        # single full-frame region instead of silently fabricating semantics.
        mask = torch.ones((1, 1, 480, 640), dtype=raw.dtype, device=self.device)
        with torch.no_grad():
            output, _ = self.model(raw, [raw], [mask], [(0, 0, 640, 480)])
            output = ((output[0] + 1.0) * 0.5).detach().cpu().clamp(0, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "official_test_size": "640x480",
                "semantic_input": "single full-frame region fallback",
                "comparability_warning": "not equivalent to evaluation with ground-truth or predicted semantic masks",
            }
        )
        return data
