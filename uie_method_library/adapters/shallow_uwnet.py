from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class ShallowUWNetAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not (repo / "model.py").exists() or not weight.exists():
            raise AdapterError("Shallow-UWNet source or checkpoint missing")
        # The released checkpoint pickles ``model.UWnet`` rather than a state
        # dict, so its original top-level module name must be importable.
        for name in list(sys.modules):
            if name == "model" or name.startswith("model."):
                del sys.modules[name]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import model  # noqa: F401

        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.model = torch.load(str(weight), map_location=self.device).to(self.device).eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        original_h, original_w = bgr.shape[:2]
        resized = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)[0].detach().cpu().clamp(0, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update({"official_test_input": "256x256 RGB [0,1]", "checkpoint_format": "pickled full model"})
        return data
