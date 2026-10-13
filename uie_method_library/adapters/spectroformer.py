from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class SpectroformerAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("Spectroformer repository or checkpoint missing")
        for name in ("model_without_CA", "Final_model_AGSSF", "final_model_AGSSF1", "network1"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(repo))
        try:
            self.device = torch.device(
                "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
            )
            self.model = torch.load(str(weight), map_location=self.device).to(self.device)
        finally:
            if sys.path and sys.path[0] == str(repo):
                sys.path.pop(0)
        self.model.eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        height, width = bgr.shape[:2]
        resized = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(tensor)
            output = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
            output = ((output[0] + 1.0) * 0.5).detach().cpu().clamp(0, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (width, height), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "checkpoint_form": "official repository pickled full model",
                "official_input": "512x512 RGB [-1,1]",
                "output_selection": "first network output, as released test.py",
            }
        )
        return data
