from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class DICAMAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        source = repo / "UIEB" / "Config" / "models.py"
        if not source.exists() or not weight.exists():
            raise AdapterError("DICAM source or UIEB checkpoint missing")
        module_spec = importlib.util.spec_from_file_location("uie_official_dicam_models", str(source))
        if module_spec is None or module_spec.loader is None:
            raise AdapterError("Cannot import official DICAM model source")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.model = module.DICAM().to(self.device)
        checkpoint = torch.load(str(weight), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
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
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update({"checkpoint_training_set": "UIEB", "official_test_input": "256x256 RGB [0,1]"})
        return data
