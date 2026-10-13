from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class UShapeTransformerAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("U-shape Transformer repository or weight missing")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from net.Ushape_Trans import Generator

        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.model = Generator().to(self.device)
        state = torch.load(str(weight), map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        original_h, original_w = bgr.shape[:2]
        resized = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)[3][0].detach().cpu().numpy().transpose(1, 2, 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # Official test.ipynb calls torchvision.save_image(..., normalize=True),
        # which applies whole-image min/max normalization before quantization.
        low, high = float(output.min()), float(output.max())
        if high <= low:
            rgb_out = np.zeros(output.shape, dtype=np.uint8)
        else:
            rgb_out = np.clip((output - low) * (255.0 / (high - low)), 0, 255).round().astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "official_test_input": "256x256 RGB [0,1]",
                "selected_output": "generator multi-scale output index 3",
                "postprocess": "official save_image normalize=True reproduced",
            }
        )
        return data
