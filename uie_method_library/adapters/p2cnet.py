from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import kornia

from uie_method_library.core import AdapterError, MethodAdapter


class P2CNetAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("P2CNet repository or weight missing")
        sys.path.insert(0, str(repo))
        from models.P2CNet import P2CNet

        self.device = torch.device("cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request)
        self.model = P2CNet().to(self.device)
        self.model.load_state_dict(torch.load(str(weight), map_location=self.device))
        self.model.eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        h, w = bgr.shape[:2]
        # Match the official TestDataset preprocessing exactly.
        target_h, target_w = 256, 256
        rgb = cv2.cvtColor(cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        lab = kornia.color.rgb_to_lab(rgb_tensor)
        tensor = torch.cat((lab[:, :1] / 100.0, lab[:, 1:] / 127.0), dim=1)
        with torch.no_grad():
            output = self.model(tensor)["lab_rgb"]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = output[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        bgr_out = cv2.cvtColor((rgb_out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (w, h), interpolation=cv2.INTER_CUBIC)
