from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class FunieGanAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("FUnIE-GAN repository or weight missing")
        sys.path.insert(0, str(repo))
        from nets.funiegan import GeneratorFunieGAN

        self.device = torch.device("cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request)
        self.model = GeneratorFunieGAN().to(self.device)
        state = torch.load(str(weight), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        dummy = torch.zeros((1, 3, 256, 256), device=self.device)
        with torch.no_grad():
            self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy((rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        array = output[0].detach().cpu().clamp(-1, 1)
        array = ((array + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
        return cv2.resize(cv2.cvtColor(array, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_CUBIC)
