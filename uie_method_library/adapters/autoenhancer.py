from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from uie_method_library.core import AdapterError, MethodAdapter


class AutoEnhancerAdapter(MethodAdapter):
    architecture = [2, 5, 5, 4, 0, 8, 8, 1, 2, 5, 5, 1]

    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("AutoEnhancer repository or checkpoint missing")
        for name in list(sys.modules):
            if name == "underwater_model" or name.startswith("underwater_model."):
                sys.modules.pop(name, None)
        sys.path.insert(0, str(repo))
        try:
            # The released base_SPOS.py imports underwater_model.vit.ViT, but
            # the file is absent from the author repository and ViT is never
            # referenced by Base construction or forward. Supply only the dead
            # import symbol; no computation or checkpoint key depends on it.
            vit_shim = types.ModuleType("underwater_model.vit")

            class UnusedViT(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    raise AdapterError("Unexpected use of missing released ViT dependency")

            vit_shim.ViT = UnusedViT
            sys.modules["underwater_model.vit"] = vit_shim
            module = importlib.import_module("underwater_model.model_SPOS")
        finally:
            if sys.path and sys.path[0] == str(repo):
                sys.path.pop(0)
        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.model = module.Water(dim=48).to(self.device)
        self.model.load_state_dict(torch.load(str(weight), map_location=self.device), strict=True)
        self.model.eval()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        height, width = bgr.shape[:2]
        rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lab_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_u8.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device) / 255.0
        lab = torch.from_numpy(np.ascontiguousarray(lab_u8.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device) / 255.0
        pad_h = (-height) % 8
        pad_w = (-width) % 8
        if pad_h or pad_w:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="reflect")
            lab = F.pad(lab, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            output = self.model(rgb, lab, self.architecture)[0][:, :, :height, :width]
            output = output[0].detach().cpu().clamp(0, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "official_architecture_choice": self.architecture,
                "inputs": "RGB and OpenCV LAB, both [0,1]",
                "geometry_protocol": "official reflect-pad to multiple of 8 and exact crop",
                "compatibility_note": "released repository omits unused underwater_model.vit; adapter supplies dead-import symbol only",
            }
        )
        return data
