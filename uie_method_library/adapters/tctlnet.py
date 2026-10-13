from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from uie_method_library.core import AdapterError, MethodAdapter


class TCTLNetAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        if not repo.exists() or not weight.exists():
            raise AdapterError("TCTL-Net repository or checkpoint missing")
        for name in list(sys.modules):
            if name == "model" or name.startswith("model.") or name == "utils" or name.startswith("utils."):
                sys.modules.pop(name, None)
        sys.path.insert(0, str(repo))
        try:
            # thop is imported only for the standalone __main__ FLOP example
            # in the released model file; inference never calls these symbols.
            if "thop" not in sys.modules:
                thop_shim = types.ModuleType("thop")
                thop_shim.profile = lambda *args, **kwargs: (_ for _ in ()).throw(
                    AdapterError("Unexpected THOP profiling call during inference")
                )
                thop_shim.clever_format = lambda values, fmt: values
                sys.modules["thop"] = thop_shim
            model_module = importlib.import_module("model.TCTLNet_model")
            util_module = importlib.import_module("utils.util")
        finally:
            if sys.path and sys.path[0] == str(repo):
                sys.path.pop(0)
        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        self.model = model_module.TCTLNet(output_size=(256, 256), init_weights=False).to(self.device)
        self.model.load_state_dict(torch.load(str(weight), map_location=self.device), strict=True)
        self.model.eval()
        self.get_gray = util_module.get_gray
        self.trans_np = util_module.trans_np
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        height, width = bgr.shape[:2]
        raw = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
        normalized = raw.astype(np.float32) / 255.0
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        gray_l, gray_a, gray_b = self.get_gray(lab)
        mean, std = self.trans_np(lab)
        lab_chw = torch.from_numpy(np.ascontiguousarray(lab.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)
        raw_lab = lab_chw.clone()
        mean_tensor = torch.from_numpy(mean).float().view(1, 3, 1, 1).to(self.device)
        std_tensor = torch.from_numpy(std).float().view(1, 3, 1, 1).to(self.device)
        with torch.no_grad():
            output = self.model(
                lab_chw,
                lab_chw,
                raw_lab,
                gray_l.unsqueeze(0).to(self.device),
                gray_a.unsqueeze(0).to(self.device),
                gray_b.unsqueeze(0).to(self.device),
                mean_tensor,
                std_tensor,
            )[3][0]
            output = output.detach().cpu().clamp(0, 255)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        bgr_out = output.permute(1, 2, 0).numpy().round().astype(np.uint8)
        return cv2.resize(bgr_out, (width, height), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "official_input": "256x256 OpenCV float BGR-to-LAB",
                "output_order": "BGR [0,255], as released cv2.imwrite path",
                "protocol": "released TestDataset preprocessing reproduced in memory",
                "compatibility_note": "thop standalone profiling import shimmed; inference graph unchanged",
            }
        )
        return data
