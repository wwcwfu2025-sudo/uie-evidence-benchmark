from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from uie_method_library.core import AdapterError, MethodAdapter


def _flatten(mapping: Mapping, prefix: str = "") -> dict:
    output = {}
    for key, value in mapping.items():
        name = "%s_%s" % (prefix, key) if prefix else str(key)
        if isinstance(value, Mapping):
            output.update(_flatten(value, name))
        else:
            output[name] = value
    return output


def _clear_generic_repo_modules() -> None:
    # Several released UIE repositories use top-level packages named ``utils``
    # and ``model``. Clear only those generic names before importing LUIEO so a
    # prior adapter cannot redirect its absolute imports to another repository.
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils.") or name == "model" or name.startswith("model."):
            del sys.modules[name]


class LuieoAdapter(MethodAdapter):
    def load(self) -> None:
        repo = Path(self.spec.repository)
        weight = Path(self.spec.weights)
        config = repo / "mobilevitv3_small_multiserver.yaml"
        if not repo.exists() or not weight.exists() or not config.exists():
            raise AdapterError("LUIEO repository, checkpoint, or YAML configuration missing")
        _clear_generic_repo_modules()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from model.mobilevit_model import mobile_backbone, mobile_head, mobile_model, mobile_neck

        with config.open("r", encoding="utf-8") as handle:
            opts = argparse.Namespace(**_flatten(yaml.safe_load(handle)))
        self.device = torch.device(
            "cuda" if self.device_request == "auto" and torch.cuda.is_available() else self.device_request
        )
        setattr(opts, "dev_device", self.device)
        setattr(opts, "dev_device_id", None)
        setattr(opts, "ddp_use_distributed", False)
        setattr(opts, "model_pretrained", str(weight))
        self.model = mobile_model(mobile_backbone(opts), mobile_neck(opts), mobile_head(opts)).to(self.device)
        checkpoint = torch.load(str(weight), map_location=self.device)
        state = checkpoint["model_state_dict"]
        try:
            self.model.load_state_dict(state, strict=True)
            self.ignored_checkpoint_keys = []
        except RuntimeError:
            current = self.model.state_dict()
            compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
            missing = [key for key in current if key not in compatible and "head.Detect_head" not in key]
            if missing:
                raise AdapterError("LUIEO enhancement path is missing checkpoint keys: %s" % missing[:5])
            current.update(compatible)
            self.model.load_state_dict(current, strict=True)
            self.ignored_checkpoint_keys = [key for key in state if key not in compatible]
        # Released predict.py deliberately uses training mode for joint inference.
        self.model.train()
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        original_h, original_w = bgr.shape[:2]
        scale = min(1.0, 512.0 / max(original_h, original_w))
        work_w = max(32, (round(original_w * scale) // 32) * 32)
        work_h = max(32, (round(original_h * scale) // 32) * 32)
        resized = cv2.resize(
            bgr,
            (work_w, work_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        original = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
        flipped = torch.flip(original, dims=[2])
        tensor = torch.stack([original, flipped], dim=0).to(self.device)
        with torch.no_grad():
            outs, _, _ = self.model(tensor, "object_detection")
            enhanced = outs["img"].clone()
            enhanced[1] = torch.flip(enhanced[1], dims=[-1])
            output = enhanced.mean(0).clamp(0, 1).detach().cpu()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rgb_out = (output.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr_out, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    def metadata(self):
        data = super().metadata()
        data.update(
            {
                "joint_task": "object_detection forward path; outs['img'] selected",
                "tta": "original plus horizontal flip, averaged as released",
                "geometry_fix": "omits released predict.py's extra final hflip, which reverses spatial correspondence",
                "model_mode": "train (matches released predict.py)",
                "size_protocol": "aspect-preserving max long side 512; multiple of 32",
                "ignored_checkpoint_keys": getattr(self, "ignored_checkpoint_keys", []),
            }
        )
        return data
