from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    name: str
    family: str
    is_deep: bool
    adapter: str
    authority: str
    status: str
    repository: str
    weights: str
    note: str


class AdapterError(RuntimeError):
    pass


class MethodAdapter(ABC):
    def __init__(self, spec: MethodSpec, device: str = "auto") -> None:
        self.spec = spec
        self.device_request = device
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.load()

    def metadata(self) -> Dict[str, Any]:
        return {
            "method_id": self.spec.method_id,
            "name": self.spec.name,
            "family": self.spec.family,
            "is_deep": self.spec.is_deep,
            "authority": self.spec.authority,
            "status": self.spec.status,
            "repository": self.spec.repository,
            "weights": self.spec.weights,
            "adapter": type(self).__name__,
        }


def read_bgr(path: Path) -> np.ndarray:
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise AdapterError("Cannot decode input image: %s" % path)
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, np.clip(image, 0, 255).astype(np.uint8))
    if not ok:
        raise AdapterError("Cannot encode output image: %s" % path)
    path.write_bytes(encoded.tobytes())
