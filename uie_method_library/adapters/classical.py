from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from scripts import benchmark_20_methods_8_datasets as baseline
from uie_method_library.core import AdapterError, MethodAdapter


FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw": lambda image: image.copy(),
    "gray_world": baseline.gray_world,
    "max_rgb": baseline.max_rgb,
    "shades_of_gray": baseline.shades_of_gray,
    "gray_edge": baseline.gray_edge,
    "white_patch": baseline.white_patch_percentile,
    "rgb_he": baseline.rgb_hist_equalization,
    "ycrcb_he": baseline.ycrcb_hist_equalization,
    "lab_clahe": baseline.lab_clahe,
    "hsv_clahe": baseline.hsv_clahe,
    "gamma07": baseline.fixed_gamma,
    "adaptive_gamma": baseline.adaptive_gamma,
    "unsharp": baseline.unsharp_mask,
    "homomorphic": baseline.homomorphic_filter,
    "ssr": baseline.ssr,
    "msr": baseline.msr,
    "msrcr": baseline.msrcr,
    "dcp": baseline.dcp,
    "udcp": baseline.udcp,
    "rcp": baseline.rcp,
    "fusion_inspired": baseline.fusion_ancuti_inspired,
}


class ClassicalAdapter(MethodAdapter):
    def load(self) -> None:
        key = self.spec.adapter.split(":", 1)[1]
        if key not in FUNCTIONS:
            raise AdapterError("Unknown classical adapter function: %s" % key)
        self.function = FUNCTIONS[key]
        self.loaded = True

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        return self.function(bgr)
