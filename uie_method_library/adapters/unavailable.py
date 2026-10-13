from __future__ import annotations

import numpy as np

from uie_method_library.core import AdapterError, MethodAdapter


class UnavailableAdapter(MethodAdapter):
    def load(self) -> None:
        raise AdapterError("Method is not executable yet: status=%s; %s" % (self.spec.status, self.spec.note))

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        raise AssertionError("unreachable")
