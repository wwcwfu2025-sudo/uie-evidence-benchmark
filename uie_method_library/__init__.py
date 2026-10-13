"""Unified, provenance-aware underwater image enhancement method library."""

import sys
from pathlib import Path

_LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".method_deps"
if _LOCAL_DEPS.exists() and str(_LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DEPS))

__version__ = "0.1.0"
