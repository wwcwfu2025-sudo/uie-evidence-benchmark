from __future__ import annotations

import sys
import traceback
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
REVISION_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts import benchmark_method_library_deep_8datasets as benchmark


METHOD_IDS = [
    "D001",
    "D002",
    "D003",
    "D005",
    "D006",
    "D007",
    "D009",
    "D010",
    "D011",
    "D024",
    "D031",
    "D035",
    "D036",
    "D044",
    "D045",
    "D046",
    "D047",
    "D048",
    "D049",
    "D050",
]


def main() -> None:
    benchmark.MANIFEST = REVISION_ROOT / "manifests" / "expanded_quality_manifest.csv"
    benchmark.ASSETS = REVISION_ROOT / "raw_results" / "deep"
    benchmark.RUN_ROOT = REVISION_ROOT / "benchmark_runs" / "expanded_deep"
    benchmark.STATE_DIR = benchmark.RUN_ROOT / "state"
    benchmark.VISUAL_DIR = benchmark.RUN_ROOT / "visual_samples"
    benchmark.main(METHOD_IDS, long_side=512, resume=True, save_samples=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

