from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from .core import MethodSpec


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "registry.csv"


def load_registry(path: Path = REGISTRY_PATH) -> List[MethodSpec]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        MethodSpec(
            method_id=row["method_id"],
            name=row["name"],
            family=row["family"],
            is_deep=row["is_deep"].lower() == "true",
            adapter=row["adapter"],
            authority=row["authority"],
            status=row["status"],
            repository=row["repository"],
            weights=row["weights"],
            note=row["note"],
        )
        for row in rows
    ]


def registry_by_id(path: Path = REGISTRY_PATH) -> Dict[str, MethodSpec]:
    return {item.method_id: item for item in load_registry(path)}
