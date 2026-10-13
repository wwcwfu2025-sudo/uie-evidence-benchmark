from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
REVISION_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = WORKSPACE / "uie_method_library" / "registry.csv"
SELECTION = REVISION_ROOT / "expanded_benchmark_methods.csv"
OUT_DIR = REVISION_ROOT / "methods"
OUT_CSV = OUT_DIR / "method_provenance.csv"
OUT_META = OUT_DIR / "method_provenance_metadata.json"
CLASSICAL_SOURCE = WORKSPACE / "scripts" / "benchmark_20_methods_8_datasets.py"
GIT = Path(os.environ.get("GIT_EXECUTABLE", shutil.which("git") or "git"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    if not (repo / ".git").exists():
        return ""
    try:
        result = subprocess.run(
            [str(GIT), "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def license_file(repo: Path) -> Path | None:
    if not repo.is_dir():
        return None
    candidates = sorted(
        [
            path
            for path in repo.iterdir()
            if path.is_file()
            and (path.name.lower().startswith("license") or path.name.lower().startswith("copying"))
        ],
        key=lambda path: path.name.lower(),
    )
    return candidates[0] if candidates else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = {row["method_id"]: row for row in read_csv(REGISTRY)}
    rows: list[dict[str, str]] = []
    classical_hash = sha256(CLASSICAL_SOURCE)
    for selected in read_csv(SELECTION):
        method_id = selected["method_id"]
        row = {
            **selected,
            "local_repository": "",
            "remote_repository": "",
            "git_commit": "",
            "source_archive": "",
            "source_archive_sha256": "",
            "weight_path": "",
            "weight_sha256": "",
            "license_file": "",
            "license_sha256": "",
            "provenance_note": "",
        }
        if method_id.startswith("C"):
            row.update(
                {
                    "local_repository": str(WORKSPACE),
                    "source_archive": str(CLASSICAL_SOURCE),
                    "source_archive_sha256": classical_hash,
                    "provenance_note": "Standardized protocol implementation; not claimed as original author code.",
                }
            )
        elif method_id in registry:
            spec = registry[method_id]
            repo = Path(spec["repository"]) if spec.get("repository") else Path()
            weight = Path(spec["weights"]) if spec.get("weights") else Path()
            license_path = license_file(repo) if spec.get("repository") else None
            row.update(
                {
                    "local_repository": str(repo) if spec.get("repository") else "",
                    "remote_repository": git_value(repo, "remote", "get-url", "origin"),
                    "git_commit": git_value(repo, "rev-parse", "HEAD"),
                    "weight_path": str(weight) if spec.get("weights") else "",
                    "weight_sha256": sha256(weight) if spec.get("weights") else "",
                    "license_file": str(license_path) if license_path else "",
                    "license_sha256": sha256(license_path) if license_path else "",
                    "provenance_note": spec.get("authority", ""),
                }
            )
        elif method_id == "F001":
            repo = REVISION_ROOT / "external_methods" / "W2WDiff-main"
            archive = REVISION_ROOT / "external_methods" / "W2WDiff-main.zip"
            license_path = license_file(repo)
            row.update(
                {
                    "local_repository": str(repo),
                    "remote_repository": "https://github.com/c-pupil/W2WDiff",
                    "source_archive": str(archive),
                    "source_archive_sha256": sha256(archive),
                    "license_file": str(license_path) if license_path else "",
                    "license_sha256": sha256(license_path) if license_path else "",
                    "provenance_note": "Official repository archive; checkpoint acquisition recorded separately.",
                }
            )
        elif method_id == "F002":
            repo = REVISION_ROOT / "external_methods" / "HRMamba-main"
            archive = REVISION_ROOT / "external_methods" / "HRMamba-main.zip"
            license_path = license_file(repo)
            row.update(
                {
                    "local_repository": str(repo),
                    "remote_repository": "https://github.com/YeFan-web/HRMamba",
                    "source_archive": str(archive),
                    "source_archive_sha256": sha256(archive),
                    "license_file": str(license_path) if license_path else "",
                    "license_sha256": sha256(license_path) if license_path else "",
                    "provenance_note": "Official repository archive; package verification pending completion.",
                }
            )
        rows.append(row)

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": len(rows),
        "weight_hash_count": sum(bool(row["weight_sha256"]) for row in rows),
        "git_commit_count": sum(bool(row["git_commit"]) for row in rows),
        "license_file_count": sum(bool(row["license_file"]) for row in rows),
        "warning": (
            "Blank commit or license fields mean that the local copy did not expose the information; "
            "they do not imply that a repository has no license or version history."
        ),
    }
    OUT_META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_CSV)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

