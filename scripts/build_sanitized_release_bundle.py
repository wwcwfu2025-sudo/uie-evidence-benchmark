from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
BUNDLE = ROOT / "release_bundle"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def windows_name(value: str) -> str:
    return PureWindowsPath(value).name if value else ""


def public_manifest(source: Path, target: Path, frontier: bool = False) -> None:
    rows: list[dict[str, object]] = []
    for item in read_csv(source):
        reference = Path(item["reference"]) if item.get("reference") else None
        row: dict[str, object] = {
            "dataset": item["dataset"],
            "sample_key": item["sample_key"],
            "cluster_id": item["cluster_id"],
            "input_file_name": windows_name(item["input"]),
            "reference_file_name": windows_name(item.get("reference", "")),
            "has_reference": item["has_reference"],
            "selection_score": item["selection_score"],
            "input_sha256": item["input_sha256"],
            "reference_sha256": sha256(reference) if reference and reference.exists() else "",
        }
        if frontier:
            row["frontier_selection_score"] = item["frontier_selection_score"]
        rows.append(row)
    write_csv(target, rows)


def public_provenance() -> None:
    fields = [
        "method_id", "display_name", "algorithm_group", "family", "role", "status",
        "remote_repository", "git_commit", "source_archive_file", "source_archive_sha256",
        "weight_file", "weight_sha256", "license_file", "license_sha256", "provenance_note",
    ]
    rows: list[dict[str, object]] = []
    for item in read_csv(ROOT / "methods" / "method_provenance.csv"):
        rows.append(
            {
                **{
                    key: item.get(key, "")
                    for key in fields
                    if key not in {"source_archive_file", "weight_file", "license_file"}
                },
                "source_archive_file": windows_name(item.get("source_archive", "")),
                "weight_file": windows_name(item.get("weight_path", "")),
                "license_file": windows_name(item.get("license_file", "")),
            }
        )
    write_csv(BUNDLE / "methods" / "method_provenance_public.csv", rows, fields)


def sanitize_per_image(source: Path, target: Path) -> None:
    drop = {"input", "reference", "output", "weight", "weight_path", "local_repository"}
    rows: list[dict[str, object]] = []
    for item in read_csv(source):
        cleaned = {key: value for key, value in item.items() if key not in drop}
        # Successful benchmark rows have an empty error field. Avoid leaking a
        # local path if a failed run embeds one in its message.
        if cleaned.get("error"):
            cleaned["error"] = "See private run log; local paths removed from public bundle."
        rows.append(cleaned)
    write_csv(target, rows)


def copy_results() -> None:
    result_sources = [
        ROOT / "raw_results" / "expanded_classical_per_image.csv",
        ROOT / "raw_results" / "expanded_classical_summary.csv",
        ROOT / "raw_results" / "deep" / "72_method_library_deep28_8datasets_per_image.csv",
        ROOT / "raw_results" / "deep" / "73_method_library_deep28_8datasets_summary.csv",
    ]
    result_sources += sorted((ROOT / "raw_results" / "frontier").glob("*.csv"))
    result_sources += sorted((ROOT / "raw_results" / "detection_urpc2020").glob("*.csv"))
    for source in result_sources:
        if not source.exists():
            continue
        name = source.name
        target = BUNDLE / "raw_results" / name
        if "per_image" in name or "72_method_library" in name:
            sanitize_per_image(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    for source in sorted((ROOT / "analysis").glob("*.csv")):
        target = BUNDLE / "analysis" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for directory in ("frontier_track", "manuscript_tables"):
        source_dir = ROOT / "analysis" / directory
        if source_dir.exists():
            shutil.copytree(source_dir, BUNDLE / "analysis" / directory, dirs_exist_ok=True)


def copy_code() -> None:
    target = BUNDLE / "scripts"
    target.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "scripts").glob("*.py")):
        shutil.copy2(source, target / source.name)
    shared = WORKSPACE / "scripts" / "benchmark_20_methods_8_datasets.py"
    if shared.exists():
        shutil.copy2(shared, target / shared.name)
    library = WORKSPACE / "uie_method_library"
    if library.exists():
        shutil.copytree(library, BUNDLE / "uie_method_library", dirs_exist_ok=True)


def write_hash_index() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or path.name == "FILE_SHA256SUMS.csv":
            continue
        rows.append(
            {
                "relative_path": path.relative_to(BUNDLE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(BUNDLE / "FILE_SHA256SUMS.csv", rows)


def main() -> None:
    public_manifest(
        ROOT / "manifests" / "expanded_quality_manifest.csv",
        BUNDLE / "manifests" / "expanded_quality_manifest_public.csv",
    )
    public_manifest(
        ROOT / "manifests" / "frontier_stratified_manifest.csv",
        BUNDLE / "manifests" / "frontier_stratified_manifest_public.csv",
        frontier=True,
    )
    public_provenance()
    copy_results()
    copy_code()
    metadata = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "source_workspace": "local private workspace; absolute paths removed",
        "third_party_data_or_weights_included": False,
        "main_images": 1253,
        "paired_images": 800,
        "frontier_images": 287,
        "main_items_including_raw": 41,
    }
    (BUNDLE / "bundle_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_hash_index()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
