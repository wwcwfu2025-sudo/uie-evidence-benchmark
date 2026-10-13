from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "revision_20260819_table_condensation" / "20260820" / "20260820.docx"
OUT_JSON = ROOT / "analysis" / "source_docx_block_inventory.json"
OUT_TEXT = ROOT / "analysis" / "source_docx_section5_6_dump.txt"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
W = "{%s}" % NS["w"]


def text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.iter(W + "t"))


def style(element: ET.Element) -> str:
    node = element.find("w:pPr/w:pStyle", NS)
    return "" if node is None else node.attrib.get(W + "val", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--json", type=Path, default=OUT_JSON)
    parser.add_argument("--text", type=Path, default=OUT_TEXT)
    args = parser.parse_args()
    source = args.source.resolve()
    out_json = args.json.resolve()
    out_text = args.text.resolve()

    with zipfile.ZipFile(source) as archive:
        bad = archive.testzip()
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", NS)
    assert body is not None
    blocks: list[dict[str, object]] = []
    section_lines: list[str] = []
    in_target = False
    for index, element in enumerate(list(body)):
        kind = element.tag.split("}")[-1]
        if kind == "p":
            value = " ".join(text(element).split())
            record: dict[str, object] = {
                "body_index": index,
                "type": "paragraph",
                "style": style(element),
                "text": value,
                "drawings": len(element.findall(".//w:drawing", NS)),
                "formulas": len(element.findall(".//m:oMath", NS)),
            }
            if value == "V. METHOD COMPARISON AND COMPREHENSIVE DISCUSSION":
                in_target = True
            if in_target:
                section_lines.append(f"[P{index:04d}][{record['style']}] {value}")
            if value == "REFERENCES":
                in_target = False
        elif kind == "tbl":
            rows = [
                [" ".join(text(cell).split()) for cell in row.findall("w:tc", NS)]
                for row in element.findall("w:tr", NS)
            ]
            record = {
                "body_index": index,
                "type": "table",
                "rows": len(rows),
                "columns": max((len(row) for row in rows), default=0),
                "data": rows,
                "drawings": len(element.findall(".//w:drawing", NS)),
            }
            if in_target:
                section_lines.append(f"[T{index:04d}] rows={len(rows)} cols={record['columns']}")
                for row in rows:
                    section_lines.append("    | " + " | ".join(row))
        else:
            record = {"body_index": index, "type": kind, "text": " ".join(text(element).split())}
        blocks.append(record)

    references = [
        int(match.group(1))
        for item in blocks
        if item["type"] == "paragraph"
        and (match := re.match(r"^\[(\d+)\]", str(item.get("text", ""))))
    ]
    payload = {
        "source": str(source),
        "zip_test": bad,
        "body_blocks": len(blocks),
        "tables": sum(item["type"] == "table" for item in blocks),
        "drawings": len(root.findall(".//w:drawing", NS)),
        "formulas": len(root.findall(".//m:oMath", NS)),
        "reference_count": len(references),
        "reference_min": min(references) if references else None,
        "reference_max": max(references) if references else None,
        "blocks": blocks,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_text.write_text("\n".join(section_lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "blocks"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
