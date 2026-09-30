#!/usr/bin/env python3
"""Validate a reviewed CVAT export and build an evaluation-ready OOD label set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


WIDTH = 512
HEIGHT = 500


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_tag_label(labels: ET.Element, name: str, color: str) -> None:
    for label in labels.findall("label"):
        if label.findtext("name") == name:
            type_node = label.find("type")
            if type_node is None:
                type_node = ET.SubElement(label, "type")
            type_node.text = "tag"
            return
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = name
    ET.SubElement(label, "color").text = color
    ET.SubElement(label, "type").text = "tag"
    ET.SubElement(label, "attributes")


def parse_points(value: str) -> np.ndarray:
    points = np.asarray(
        [tuple(map(float, item.split(","))) for item in value.split(";")],
        dtype=np.float32,
    )
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError(f"invalid polyline points: {value!r}")
    return points


def rasterize(image: ET.Element) -> np.ndarray:
    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for line in image.findall("polyline"):
        vertices = np.rint(parse_points(line.attrib["points"])).astype(np.int32)
        cv2.polylines(
            canvas,
            [vertices.reshape((-1, 1, 2))],
            False,
            color=255,
            thickness=1,
            lineType=cv2.LINE_8,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    with (args.dataset / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    expected = [row["image_name"] for row in manifest]
    if len(expected) != 20 or len(set(expected)) != 20:
        raise ValueError(f"expected 20 unique manifest images, found {len(expected)}")

    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if Path(name).name == "annotations.xml"]
        if len(members) != 1:
            raise ValueError(f"expected exactly one annotations.xml, found {members}")
        payload = archive.read(members[0])
    root = ET.fromstring(payload)
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise ValueError("input is not a CVAT for images 1.1 document")

    images = root.findall("image")
    actual = [Path(image.attrib["name"]).name for image in images]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(
            f"CVAT/manifest mismatch: missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    image_by_name = {Path(image.attrib["name"]).name: image for image in images}

    ordered: list[ET.Element] = []
    source_counts: Counter[str] = Counter()
    per_image: list[dict[str, object]] = []
    for image_id, name in enumerate(expected):
        image = image_by_name[name]
        if (int(image.attrib["width"]), int(image.attrib["height"])) != (WIDTH, HEIGHT):
            raise ValueError(f"unexpected geometry for {name}: {image.attrib}")
        unexpected = [shape.tag for shape in image if shape.tag not in {"polyline", "tag"}]
        if unexpected:
            raise ValueError(f"unsupported CVAT shapes for {name}: {unexpected}")
        lines = image.findall("polyline")
        if any(line.attrib.get("label") != "breaker" for line in lines):
            raise ValueError(f"unexpected polyline label in {name}")
        vertex_count = 0
        for line in lines:
            points = parse_points(line.attrib["points"])
            if np.any(points[:, 0] < 0) or np.any(points[:, 0] > WIDTH):
                raise ValueError(f"out-of-bounds x coordinate in {name}")
            if np.any(points[:, 1] < 0) or np.any(points[:, 1] > HEIGHT):
                raise ValueError(f"out-of-bounds y coordinate in {name}")
            vertex_count += len(points)
            source_counts[line.attrib.get("source", "unspecified")] += 1
        image.attrib["id"] = str(image_id)
        image.attrib["name"] = name
        for old_tag in list(image.findall("tag")):
            image.remove(old_tag)
        ET.SubElement(image, "tag", label="done", source="manual")
        ET.SubElement(image, "tag", label="annotated", source="manual")
        ordered.append(image)
        per_image.append({
            "image_name": name,
            "breaker_polylines": len(lines),
            "polyline_vertices": vertex_count,
        })

    for image in images:
        root.remove(image)
    root.extend(ordered)
    labels = root.find("./meta/job/labels")
    if labels is None:
        labels = root.find("./meta/task/labels")
    if labels is None:
        meta = root.find("meta")
        if meta is None:
            meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "size").text = str(len(ordered))
        labels = ET.SubElement(task, "labels")
    ensure_tag_label(labels, "done", "#1e6144")
    ensure_tag_label(labels, "annotated", "#4c78a8")

    canonical_path = output_dir / "annotations_human_canonical.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(canonical_path, encoding="utf-8", xml_declaration=True)

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    mapping_path = output_dir / "image_mask_mapping.csv"
    mapping_rows: list[dict[str, object]] = []
    mask_hashes: dict[str, str] = {}
    for image, item in zip(ordered, per_image):
        name = str(item["image_name"])
        source_image = args.dataset / "images" / name
        with Image.open(source_image) as loaded:
            if loaded.size != (WIDTH, HEIGHT):
                raise ValueError(f"unexpected source image geometry: {source_image} {loaded.size}")
        mask_name = "mask_" + name
        mask_path = masks_dir / mask_name
        Image.fromarray(rasterize(image), mode="L").save(mask_path)
        with Image.open(mask_path) as mask:
            values = set(np.unique(np.asarray(mask)).tolist())
            if mask.size != (WIDTH, HEIGHT) or not values.issubset({0, 255}):
                raise ValueError(f"invalid rasterized mask: {mask_path}")
        mask_hashes[mask_name] = sha256(mask_path)
        mapping_rows.append({
            "image_name": name,
            "mask_name": mask_name,
            "num_polylines": item["breaker_polylines"],
            "width": WIDTH,
            "height": HEIGHT,
            "split": "ood_test",
        })
    with mapping_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    cell_width, cell_height, title_height = 256, 250, 18
    overlay = Image.new(
        "RGB", (5 * cell_width, 4 * (cell_height + title_height)), (255, 255, 255),
    )
    overlay_draw = ImageDraw.Draw(overlay)
    for index, row in enumerate(mapping_rows):
        name = str(row["image_name"])
        image = Image.open(args.dataset / "images" / name).convert("RGB").resize(
            (cell_width, cell_height), Image.Resampling.BILINEAR,
        )
        mask = Image.open(masks_dir / str(row["mask_name"])).resize(
            (cell_width, cell_height), Image.Resampling.NEAREST,
        )
        image_array = np.asarray(image).copy()
        image_array[np.asarray(mask) > 0] = (255, 20, 40)
        x = (index % 5) * cell_width
        y = (index // 5) * (cell_height + title_height)
        overlay.paste(Image.fromarray(image_array), (x, y + title_height))
        overlay_draw.text((x + 4, y + 3), f"{name} - reviewed", fill=(0, 0, 0))
    overlay_path = output_dir / "annotation_overlay.png"
    overlay.save(overlay_path)

    total_lines = sum(int(item["breaker_polylines"]) for item in per_image)
    total_vertices = sum(int(item["polyline_vertices"]) for item in per_image)
    status = {
        "status": "reviewed/canonical",
        "annotation_status": "done",
        "label_status": "annotated",
        "images": len(ordered),
        "breaker_polylines": total_lines,
        "polyline_vertices": total_vertices,
        "polyline_source_counts": dict(sorted(source_counts.items())),
        "source_archive": args.archive.name,
        "source_archive_sha256": sha256(args.archive),
        "source_annotations_sha256": sha256_bytes(payload),
        "canonical_annotations_sha256": sha256(canonical_path),
        "mapping_sha256": sha256(mapping_path),
        "annotation_overlay_sha256": sha256(overlay_path),
        "mask_sha256": mask_hashes,
        "job_id": root.findtext("./meta/job/id"),
        "job_updated_at": root.findtext("./meta/job/updated"),
        "export_dumped_at": root.findtext("./meta/dumped"),
        "orientation": "stored model/CVAT orientation; no flip",
        "rasterization": "rounded CVAT vertices; OpenCV 1-pixel LINE_8 polylines",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "images_detail": per_image,
    }
    status_path = output_dir / "annotation_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(
        f"normalized {len(ordered)} reviewed images, {total_lines} polylines, "
        f"and {total_vertices} vertices"
    )
    print("added image tags: done, annotated")
    print(f"wrote {len(mapping_rows)} binary masks")


if __name__ == "__main__":
    main()
