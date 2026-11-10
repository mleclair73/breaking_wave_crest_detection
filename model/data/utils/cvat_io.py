"""CVAT 1.1 XML read/write and polyline-to-mask rasterization.

Coordinates are image `(x, y)` pixels; masks are `(height, width)`.
Recovered from the pre-cleanup notes/independent_cvat_bootstrap/scripts/cvat_io.py.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class Polyline:
    points: tuple[Point, ...]
    label: str = "breaker"
    source: str = "manual"
    occluded: bool = False
    z_order: int = 0
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("a polyline needs at least two points")


@dataclass(frozen=True)
class CvatImage:
    image_id: int
    name: str
    width: int
    height: int
    polylines: tuple[Polyline, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


def _format_points(points: Sequence[Point]) -> str:
    return ";".join(f"{x:.6f},{y:.6f}" for x, y in points)


def parse_points(value: str) -> tuple[Point, ...]:
    try:
        points = tuple(tuple(map(float, point.split(","))) for point in value.split(";") if point)
    except ValueError as exc:
        raise ValueError(f"invalid CVAT points: {value!r}") from exc
    if any(len(point) != 2 for point in points):
        raise ValueError(f"invalid CVAT points: {value!r}")
    return points  # type: ignore[return-value]


def parse_image(image: ET.Element, label: str = "breaker") -> CvatImage:
    """Parse one CVAT `<image>` element, keeping only polylines with `label`."""
    metadata = {tag.attrib["label"]: (tag.text or "") for tag in image.findall("tag") if "label" in tag.attrib}
    lines: list[Polyline] = []
    for element in image.findall("polyline"):
        if element.attrib.get("label") != label:
            continue
        attrs = {a.attrib["name"]: (a.text or "") for a in element.findall("attribute")}
        lines.append(Polyline(
            points=parse_points(element.attrib.get("points", "")), label=label,
            source=element.attrib.get("source", "manual"), occluded=element.attrib.get("occluded", "0") == "1",
            z_order=int(element.attrib.get("z_order", "0")), attributes=attrs,
        ))
    return CvatImage(int(image.attrib["id"]), image.attrib["name"], int(image.attrib["width"]),
                     int(image.attrib["height"]), tuple(lines), metadata)


def read_cvat_xml(path: Path, label: str = "breaker") -> list[CvatImage]:
    """Read every image and its `label` polylines from a CVAT export."""
    root = ET.parse(path).getroot()
    if root.tag != "annotations":
        raise ValueError(f"{path} is not a CVAT annotations document")
    return [parse_image(image, label) for image in root.findall("image")]


def write_cvat_xml(images: Iterable[CvatImage], path: Path, label: str = "breaker", *,
                   include_metadata: bool = True) -> None:
    """Write a CVAT 1.1 image task containing only `label` polylines.

    CVAT treats image `<tag>` values as task labels; set `include_metadata`
    false for task-upload XML unless every metadata key is a configured tag label.
    """
    images = list(images)
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "size").text = str(len(images))
    label_node = ET.SubElement(ET.SubElement(task, "labels"), "label")
    ET.SubElement(label_node, "name").text = label
    ET.SubElement(label_node, "type").text = "any"
    ET.SubElement(label_node, "attributes")
    for image in images:
        node = ET.SubElement(root, "image", id=str(image.image_id), name=image.name,
                             width=str(image.width), height=str(image.height))
        if include_metadata:
            for key, value in sorted(image.metadata.items()):
                ET.SubElement(node, "tag", label=key).text = str(value)
        for line in image.polylines:
            if line.label != label:
                raise ValueError(f"unexpected label {line.label!r}; expected {label!r}")
            element = ET.SubElement(node, "polyline", label=line.label, source=line.source,
                                    occluded="1" if line.occluded else "0", z_order=str(line.z_order),
                                    points=_format_points(line.points))
            for key, value in sorted(line.attributes.items()):
                ET.SubElement(element, "attribute", name=key).text = str(value)
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def rasterize_polylines(polylines: Iterable[Polyline | Sequence[Point]], shape: tuple[int, int],
                        width: float = 1) -> np.ndarray:
    """Rasterize centreline strokes (vertices rounded, 8-connected) to a boolean mask."""
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must be positive (height, width)")
    if width <= 0:
        raise ValueError("width must be positive")
    canvas = np.zeros(shape, dtype=np.uint8)
    thickness = max(1, int(round(width)))
    for line in polylines:
        points = line.points if isinstance(line, Polyline) else line
        if len(points) < 2:
            raise ValueError("a polyline needs at least two points")
        vertices = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [vertices], False, color=1, thickness=thickness, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def to_mask_image(mask: np.ndarray) -> np.ndarray:
    """Boolean mask -> uint8 0/255, the stored PNG convention."""
    return mask.astype(np.uint8) * 255


_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _trace_skeleton(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Split a one-pixel skeleton into paths between endpoints/junctions, as (row, col)."""
    pixels = set(zip(*np.nonzero(skeleton)))

    def neighbours(p: tuple[int, int]) -> list[tuple[int, int]]:
        return [(p[0] + dr, p[1] + dc) for dr, dc in _NEIGHBOURS if (p[0] + dr, p[1] + dc) in pixels]

    nodes = {p for p in pixels if len(neighbours(p)) != 2}
    visited: set[frozenset] = set()
    paths: list[list[tuple[int, int]]] = []

    def walk(start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
        path, prev, cur = [start], start, step
        visited.add(frozenset((start, step)))
        while True:
            path.append(cur)
            if cur in nodes:
                return path
            nxt = [n for n in neighbours(cur) if n != prev and frozenset((cur, n)) not in visited]
            if not nxt:
                return path
            visited.add(frozenset((cur, nxt[0])))
            prev, cur = cur, nxt[0]

    for node in sorted(nodes):
        for n in neighbours(node):
            if frozenset((node, n)) not in visited:
                paths.append(walk(node, n))
    for p in sorted(pixels):  # closed loops have no endpoint/junction
        for n in neighbours(p):
            if frozenset((p, n)) not in visited:
                paths.append(walk(p, n))
    paths += [[p] for p in nodes if not neighbours(p)]  # isolated pixels
    return paths


def mask_to_polylines(mask: np.ndarray, *, min_size: int = 20, epsilon: float = 1.0,
                      label: str = "breaker", source: str = "auto") -> tuple[Polyline, ...]:
    """Vectorize a binary crest mask: drop small components, skeletonize, trace, RDP-simplify."""
    from skimage.morphology import skeletonize

    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), connectivity=8)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_size)
    mask = np.isin(labels, keep[keep > 0])  # drop 8-connected components < min_size px
    lines = []
    for path in _trace_skeleton(skeletonize(mask)):
        if len(path) < 2:
            continue
        xy = np.asarray([(c, r) for r, c in path], dtype=np.int32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(xy, epsilon, False).reshape(-1, 2)
        if len(simplified) >= 2:
            lines.append(Polyline(tuple((float(x), float(y)) for x, y in simplified), label=label, source=source))
    return tuple(lines)
