"""Lecture minimale des annotations Pascal VOC du dataset Makerere.

Les annotations ne sont JAMAIS modifiées : on lit, on compte, on clippe en mémoire.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def read_voc_boxes(xml_path: str | Path) -> list[tuple[str, float, float, float, float]]:
    """Retourne [(label, xmin, ymin, xmax, ymax), ...] en float, tel que dans le XML."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        label = (obj.findtext("name") or "").strip()
        xmin, ymin, xmax, ymax = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((label, xmin, ymin, xmax, ymax))
    return boxes


def summarize_boxes(boxes, width: int, height: int) -> dict:
    """Compte les boîtes après clipping aux dimensions de l'image.

    n_boxes            : boîtes conservées (aire > 0 après clipping)
    n_boxes_clipped    : parmi elles, celles qui dépassaient de l'image
    n_boxes_degenerate : boîtes d'aire nulle après clipping (écartées du compte)
    """
    kept = clipped = degenerate = 0
    for _, x0, y0, x1, y1 in boxes:
        cx0, cy0 = max(x0, 0.0), max(y0, 0.0)
        cx1, cy1 = min(x1, float(width)), min(y1, float(height))
        if cx1 <= cx0 or cy1 <= cy0:
            degenerate += 1
            continue
        kept += 1
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            clipped += 1
    return {
        "n_boxes_raw": len(boxes),
        "n_boxes": kept,
        "n_boxes_clipped": clipped,
        "n_boxes_degenerate": degenerate,
    }