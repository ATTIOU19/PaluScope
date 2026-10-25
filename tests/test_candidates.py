import cv2
import numpy as np
import pytest

from src.candidates.evaluate import clip_boxes, match_candidates
from src.candidates.extract import CandidateParams, extract_candidates


def make_image(seed=0):
    """Image synthétique : fond rosé bruité + 3 petits points sombres (parasites),
    1 grosse tache bleue (cellule blanche) et 1 trait fin (débris)."""
    rng = np.random.default_rng(seed)
    img = np.full((200, 200, 3), (215, 170, 165), dtype=np.uint8)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    for cx, cy in [(40, 40), (120, 60), (60, 150)]:
        cv2.circle(img, (cx, cy), 4, (70, 30, 90), -1)          # petit point sombre
    cv2.circle(img, (160, 150), 18, (40, 20, 110), -1)           # grosse tache
    cv2.line(img, (100, 120), (150, 122), (70, 30, 90), 1)       # trait fin
    return img


def test_finds_the_three_small_dots_only():
    cands = extract_candidates(make_image())
    centers = sorted((c["bbox"][0] + c["bbox"][2] / 2, c["bbox"][1] + c["bbox"][3] / 2) for c in cands)
    assert len(cands) == 3
    for (cx, cy), (ex, ey) in zip(centers, [(40, 40), (60, 150), (120, 60)]):
        assert abs(cx - ex) <= 3 and abs(cy - ey) <= 3


def test_output_format_matches_conventions():
    for c in extract_candidates(make_image()):
        assert set(c) == {"id", "bbox", "area", "perimeter", "circularity"}
        assert len(c["bbox"]) == 4
        assert 0.0 <= c["circularity"] <= 1.0


def test_blank_image_gives_no_candidates():
    blank = np.full((100, 100, 3), (200, 160, 160), dtype=np.uint8)
    assert extract_candidates(blank) == []


def test_rejects_wrong_input():
    with pytest.raises(ValueError):
        extract_candidates(np.zeros((10, 10), dtype=np.uint8))


def test_matching_and_clipping():
    cands = extract_candidates(make_image())
    gt = [{"xmin": 30, "ymin": 30, "xmax": 50, "ymax": 50},
          {"xmin": 100, "ymin": 5, "xmax": 140, "ymax": 25}]   # 2e boîte : rien dedans
    r = match_candidates(cands, gt)
    assert r["n_found"] == 1 and r["n_gt"] == 2
    assert clip_boxes([{"xmin": -19, "ymin": 0, "xmax": 30, "ymax": 770}], 750, 750)[0]["ymax"] == 750
