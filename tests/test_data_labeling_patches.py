import numpy as np

from src.data.labeling import label_candidates
from src.data.patches import PATCH_SIZE, extract_patch

BOXES = [
    {"xmin": 10.0, "ymin": 10.0, "xmax": 50.0, "ymax": 50.0},
    {"xmin": 40.0, "ymin": 40.0, "xmax": 90.0, "ymax": 90.0},  # chevauche la première
    {"xmin": 200.0, "ymin": 200.0, "xmax": 240.0, "ymax": 240.0},  # sans candidat
]


def _cand(i, x, y, w=10, h=10):
    return {"id": i, "bbox": (x, y, w, h), "area": 80.0, "perimeter": 35.0, "circularity": 0.8}


def test_label_inside_outside_and_original_untouched():
    cands = [_cand(0, 20, 20), _cand(1, 300, 300)]
    out = label_candidates(cands, BOXES)
    assert [c["label"] for c in out] == [1, 0]
    assert [c["match_box_id"] for c in out] == [0, -1]
    assert "label" not in cands[0]  # entrée non modifiée


def test_label_overlap_picks_nearest_box_center():
    # centre (46, 46), dans les boîtes 0 et 1 : d0 = 512 (centre 30,30) < d1 = 722 (centre 65,65)
    out = label_candidates([_cand(0, 41, 41)], BOXES)
    assert out[0]["match_box_id"] == 0


def test_label_shrink_tightens_the_box():
    near_edge = _cand(0, 7, 25)  # centre (12, 30) : dans la boîte 0, loin de son centre
    assert label_candidates([near_edge], BOXES)[0]["label"] == 1
    assert label_candidates([near_edge], BOXES, shrink=0.5)[0]["label"] == 0  # boîte resserrée : x dans [20, 40]


def test_patch_shape_dtype_and_centering():
    img = np.zeros((100, 100, 3), np.uint8)
    img[48:52, 48:52] = 255  # point lumineux au centre du candidat
    p = extract_patch(img, (45, 45, 10, 10))  # centre (50, 50)
    assert p.shape == (PATCH_SIZE, PATCH_SIZE, 3) and p.dtype == np.uint8
    ys, xs = np.where(p[..., 0] == 255)
    assert abs(ys.mean() - (PATCH_SIZE / 2 - 0.5)) < 1.5 and abs(xs.mean() - (PATCH_SIZE / 2 - 0.5)) < 1.5


def test_patch_at_border_reflects_and_keeps_shape():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    for bbox in [(0, 0, 10, 10), (90, 90, 10, 10), (95, 0, 5, 10)]:
        assert extract_patch(img, bbox).shape == (PATCH_SIZE, PATCH_SIZE, 3)