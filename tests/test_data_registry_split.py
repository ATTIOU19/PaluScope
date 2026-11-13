"""Tests du registre, des groupes de quasi-doublons et de la séparation (données synthétiques)."""
import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.dedup import group_near_duplicates, hamming_matrix, phash
from src.data.registry import build_registry, load_registry
from src.data.split import SPLITS, assert_no_leakage, make_split
from src.data.voc import summarize_boxes

W = H = 128
XML = """<annotation><filename>{name}.jpg</filename>{objs}</annotation>"""
OBJ = "<object><name>plasmodium</name><bndbox><xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax><ymax>{}</ymax></bndbox></object>"


def _smooth_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.random((H, W, 3)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 9)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _write(d, name, img, boxes):
    cv2.imwrite(str(d / f"{name}.jpg"), img)
    objs = "".join(OBJ.format(*b) for b in boxes)
    (d / f"{name}.xml").write_text(XML.format(name=name, objs=objs), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    """24 images distinctes ; 6 d'entre elles ont un quasi-doublon (décalage + bruit)."""
    d = tmp_path / "raw"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(24):
        n = 0 if i % 5 == 0 else int(rng.integers(1, 30))
        boxes = [(10 + k, 10 + k, 50 + k, 50 + k) for k in range(n)]
        if i == 1:
            boxes.append((100.5, 100.5, 140.0, 140.0))  # dépasse de l'image 128x128
        _write(d, f"img{i:03d}", _smooth_image(i), boxes)
    for i in range(6):
        base = _smooth_image(i)
        dup = np.roll(base, 2, axis=1)
        dup = np.clip(dup.astype(int) + rng.integers(-3, 4, dup.shape), 0, 255).astype(np.uint8)
        _write(d, f"img{i:03d}_dup", dup, [(10, 10, 50, 50)])
    return tmp_path


def test_summarize_boxes_clips_and_counts():
    boxes = [("p", 10, 10, 50, 50), ("p", 100.5, 100.5, 140, 140), ("p", 200, 200, 210, 210)]
    s = summarize_boxes(boxes, 128, 128)
    assert s == {"n_boxes_raw": 3, "n_boxes": 2, "n_boxes_clipped": 1, "n_boxes_degenerate": 1}


def test_phash_robust_to_small_shift_and_noise():
    a = _smooth_image(3)
    b = np.roll(a, 2, axis=1)
    c = _smooth_image(4)
    ga, gb, gc = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (a, b, c))
    bits = np.vstack([phash(ga), phash(gb), phash(gc)])
    D = hamming_matrix(bits)
    assert D[0, 1] <= 8 < D[0, 2]


def test_registry_groups_duplicates_and_flags(dataset):
    reg, _ = build_registry(dataset / "raw", root=dataset, threshold=8)
    assert len(reg) == 30
    assert set(reg["width"]) == {W} and set(reg["height"]) == {H}
    for i in range(6):  # chaque original est groupé avec son quasi-doublon
        g = reg.set_index("image_id").loc[[f"img{i:03d}", f"img{i:03d}_dup"], "group_id"]
        assert g.nunique() == 1
    assert reg[~reg["image_id"].str.endswith("_dup")]["group_id"].nunique() == 24
    assert reg.set_index("image_id").loc["img001", "n_boxes_clipped"] == 1
    assert reg.set_index("image_id").loc["img000", "image_unannotated"]
    assert not reg["image_path"].str.contains("\\\\", regex=False).any()  # chemins POSIX


def test_split_no_leakage_deterministic_and_proportions(dataset):
    reg, _ = build_registry(dataset / "raw", root=dataset)
    s1 = make_split(reg, seed=42)
    s2 = make_split(reg, seed=42)
    pd.testing.assert_frame_equal(s1, s2)
    assert_no_leakage(s1)
    assert set(s1["split"]) == set(SPLITS)
    shares = s1["split"].value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.15
    # un original et son doublon sont dans la même partition
    sp = s1.set_index("image_id")["split"]
    assert all(sp[f"img{i:03d}"] == sp[f"img{i:03d}_dup"] for i in range(6))


def test_assert_no_leakage_detects_problem():
    bad = pd.DataFrame({"image_id": ["a", "b"], "group_id": ["G0", "G0"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_load_registry_roundtrip(dataset):
    reg, _ = build_registry(dataset / "raw", root=dataset)
    sp = make_split(reg)
    reg.to_csv(dataset / "images.csv", index=False)
    sp.to_csv(dataset / "split.csv", index=False)
    out = load_registry(dataset / "images.csv", dataset / "split.csv")
    assert len(out) == len(reg) and "split" in out.columns