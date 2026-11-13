"""Construction d'un mini-dataset synthétique et contrôles anti-fuite / qualité."""
import cv2
import numpy as np
import pandas as pd
import pytest

import src.features.quality as quality
from src.candidates.extract import CandidateParams
from src.features.builder import FAMILIES, build_dataset, feature_columns
from src.features.checks import check_dataset, dataset_summary

XML = "<annotation>{objs}</annotation>"
OBJ = "<object><name>plasmodium</name><bndbox><xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax><ymax>{}</ymax></bndbox></object>"
NO_QUAL = tuple(f for f in FAMILIES if f != "qual")


@pytest.fixture
def mini(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    for i, split in enumerate(splits):
        img = np.clip(np.array([200, 150, 170]) + rng.normal(0, 3, (160, 160, 3)), 0, 255).astype(np.uint8)
        pts = [(int(rng.integers(30, 130)), int(rng.integers(30, 130))) for _ in range(3)]
        for cx, cy in pts:
            cv2.circle(img, (cx, cy), 4, (60, 40, 80), -1)
        name = f"img{i:02d}"
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        objs = "".join(OBJ.format(cx - 20, cy - 20, cx + 20, cy + 20) for cx, cy in pts[:2]) if i % 4 else ""
        (tmp_path / f"{name}.xml").write_text(XML.format(objs=objs), encoding="utf-8")
        rows.append({"image_id": name, "image_path": f"{name}.jpg", "xml_path": f"{name}.xml", "width": 160,
                     "height": 160, "group_id": f"G{i:04d}", "split": split, "image_unannotated": i % 4 == 0,
                     "dataset_source": "synthetic", "device": "synthetic", "smear_type": "thick"})
    return pd.DataFrame(rows), tmp_path


def test_build_dataset_columns_labels_and_checks(mini):
    reg, root = mini
    df, patches = build_dataset(reg, CandidateParams(), families=NO_QUAL, root=root)
    assert len(df) > 0 and len(patches) == len(df) and patches.shape[1:] == (48, 48, 3)
    assert {"geo", "col", "grd", "tex", "hog"} <= {c.split("_")[0] for c in df.columns}
    assert len(feature_columns(df, ("hog",))) == 324
    assert df["label"].isin([0, 1]).all() and df["label"].sum() > 0 and (df["label"] == 0).sum() >= 0
    assert (df.loc[df["image_unannotated"], "label"] == 0).all()
    assert check_dataset(df, reg) == []
    s = dataset_summary(df)
    assert s["n_candidates"] == len(df) and s["n_nan_features"] == 0


def test_build_is_deterministic(mini):
    reg, root = mini
    a, pa = build_dataset(reg, CandidateParams(), families=NO_QUAL, root=root)
    b, pb = build_dataset(reg.sample(frac=1.0, random_state=3), CandidateParams(), families=NO_QUAL, root=root)
    pd.testing.assert_frame_equal(a, b)
    assert np.array_equal(pa, pb)


def test_quality_family_adds_qual_columns(mini, monkeypatch):
    reg, root = mini
    monkeypatch.setattr(quality, "_analyze", lambda img: {k: 1.0 for k in quality.QUAL_KEYS})
    df, _ = build_dataset(reg, CandidateParams(), families=("geo", "qual"), root=root)
    assert len(feature_columns(df, ("qual",))) == 5 and check_dataset(df, reg) == []


def test_checks_detect_leak_nan_and_inconsistency(mini):
    reg, root = mini
    df, _ = build_dataset(reg, CandidateParams(), families=("geo",), root=root)

    leak = df.copy()
    g = leak["group_id"].iloc[0]
    leak.loc[leak.index[leak["group_id"] == g][0], "split"] = "val" if leak["split"].iloc[0] != "val" else "test"
    leak = pd.concat([leak, df[df["group_id"] == g].assign(split="test", candidate_id=lambda d: d["candidate_id"] + "x")])
    assert any("group_id" in p for p in check_dataset(leak))

    nan = df.copy()
    nan.loc[nan.index[0], "geo_area"] = np.nan
    assert any("NaN" in p for p in check_dataset(nan))

    bad = df.copy()
    bad.loc[bad.index[0], "label"] = 1 - bad.loc[bad.index[0], "label"]
    assert any("incohérents" in p for p in check_dataset(bad))