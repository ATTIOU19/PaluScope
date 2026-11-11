"""Tests de la Phase 2 sur une table SYNTHÉTIQUE respectant le contrat de la Phase 1."""
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cv import grouped_cv, summarize_folds, undersample_indices
from src.evaluation.dataset import feature_columns, load_table
from src.evaluation.metrics import choose_threshold, evaluate, object_level
from src.models.zoo import default_models


def make_table(n_images=120, per_image=15, seed=0):
    """Table de candidats synthétique : 2 images par groupe, signal faible sur les features."""
    rng = np.random.default_rng(seed)
    rows, imgs = [], []
    groups = np.arange(n_images) // 2
    group_split = {g: s for g, s in zip(range(n_images // 2),
                   rng.choice(["train", "val", "test"], size=n_images // 2, p=[0.7, 0.15, 0.15]))}
    for i in range(n_images):
        img, g = f"img{i:03d}", int(groups[i])
        n_box = int(rng.integers(0, 5))
        boxes = [f"{img}#{k:03d}" for k in range(n_box)]
        imgs.append({"image_id": img, "n_boxes": n_box, "split": group_split[g]})
        for c in range(per_image):
            pos = c < n_box - 1 or (n_box > 0 and c == 0)          # une boîte peut rester sans candidat
            match = boxes[c] if (pos and c < n_box) else None
            lab = int(match is not None)
            rows.append({
                "image_id": img, "candidate_id": c, "split": group_split[g], "group_id": g,
                "label": lab, "match_box_id": match,
                "geo_circularity": rng.normal(0.8 if lab else 0.6, 0.15),
                "col_contrast": rng.normal(30 if lab else 20, 8),
                "grd_mean": rng.normal(5 + 2 * lab, 2),
                "tex_lbp0": rng.normal(0, 1),
                "hog_0": rng.normal(0, 1),
                "qual_sharp": rng.normal(10, 1),
                "norm_mode": "none", "candidates_cfg": "v1", "features_cfg": "v1",
            })
    return pd.DataFrame(rows), pd.DataFrame(imgs)


@pytest.fixture(scope="module")
def table():
    return make_table()


def test_evaluate_matches_hand_computation():
    y = [1, 1, 0, 0, 0, 1]
    s = [0.9, 0.4, 0.6, 0.1, 0.2, 0.8]
    m = evaluate(y, s, thr=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["prevalence"] == pytest.approx(0.5)
    assert 0 <= m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1


def test_evaluate_uses_none_not_nan_when_undefined():
    m = evaluate([0, 0, 0], [0.1, 0.2, 0.3], thr=0.9)          # aucun positif, aucune prédiction positive
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["roc_auc"] is None and m["pr_auc"] is None        # une seule classe
    assert m["specificity"] == 1.0


def test_choose_threshold_reaches_target_recall():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = y * 1.0 + rng.normal(0, 0.7, 500)
    th = choose_threshold(y, s, target_recall=0.9)
    assert evaluate(y, s, th["thr"])["recall"] >= 0.9
    with pytest.raises(ValueError):
        choose_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_undersample_keeps_all_positives_and_is_deterministic():
    y = np.array([1] * 10 + [0] * 500)
    a, b = undersample_indices(y, 5, seed=1), undersample_indices(y, 5, seed=1)
    assert (a == b).all()
    assert y[a].sum() == 10 and (y[a] == 0).sum() == 50
    assert len(undersample_indices(y, None)) == len(y)


def test_grouped_cv_has_no_group_leakage_and_is_reproducible(table):
    df, _ = table
    tr = df[df["split"] == "train"]
    cols = feature_columns(df, "F3")
    X, y, g = tr[cols].to_numpy(float), tr["label"].to_numpy(int), tr["group_id"].to_numpy()
    est = default_models(class_weight="balanced")["logreg"]
    r1 = grouped_cv(est, X, y, g, n_splits=5)             # lève RuntimeError en cas de fuite
    r2 = grouped_cv(est, X, y, g, n_splits=5)
    assert np.allclose(r1["oof_scores"], r2["oof_scores"])
    assert not np.isnan(r1["oof_scores"]).any()
    summ = summarize_folds(r1["folds"])
    assert summ["roc_auc_mean"] > 0.6                      # le signal synthétique est détecté


def test_object_level_merges_candidates_per_box():
    cands = pd.DataFrame({
        "image_id": ["a", "a", "a", "b"],
        "match_box_id": ["a#0", "a#0", None, None],        # 2 candidats sur la même boîte + 1 faux positif
    })
    images = pd.DataFrame({"image_id": ["a", "b", "c"], "n_boxes": [2, 0, 1]})
    r = object_level(cands, [1, 1, 1, 0], thr=0.5, images=images)
    assert r["n_true_objects"] == 3 and r["n_found_objects"] == 1
    assert r["object_recall"] == pytest.approx(1 / 3)
    # image a : prédit 1 (boîte fusionnée) + 1 faux positif = 2 (vrai 2) ; b : 0 (vrai 0) ; c : 0 (vrai 1)
    assert r["count_mae"] == pytest.approx((0 + 0 + 1) / 3)


def test_load_table_rejects_leakage_and_nan(tmp_path, table):
    df, _ = table
    ok = tmp_path / "ok.csv"
    df.to_csv(ok, index=False)
    assert len(load_table(ok)) == len(df)

    leak = df.copy()
    g = leak["group_id"].iloc[0]
    leak.loc[leak.index[leak["group_id"] == g][0], "split"] = "test" if leak["split"].iloc[0] != "test" else "train"
    leak.to_csv(tmp_path / "leak.csv", index=False)
    with pytest.raises(ValueError, match="FUITE"):
        load_table(tmp_path / "leak.csv")

    nan = df.copy()
    nan.loc[0, "geo_circularity"] = np.nan
    nan.to_csv(tmp_path / "nan.csv", index=False)
    with pytest.raises(ValueError, match="NaN"):
        load_table(tmp_path / "nan.csv")


def test_feature_sets_and_qual_excluded_by_default(table):
    df, _ = table
    assert feature_columns(df, "F1") == ["geo_circularity"]
    assert "hog_0" not in feature_columns(df, "F3") and "hog_0" in feature_columns(df, "F4")
    assert "qual_sharp" not in feature_columns(df, "F4")
    assert "qual_sharp" in feature_columns(df, "F4", use_qual=True)


def test_script_end_to_end_and_test_is_used_only_once(tmp_path, table):
    from scripts.train_classical import main

    df, imgs = table
    ds, reg = tmp_path / "dataset.csv", tmp_path / "images.csv"
    df.to_csv(ds, index=False)
    imgs.to_csv(reg, index=False)
    out, mdir = tmp_path / "reports", tmp_path / "models"
    base = ["--dataset", str(ds), "--registry", str(reg), "--out", str(out), "--models-dir", str(mdir),
            "--feature-set", "F3", "--models", "logreg,tree"]

    assert main(base) == 0
    comp = pd.read_csv(out / "comparison_F3_class_weight.csv")
    assert {"dummy", "rule:geo_circularity", "logreg", "tree"} == set(comp["model"])
    assert not (out / "test_result.json").exists()                      # le test n'a pas été touché
    dummy = comp[comp["model"] == "dummy"].iloc[0]
    assert dummy["val_roc_auc"] == pytest.approx(0.5)                   # la baseline triviale n'apprend rien

    assert main(base + ["--final-test", "--final-model", "logreg"]) == 0
    meta = json.loads((mdir / "classical_v1.json").read_text())
    assert meta["split_version"] == "split_v1" and "threshold" in meta and meta["features"]
    assert (mdir / "classical_v1.joblib").exists()

    with pytest.raises(SystemExit):                                     # deuxième lecture du test refusée
        main(base + ["--final-test", "--final-model", "tree"])
