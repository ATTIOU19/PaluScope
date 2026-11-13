import cv2
import numpy as np
import pytest

from src.candidates.extract import CandidateParams
from src.data.candidate_eval import evaluate_samples, load_params, save_params


def _image():
    img = np.full((200, 200, 3), 200, np.uint8)
    for cx, cy in [(40, 40), (120, 60), (60, 150)]:
        cv2.circle(img, (cx, cy), 4, (60, 60, 60), -1)  # taches sombres ~50 px²
    return img


def _gt():
    boxes = [(40, 40), (120, 60), (60, 150), (170, 170)]  # la 4e boîte n'a aucune tache
    return [{"xmin": x - 20.0, "ymin": y - 20.0, "xmax": x + 20.0, "ymax": y + 20.0} for x, y in boxes]


def test_evaluate_samples_recall_and_counts():
    res = evaluate_samples([(_image(), _gt())], CandidateParams())
    assert res["n_gt"] == 4 and res["n_found"] == 3
    assert res["recall"] == pytest.approx(0.75)
    assert res["n_candidates"] == 3 and res["candidate_precision"] == pytest.approx(1.0)


def test_params_json_roundtrip_and_unknown_field(tmp_path):
    p = CandidateParams(channel="green", blackhat_ksize=21, min_contrast=15.0)
    save_params(p, tmp_path / "c.json", {"note": "test"})
    assert load_params(tmp_path / "c.json") == p
    bad = tmp_path / "bad.json"
    bad.write_text('{"params": {"inconnu": 1}}', encoding="utf-8")
    with pytest.raises(TypeError):
        load_params(bad)