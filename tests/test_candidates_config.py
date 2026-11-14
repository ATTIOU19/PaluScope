import json
from dataclasses import fields
from pathlib import Path

from src.candidates.extract import CandidateParams

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "candidates_v1.json"


def test_la_config_figee_existe_et_se_charge():
    assert CONFIG.exists(), "configs/candidates_v1.json est absent : lance scripts/freeze_candidates.py"
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    params = CandidateParams(**cfg)                       # échoue si une clé est inconnue
    assert set(cfg) == {f.name for f in fields(CandidateParams)}
    assert params.min_contrast > 0
    assert 0 < params.min_area < params.max_area
    assert 0.0 <= params.min_circularity <= 1.0