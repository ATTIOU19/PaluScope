"""Test de fumée de l'application Streamlit : lance l'app sans navigateur et vérifie qu'elle ne plante pas."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(not list((ROOT / "data" / "samples").glob("*.jpg")), reason="data/samples est vide")


def test_app_starts_and_shows_quality_metrics():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Blood Smear Quality Analyzer"
    assert len(at.metric) >= 5          # métriques de qualité


def test_app_runs_on_several_samples():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    for i in range(min(5, len(at.selectbox[0].options))):
        at.selectbox[0].select_index(i).run()
        assert not at.exception

def test_app_shows_candidate_metrics_with_annotations():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Régions candidates" in labels
    assert "Parasites annotés retrouvés" in labels


def test_app_accepts_changed_detection_params():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.selectbox[1].select("green").run()      # canal vert
    assert not at.exception        