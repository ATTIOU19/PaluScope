from src.data.labeling import label_candidates


def test_dedans_vaut_1_dehors_vaut_0():
    candidats = [
        {"id": 0, "bbox": (27, 27, 6, 6)},    # centre (30, 30) : dans le carré
        {"id": 1, "bbox": (97, 97, 6, 6)},    # centre (100, 100) : dehors
    ]
    boites = [{"box_id": "A", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}]
    resultat = label_candidates(candidats, boites)
    assert resultat[0]["label"] == 1
    assert resultat[0]["match_box_id"] == "A"
    assert resultat[1]["label"] == 0
    assert resultat[1]["match_box_id"] is None


def test_bord_compte_comme_dedans():
    candidats = [{"id": 0, "bbox": (47, 27, 6, 6)}]   # centre (50, 30) : pile sur le bord droit
    boites = [{"box_id": "A", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}]
    assert label_candidates(candidats, boites)[0]["label"] == 1


def test_plusieurs_boites_on_garde_la_plus_proche():
    candidats = [{"id": 0, "bbox": (27, 27, 6, 6)}]   # centre (30, 30)
    boites = [
        {"box_id": "loin", "xmin": 0, "ymin": 0, "xmax": 40, "ymax": 40},      # centre (20, 20)
        {"box_id": "proche", "xmin": 20, "ymin": 20, "xmax": 42, "ymax": 42},  # centre (31, 31)
    ]
    assert label_candidates(candidats, boites)[0]["match_box_id"] == "proche"


def test_les_entrees_ne_sont_pas_modifiees():
    candidats = [{"id": 0, "bbox": (27, 27, 6, 6)}]
    boites = [{"box_id": "A", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}]
    label_candidates(candidats, boites)
    assert "label" not in candidats[0]