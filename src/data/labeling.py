"""Étiquetage des candidats : un candidat est-il sur un vrai parasite ?"""


def centre(bbox):
    """Renvoie le centre (cx, cy) d'une boîte (x, y, largeur, hauteur)."""
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def est_dans_la_boite(cx, cy, boite):
    """Vrai si le point (cx, cy) est dans la boîte, bords compris."""
    return (boite["xmin"] <= cx <= boite["xmax"]
            and boite["ymin"] <= cy <= boite["ymax"])


def label_candidates(candidates, boxes):
    """Ajoute 'label' et 'match_box_id' à une COPIE de chaque candidat."""
    resultats = []
    for cand in candidates:
        cx, cy = centre(cand["bbox"])
        meilleure_cle = None
        meilleur_id = None
        for boite in boxes:
            if est_dans_la_boite(cx, cy, boite):
                bx = (boite["xmin"] + boite["xmax"]) / 2
                by = (boite["ymin"] + boite["ymax"]) / 2
                distance = (cx - bx) ** 2 + (cy - by) ** 2
                cle = (distance, boite["box_id"])
                if meilleure_cle is None or cle < meilleure_cle:
                    meilleure_cle = cle
                    meilleur_id = boite["box_id"]
        copie = dict(cand)
        copie["label"] = 0 if meilleur_id is None else 1
        copie["match_box_id"] = meilleur_id
        resultats.append(copie)
    return resultats