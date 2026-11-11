"""Registre des données PaluScope (Phase 1).

Construit deux tables versionnables à partir des images et des annotations :

- ``data/registry/images_registry.csv`` : une ligne par image ;
- ``data/registry/boxes.csv``           : une ligne par boîte annotée.

Conventions du projet :
- images lues en dimensions réelles (pas de balise <size> dans les XML) ;
- boîtes au format (x, y, w, h), flottantes ;
- les annotations brutes ne sont JAMAIS modifiées : on stocke les valeurs
  brutes (``raw_*``) ET les valeurs clippées aux dimensions de l'image
  (``x, y, w, h``) ;
- ``group_id`` et ``split`` sont laissés vides ici : ils sont remplis par
  ``src/data/duplicates.py`` et ``src/data/split.py``.

Annotations : Pascal VOC (<object><label>plasmodium</label><bndbox>...). On réutilise
``parse_boxes`` et ``clip_boxes`` de ``src.candidates.evaluate`` (Phase 0) pour rester
strictement cohérent avec les résultats déjà obtenus (7 628 boîtes, rappel des candidats).

Utilisation :
    python -m src.data.registry --images data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from src.candidates.evaluate import clip_boxes, parse_boxes

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = ROOT / "data" / "registry"
IMAGES_CSV = REGISTRY_DIR / "images_registry.csv"
BOXES_CSV = REGISTRY_DIR / "boxes.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL = "plasmodium"          # classe unique du dataset (balise <label> des XML)

IMAGE_COLUMNS = [
    "image_id", "path", "width", "height",
    "n_boxes", "n_boxes_clipped", "n_boxes_dropped",
    "image_unannotated", "has_xml", "read_error",
    "dataset_source", "device", "smear_type",
    "group_id", "split",
]
BOX_COLUMNS = [
    "box_id", "image_id", "label",
    "x", "y", "w", "h",                       # boîte clippée, format (x, y, w, h)
    "raw_xmin", "raw_ymin", "raw_xmax", "raw_ymax",
    "was_clipped",
]


# --------------------------------------------------------------------------
# Construction du registre
# --------------------------------------------------------------------------
def _find_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _index_xml(annotation_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Index des XML : par chemin relatif sans extension, et par nom de fichier."""
    by_rel: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = {}
    for p in sorted(annotation_dir.rglob("*.xml")):
        by_rel[p.relative_to(annotation_dir).with_suffix("").as_posix()] = p
        by_stem.setdefault(p.stem, []).append(p)
    return by_rel, by_stem


def _match_xml(image_rel: str, stem: str, by_rel: dict, by_stem: dict) -> Path | None:
    if image_rel in by_rel:
        return by_rel[image_rel]
    candidates = by_stem.get(stem, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"XML ambigus pour l'image « {stem} » : {candidates}")
    return None


def build_registry(
    image_dir: str | Path,
    annotation_dir: str | Path | None = None,
    dataset_source: str = "makerere_plasmodium_phonecamera",
    device: str = "unknown",
    smear_type: str = "thick",
    data_root: str | Path = ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit (table des images, table des boîtes). Déterministe (tri fixe)."""
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir) if annotation_dir else image_dir
    data_root = Path(data_root)
    by_rel, by_stem = _index_xml(annotation_dir)

    image_rows: list[dict] = []
    box_rows: list[dict] = []

    for img_path in _find_images(image_dir):
        rel = img_path.relative_to(image_dir).with_suffix("").as_posix()
        image_id = rel.replace("/", "__")
        try:
            display_path = img_path.relative_to(data_root).as_posix()
        except ValueError:
            display_path = img_path.as_posix()

        width = height = None
        read_error = False
        try:
            with Image.open(img_path) as im:          # lecture de l'en-tête seulement
                width, height = im.size
        except Exception:
            read_error = True

        xml_path = _match_xml(rel, img_path.stem, by_rel, by_stem)
        raw_boxes = parse_boxes(xml_path) if xml_path else []

        n_clipped = n_dropped = n_kept = 0
        if not read_error:
            for k, raw in enumerate(raw_boxes):
                clipped = clip_boxes([raw], width, height)   # [] si la boîte est entièrement hors image
                if not clipped:
                    n_dropped += 1
                    continue
                c = clipped[0]
                was_clipped = any(c[key] != raw[key] for key in ("xmin", "ymin", "xmax", "ymax"))
                n_clipped += int(was_clipped)
                n_kept += 1
                box_rows.append({
                    "box_id": f"{image_id}#{k:03d}",
                    "image_id": image_id,
                    "label": LABEL,
                    "x": c["xmin"], "y": c["ymin"],
                    "w": c["xmax"] - c["xmin"], "h": c["ymax"] - c["ymin"],
                    "raw_xmin": raw["xmin"], "raw_ymin": raw["ymin"],
                    "raw_xmax": raw["xmax"], "raw_ymax": raw["ymax"],
                    "was_clipped": was_clipped,
                })

        image_rows.append({
            "image_id": image_id,
            "path": display_path,
            "width": width,
            "height": height,
            "n_boxes": n_kept,
            "n_boxes_clipped": n_clipped,
            "n_boxes_dropped": n_dropped,
            "image_unannotated": n_kept == 0,
            "has_xml": xml_path is not None,
            "read_error": read_error,
            "dataset_source": dataset_source,
            "device": device,
            "smear_type": smear_type,
            "group_id": pd.NA,
            "split": pd.NA,
        })

    images_df = pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)
    boxes_df = pd.DataFrame(box_rows, columns=BOX_COLUMNS)
    return images_df, boxes_df


# --------------------------------------------------------------------------
# Sauvegarde / chargement (contrat d'interface : load_registry())
# --------------------------------------------------------------------------
def save_registry(images_df: pd.DataFrame, boxes_df: pd.DataFrame,
                  images_csv: Path = IMAGES_CSV, boxes_csv: Path = BOXES_CSV) -> None:
    images_csv.parent.mkdir(parents=True, exist_ok=True)
    images_df.sort_values("image_id").to_csv(images_csv, index=False)
    boxes_df.sort_values("box_id").to_csv(boxes_csv, index=False)


def load_registry(path: str | Path | None = None) -> pd.DataFrame:
    """Une ligne par image : image_id, path, dimensions, group_id, split,
    nombre de boîtes, drapeaux qualité."""
    return pd.read_csv(path or IMAGES_CSV)


def load_boxes(path: str | Path | None = None, image_id: str | None = None) -> pd.DataFrame:
    """Boîtes clippées (x, y, w, h). Filtre optionnel sur une image."""
    df = pd.read_csv(path or BOXES_CSV)
    return df if image_id is None else df[df["image_id"] == image_id].reset_index(drop=True)


# --------------------------------------------------------------------------
# Contrôles rapides (réutilisables dans tests/test_registry.py)
# --------------------------------------------------------------------------
def validate_registry(images_df: pd.DataFrame, boxes_df: pd.DataFrame) -> dict:
    """Renvoie un dictionnaire de compteurs ; toutes les clés « problème » doivent valoir 0."""
    merged = boxes_df.merge(images_df[["image_id", "width", "height"]], on="image_id")
    outside = (
        (merged["x"] < 0) | (merged["y"] < 0)
        | (merged["x"] + merged["w"] > merged["width"] + 1e-9)
        | (merged["y"] + merged["h"] > merged["height"] + 1e-9)
    ).sum()
    return {
        "n_images": len(images_df),
        "n_boxes": len(boxes_df),
        "n_unannotated": int(images_df["image_unannotated"].sum()),
        "n_images_with_clipped_boxes": int((images_df["n_boxes_clipped"] > 0).sum()),
        "PROBLEME_image_id_dupliques": int(images_df["image_id"].duplicated().sum()),
        "PROBLEME_box_id_dupliques": int(boxes_df["box_id"].duplicated().sum()),
        "PROBLEME_images_illisibles": int(images_df["read_error"].sum()),
        "PROBLEME_boites_hors_image": int(outside),
        "PROBLEME_boites_orphelines": int((~boxes_df["image_id"].isin(images_df["image_id"])).sum()),
    }


# --------------------------------------------------------------------------
# Ligne de commande
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Construit le registre des images et des boîtes.")
    ap.add_argument("--images", required=True, help="Dossier des images")
    ap.add_argument("--annotations", default=None, help="Dossier des XML (défaut : dossier des images)")
    ap.add_argument("--source", default="makerere_plasmodium_phonecamera")
    ap.add_argument("--device", default="unknown")
    ap.add_argument("--smear-type", default="thick")
    args = ap.parse_args()

    images_df, boxes_df = build_registry(
        args.images, args.annotations, args.source, args.device, args.smear_type
    )
    save_registry(images_df, boxes_df)
    for key, value in validate_registry(images_df, boxes_df).items():
        print(f"{key:35s} {value}")
    print(f"\nÉcrit : {IMAGES_CSV}\n        {BOXES_CSV}")


if __name__ == "__main__":
    main()
