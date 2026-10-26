"""Blood Smear Quality Analyzer : démo Streamlit de la Phase 0 de PaluScope.

Lancer depuis la racine du dépôt :   streamlit run app/streamlit_app.py

Chaîne : image → contrôle qualité → régions candidates. Vision classique uniquement (aucun CNN, aucun ML).
PaluScope est un prototype de recherche et de formation : il ne pose aucun diagnostic clinique.
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Le dossier du projet doit être dans le chemin d'import (Streamlit n'y met que le dossier app/).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidates.evaluate import clip_boxes, match_candidates, parse_boxes  # noqa: E402
from src.candidates.extract import CandidateParams, extract_candidates  # noqa: E402
from src.quality.analyzer import analyze_image  # noqa: E402

SAMPLES_DIR = ROOT / "data" / "samples"
BANNER = ROOT / "docs" / "assets" / "banner.svg"
REF_SHAPE = (750, 750)          # taille des images sur lesquelles les seuils ont été calibrés
ORANGE, GREEN = (255, 140, 0), (57, 255, 136)   # couleurs RGB : candidats, annotations
DEFAULTS = CandidateParams()

FLAG_LABELS = {
    "blurry": "Image floue",
    "low_contrast": "Contraste insuffisant",
    "too_dark": "Image trop sombre",
    "too_bright": "Image trop claire",
    "overexposed": "Image surexposée",
    "dark_borders": "Bords noirs (vignette) probables",
}

st.set_page_config(page_title="Blood Smear Quality Analyzer · PaluScope", page_icon="🔬", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem;max-width:1400px}</style>", unsafe_allow_html=True)


# ───────────────────────────── fonctions utilitaires ─────────────────────────────
def decode_image(data: bytes) -> np.ndarray:
    """Décode des octets (JPEG, PNG) en tableau RGB uint8 à 3 canaux."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Fichier illisible : choisissez une image JPEG ou PNG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def banner_html() -> str:
    """Bannière du dépôt, intégrée en base64 (rien à servir en plus). Vide si le fichier est absent."""
    if not BANNER.exists():
        return ""
    b64 = base64.b64encode(BANNER.read_bytes()).decode()
    return ('<img alt="PaluScope : microscopie intelligente du paludisme" '
            f'src="data:image/svg+xml;base64,{b64}" style="width:100%;border-radius:14px;display:block">')


@st.cache_data(show_spinner="Analyse de la qualité…")
def run_quality(data: bytes) -> dict:
    return analyze_image(decode_image(data))


@st.cache_data(show_spinner="Recherche des candidats…")
def run_candidates(data: bytes, params: CandidateParams) -> list:
    return extract_candidates(decode_image(data), params)


def rgb_histogram(image: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({name: np.bincount(image[..., i].ravel(), minlength=256)
                         for i, name in enumerate(("Rouge", "Vert", "Bleu"))})


def candidates_table(candidates: list) -> pd.DataFrame:
    rows = [{"id": c["id"], "x": c["bbox"][0], "y": c["bbox"][1], "largeur": c["bbox"][2], "hauteur": c["bbox"][3],
             "aire (px²)": c["area"], "périmètre (px)": c["perimeter"], "circularité": c["circularity"]}
            for c in candidates]
    return pd.DataFrame(rows).round(3)


def draw_overlay(image, candidates, annotations, show_ids, pad=6):
    """Superpose les candidats (orange, avec une marge pour rester visibles) et les annotations (vert)."""
    out = image.copy()
    for b in annotations:
        cv2.rectangle(out, (round(b["xmin"]), round(b["ymin"])), (round(b["xmax"]), round(b["ymax"])), GREEN, 1)
    for c in candidates:
        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x - pad, y - pad), (x + w + pad, y + h + pad), ORANGE, 1)
        if show_ids:
            cv2.putText(out, str(c["id"]), (x - pad, max(y - pad - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1,
                        cv2.LINE_AA)
    return out


def png_bytes(image_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


def reset_params():
    """Remet les réglages de détection aux valeurs par défaut du Membre 3."""
    st.session_state.update(p_channel=DEFAULTS.channel, p_blackhat=DEFAULTS.blackhat_ksize,
                            p_contrast=int(DEFAULTS.min_contrast),
                            p_area=(int(DEFAULTS.min_area), int(DEFAULTS.max_area)),
                            p_circ=DEFAULTS.min_circularity)


# ───────────────────────────── en-tête ─────────────────────────────
if banner_html():
    st.markdown(banner_html(), unsafe_allow_html=True)
st.title("Blood Smear Quality Analyzer")
st.caption("Démo Phase 0 · Image → contrôle qualité → régions candidates · vision classique uniquement "
           "(aucun réseau de neurones).")

# ───────────────────────────── barre latérale ─────────────────────────────
with st.sidebar:
    st.subheader("1 · Image")
    source = st.radio("Source", ["Échantillon du dépôt", "Importer une image"], label_visibility="collapsed")
    xml_path = None
    if source == "Échantillon du dépôt":
        samples = sorted(SAMPLES_DIR.glob("*.jpg"))
        if not samples:
            st.error("Le dossier data/samples est vide : importez une image.")
            st.stop()
        choice = st.selectbox("Image d'exemple", samples, format_func=lambda p: p.stem.replace("plasmodium-", ""))
        data = choice.read_bytes()
        xml_path = choice.with_suffix(".xml")
    else:
        upload = st.file_uploader("Frottis (JPEG ou PNG)", type=["jpg", "jpeg", "png"])
        if upload is None:
            st.info("Importez une image pour lancer l'analyse.")
            st.stop()
        data = upload.getvalue()

    has_annotations = xml_path is not None and xml_path.exists()

    st.subheader("2 · Affichage")
    show_ids = st.checkbox("Numéroter les candidats", value=True)
    show_gt = st.checkbox("Afficher les annotations (vert)", value=has_annotations, disabled=not has_annotations,
                          help="Disponible pour les échantillons du dépôt uniquement.")

    st.subheader("3 · Détection")
    with st.expander("Réglages avancés"):
        channel = st.selectbox("Canal", ["gray", "green"], key="p_channel",
                               help="Canal de l'image sur lequel on cherche les taches sombres.")
        blackhat_ksize = st.slider("Filtre black-hat (px)", 5, 41, DEFAULTS.blackhat_ksize, 2, key="p_blackhat",
                                   help="Doit être plus grand que les taches à détecter.")
        min_contrast = st.slider("Contraste minimal", 5, 60, int(DEFAULTS.min_contrast), 1, key="p_contrast",
                                 help="Trop bas : le bruit passe. Trop haut : des parasites pâles sont perdus.")
        min_area, max_area = st.slider("Aire (px²)", 1, 400, (int(DEFAULTS.min_area), int(DEFAULTS.max_area)),
                                       key="p_area", help="En dessous : bruit. Au-dessus : cellule blanche, amas.")
        min_circ = st.slider("Circularité minimale", 0.0, 1.0, DEFAULTS.min_circularity, 0.05, key="p_circ",
                             help="À cette taille (~9 px de diamètre) elle sépare peu : presque tout vaut ~1.")
        st.button("Réinitialiser", on_click=reset_params, width="stretch")
    params = CandidateParams(channel=channel, blackhat_ksize=blackhat_ksize, min_contrast=float(min_contrast),
                             min_area=float(min_area), max_area=float(max_area), min_circularity=min_circ)

    st.divider()
    st.caption("AMA · PIIA Cohorte 2 · Spécial Machine Learning")

# ───────────────────────────── analyse ─────────────────────────────
try:
    image = decode_image(data)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

quality = run_quality(data)
try:
    candidates, candidates_error = run_candidates(data, params), None
except Exception as exc:  # noqa: BLE001  (on affiche l'erreur dans l'interface au lieu de planter)
    candidates, candidates_error = [], exc

if image.shape[:2] != REF_SHAPE:
    st.info(f"Cette image fait {image.shape[1]} × {image.shape[0]} px. Les seuils de la démo ont été calibrés sur des "
            f"images de {REF_SHAPE[1]} × {REF_SHAPE[0]} px : les résultats sont à interpréter avec prudence.")

raised = [FLAG_LABELS[k] for k, v in quality["flags"].items() if v]
tab_quality, tab_candidates, tab_method = st.tabs(["Qualité de l'image", "Régions candidates", "Méthode et limites"])

# ───────────────────────────── onglet qualité ─────────────────────────────
with tab_quality:
    if raised:
        st.warning("**Image à contrôler** : " + " · ".join(raised), icon="⚠️")
    else:
        st.success("**Aucune alerte de qualité** : l'image est dans la norme du jeu de données.", icon="✅")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.image(image, caption="Image analysée", width="stretch")
    with right:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("Netteté", f"{quality['sharpness']:.1f}", help="Variance du Laplacien : plus c'est haut, plus l'image est nette.")
            c2.metric("Luminosité", f"{quality['brightness']:.0f} / 255", help="Niveau de gris moyen.")
            c3, c4 = st.columns(2)
            c3.metric("Contraste", f"{quality['contrast']:.0f}", help="Étendue entre les centiles 1 et 99.")
            c4.metric("Pixels saturés", f"{100 * quality['overexposure_ratio']:.2f} %")
            st.metric("Pixels très sombres", f"{100 * quality['underexposure_ratio']:.2f} %")
        if any(v is None for v in quality["flags"].values()):
            st.caption("Certaines alertes ne sont pas évaluées : leur seuil n'est pas calibré.")
        st.caption("Les seuils de netteté et de contraste sont relatifs au jeu de données (5 % des images les moins "
                   "bonnes) : une alerte signale une image moins bonne que les autres, pas une image inutilisable.")

    with st.expander("Statistiques de couleur et histogrammes RGB"):
        st.dataframe(pd.Series(quality["color_statistics"], name="valeur").round(3).to_frame(), width="stretch")
        st.line_chart(rgb_histogram(image), color=["#e5484d", "#30a46c", "#3e63dd"])

# ───────────────────────────── onglet candidats ─────────────────────────────
with tab_candidates:
    if candidates_error is not None:
        if isinstance(candidates_error, NotImplementedError):
            st.warning("La détection de candidats n'est pas encore branchée dans cette version.")
        else:
            st.error(f"La détection de candidats a échoué : {candidates_error}")
    else:
        annotations = clip_boxes(parse_boxes(xml_path), *image.shape[1::-1]) if has_annotations else []
        overlay = draw_overlay(image, candidates, annotations if show_gt else [], show_ids)

        with st.container(border=True):
            cols = st.columns(3 if annotations else 1)
            cols[0].metric("Régions candidates", len(candidates))
            if annotations:
                m = match_candidates(candidates, annotations)
                cols[1].metric("Parasites annotés retrouvés", f"{m['n_found']} / {m['n_gt']}",
                               help="Un parasite est retrouvé si le centre d'au moins un candidat tombe dans sa boîte.")
                cols[2].metric("Candidats hors annotation", f"{m['n_false_candidates']} / {m['n_candidates']}",
                               help="Faux positifs acceptables à ce stade : la Phase 2 les éliminera.")

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.image(overlay, caption="Orange : régions candidates · Vert : annotations du jeu de données",
                     width="stretch")
            st.download_button("Télécharger l'image annotée (PNG)", png_bytes(overlay),
                               file_name="candidats.png", mime="image/png")
        with right:
            if candidates:
                table = candidates_table(candidates)
                st.dataframe(table, hide_index=True, width="stretch", height=min(38 + 35 * len(candidates), 420))
                st.download_button("Télécharger les candidats (CSV)", table.to_csv(index=False).encode("utf-8"),
                                   file_name="candidats.csv", mime="text/csv")
            else:
                st.info("Aucune région candidate pour ces réglages.")
            st.caption("Un candidat est une tache sombre compatible avec un parasite : ce peut aussi être un débris "
                       "ou un artefact. La classification relève de la Phase 2. Les boîtes annotées du jeu de "
                       "données font environ 40 px et ne donnent pas la taille réelle du parasite.")

# ───────────────────────────── onglet méthode ─────────────────────────────
with tab_method:
    st.markdown("""
**Chaîne de traitement de la Phase 0**

`Image` → `Contrôle qualité` → `Prétraitement` → `Régions candidates`

- **Contrôle qualité** : netteté (variance du Laplacien), luminosité, contraste (étendue entre les centiles 1 et 99),
  part de pixels saturés ou très sombres, statistiques de couleur.
- **Régions candidates** : filtre black-hat (taches sombres sur fond clair), seuil d'Otsu avec plancher de contraste,
  morphologie mathématique, composantes connexes, contours, puis filtre sur l'aire et mesure du périmètre et de la
  circularité (4πA / P²).

**Limites connues**
- Les seuils ont été calibrés sur un seul jeu de données (frottis épais, smartphone, coloration de Field).
- Les alertes de qualité sont relatives à ce jeu de données ; aucun pixel n'y est saturé, la détection de
  surexposition est validée par des dégradations synthétiques.
- Les candidats ne sont pas classés : parasites, noyaux de globules blancs et débris peuvent être confondus.
- Aucune évaluation hors domaine (autre microscope, autre laboratoire) n'a encore été faite : c'est l'objet des phases suivantes.
""")
    st.info("**Prototype de recherche et de formation.** PaluScope ne pose aucun diagnostic clinique, ne prescrit aucun "
            "traitement et n'est pas un dispositif médical : il assiste la lecture d'un frottis par un humain.",
            icon="ℹ️")

# ───────────────────────────── pied de page ─────────────────────────────
st.divider()
st.caption("⚕️ Prototype de recherche et de formation : aucun diagnostic clinique. "
           "Les résultats doivent être relus par un professionnel.")