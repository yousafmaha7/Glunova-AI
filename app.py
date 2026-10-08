"""
============================================================
                         GLUNOVA AI
       AI-based Gestational Diabetes Mellitus Prediction Tool
============================================================

Run:
    streamlit run app.py

Research Use Only.
"""

import base64
import io
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from report_extraction import (
    FIELD_LABELS,
    FILE_TYPES,
    ocr_available,
    read_reports,
    summarise_for_form,
)


# ============================================================
# RESOURCE PATH
# ============================================================

def resource_path(relative_path):
    """Return correct path for development and PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(resource_path("model/model.pkl"))
PREPROCESSOR_PATH = Path(resource_path("model/gdm_preprocessor.pkl"))
LOGO_PATH = Path(resource_path("assets/images/glunova_logo.png"))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Glunova AI | GDM Prediction",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# MODEL + PREPROCESSOR
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


if not MODEL_PATH.exists():
    st.error(
        "model.pkl was not found. Please place it inside the "
        "'model' folder."
    )
    st.stop()

if not PREPROCESSOR_PATH.exists():
    st.error(
        "gdm_preprocessor.pkl was not found. Please place it inside "
        "the 'model' folder."
    )
    st.stop()

try:
    model = load_model()
    prep = load_preprocessor()
except Exception as e:
    st.error(f"Unable to load the Glunova AI model files: {e}")
    st.stop()


MODEL_FEATURES = prep["model_features"]
RAW_FEATURE_COLUMNS = prep["raw_feature_columns_after_dummies"]
LOG_COLUMNS = prep["log_columns"]
REMOVED_COLUMNS = prep["removed_columns"]
IMPUTER = prep["imputer"]


# ============================================================
# USER-FACING FEATURE DEFINITIONS
# ============================================================

CLINICAL_FEATURES = [
    "Maternal age",
    "Gravidity",
    "Family History of Diabetes",
    "Mean Diastolic BP",
    "Mean Systolic BP",
    "Fasting Glucose (mg/dl)",
    "Hb (g/dl)",
    "RBC (millions/ml)",
    "WBC (10^3/uL)",
    "MCHC (g/dl)",
    "Lymphocytes (Absolute count 10^3/uL)",
    "Eosinophils (Absolute count 10^3/uL)",
    "BMI",
    "Blood Type",
]


# ============================================================
# PREPROCESSING
# ============================================================

def add_blood_type_dummies(df):
    """
    Reproduce the notebook's Blood Type encoding:
    pd.get_dummies(..., drop_first=True).

    The saved model expects Blood Type_AB and Blood Type_B.
    Blood Type_O is a reference category and was removed later.
    """

    df = df.copy()

    if "Blood Type" not in df.columns:
        raise ValueError("Blood Type column is required.")

    blood = df["Blood Type"].astype(str).str.strip().str.upper()

    df["Blood Type_AB"] = (blood == "AB").astype(int)
    df["Blood Type_B"] = (blood == "B").astype(int)
    df["Blood Type_O"] = (blood == "O").astype(int)

    df.drop(columns=["Blood Type"], inplace=True)

    return df


def preprocess_for_model(df):
    """
    Reproduce the preprocessing used before fitting the saved model:

    1. Blood Type dummy encoding
    2. Iterative imputation using the saved training imputer
    3. Remove Random Glucose
    4. log1p selected skewed variables
    5. Remove selected features
    6. Return exact model feature order
    """

    data = df.copy()
    data.columns = data.columns.str.strip()

    # Blood type
    data = add_blood_type_dummies(data)

    # Ensure every raw training column exists.
    for col in RAW_FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = np.nan

    data = data[RAW_FEATURE_COLUMNS]

    # Iterative imputation fitted on the original training split.
    data = pd.DataFrame(
        IMPUTER.transform(data),
        columns=RAW_FEATURE_COLUMNS,
        index=data.index,
    )

    # Remove Random Glucose exactly as in notebook.
    if "Random Glucose (mg/dl)" in data.columns:
        data.drop(columns=["Random Glucose (mg/dl)"], inplace=True)

    # Log transform exactly as in notebook.
    for col in LOG_COLUMNS:
        if col in data.columns:
            # Clip tiny negative numerical artifacts caused by imputation.
            data[col] = data[col].clip(lower=0)
            data[col] = np.log1p(data[col])

    # Feature selection exactly as in notebook.
    for col in REMOVED_COLUMNS:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    # Exact model order.
    missing = [c for c in MODEL_FEATURES if c not in data.columns]

    if missing:
        raise ValueError(
            "The preprocessed data is missing model features: "
            + ", ".join(missing)
        )

    return data[MODEL_FEATURES]


# ============================================================
# SITE CONTENT
# Edit the values below; empty strings ("") are hidden on the page.
# ============================================================

LAB_NAME = "Integrative Omics and Molecular Modelling Lab"
LAB_INSTITUTION = (
    "Department of Bioinformatics and Biotechnology, "
    "Government College University Faisalabad "
    "Pakistan "
)
LAB_ADDRESS = ""       # e.g. "City, Country"
LAB_EMAIL = ""         # e.g. "contact@university.edu"
LAB_WEBSITE = ""       # e.g. "https://lab.university.edu"
CITATION = ""          # e.g. "Author A, et al. Journal (2026). doi:..."

# Team / authors. "photo" is a file name inside assets/team/ (optional).
AUTHORS = [
    {
        "name": "Ms. Tahreem Sajjad",
        "affiliation": (
            "Integrative Omics and Molecular Modeling Lab<br>"
            "Department of Bioinformatics and Biotechnology<br>"
            "Government College University Faisalabad, Pakistan"
        ),
        "email": "tahreemsajjad1072@gmail.com",
        "photo": "female_icon.png",
    },
    {
        "name": "Mr. Rana Sheraz Ahmad",
        "affiliation": (
            "Syed Babar Ali School of Science and Engineering (SBASSE)<br>"
            "Lahore University of Management Sciences (LUMS)<br>"
            "Pakistan"
        ),
        "email": "rana.a@lums.edu.pk",
        "photo": "male_icon.png",
    },
    {
        "name": "Ms. Maha Yousaf",
        "affiliation": (
            "Department of Precision Medicine<br>"
            "Sungkyunkwan University<br>"
            "Republic of Korea"
        ),
        "email": "maha.yousaf@g.skku.edu",
        "photo": "female_icon.png",
    },
    {
        "name": "Dr. Muhammad Tahir ul Qamar",
        "affiliation": (
            "Integrative Omics and Molecular Modeling Lab<br>"
            "Department of Bioinformatics and Biotechnology<br>"
            "Government College University Faisalabad, Pakistan"
        ),
        "email": "tahirulqamar@gcuf.edu.pk",
        "photo": "male_icon.png",
    },
]

# Collaborating hospitals shown in the logo banner on the home page.
# "logo" is a file name inside assets/partners/; a labelled placeholder is
# shown until the file exists.
PARTNERS = []

# Hero slideshow: (image file inside assets/slides/, caption).
# Images are 16:10 (e.g. 832 x 520 px); PNG, JPG or SVG.
SLIDES_DIR = Path(resource_path("assets/slides"))
SLIDES = [
    ("01_gdm.jpg", "Gestational Diabetes Mellitus: early identification "
                   "supports timely antenatal care"),
    ("02_workflow.svg", "From the antenatal record to a GDM assessment"),
    ("03_variables.svg", "Routine maternal parameters used by Glunova AI"),
]
SLIDE_SECONDS = 5

TEAM_DIR = Path(resource_path("assets/team"))
PARTNERS_DIR = Path(resource_path("assets/partners"))
RANDOM_FOREST_FIGURE = Path(resource_path("assets/images/random_forest.png"))

PAGES = {
    "home": "Home",
    "predict": "Patient Assessment",
    "batch": "Cohort Assessment",
    "about": "About",
    "team": "Team",
}


# ============================================================
# STYLING
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
    --gn-primary: #2F5597;
    --gn-primary-dark: #23427A;
    --gn-primary-tint: #EAF0F9;
    --gn-ink: #1F2A33;
    --gn-muted: #5B6B77;
    --gn-line: #DCE2EC;
    --gn-surface: #F5F7FB;
    --gn-positive: #9E2A2B;
    --gn-positive-tint: #FBEFEF;
    --gn-negative: #1E6B4F;
    --gn-negative-tint: #ECF5F1;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Source Sans 3', 'Source Sans Pro', -apple-system, sans-serif;
    color: var(--gn-ink);
}
.stApp { background: #FFFFFF; }

h1, h2, h3, h4 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: var(--gn-ink);
}

/* ---------- Page frame ----------
   The root font size scales with the viewport (about 15px on a small
   laptop, 18px on a full-HD screen), and every size below is in rem, so
   the whole page keeps the same proportions on any screen. */
html { font-size: clamp(14px, 0.42vw + 10px, 19px) !important; }

/* Content sits in a centred column (like most academic web servers),
   leaving white margins on wide screens; the gutter keeps a minimum
   margin on small screens. */
.stApp { --gn-gutter: clamp(1rem, 3vw, 2.5rem); --gn-content: 78rem; }
[data-testid="stMain"] { overflow-x: hidden; }

.block-container {
    max-width: calc(var(--gn-content) + 2 * var(--gn-gutter)) !important;
    margin: 0 auto;
    padding: 0 var(--gn-gutter) 0 var(--gn-gutter);
}
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
    display: none;
}
[data-testid="stMainBlockContainer"] { padding-top: 0 !important; }
[data-testid="stMain"] { padding-top: 0 !important; }
/* The element holding this stylesheet should not take up layout space */
[data-testid="stElementContainer"]:has(style) { display: none; }

/* Keep the footer at the bottom on short pages */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    min-height: calc(100vh - 1rem);
}
[data-testid="stElementContainer"]:has(.gn-footer) { margin-top: auto; }

/* Full-width bands (partner banner, footer): the background spans the
   whole window while the content stays aligned with the centred column. */
.gn-bleed {
    margin-left: calc(50% - 50vw);
    margin-right: calc(50% - 50vw);
    padding-left: calc(50vw - 50%);
    padding-right: calc(50vw - 50%);
}

a { color: var(--gn-primary); }

/* ---------- Navigation bar ---------- */
.gn-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.8rem;
    padding-top: 0.9rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--gn-line);
    background: #FFFFFF;
}
.gn-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    text-decoration: none !important;
}
.gn-brand img { width: 42px; height: 42px; object-fit: contain; }
.gn-brand span {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--gn-primary);
    letter-spacing: 0.01em;
}
.gn-links {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 1.6rem;
}
.gn-links a {
    font-size: 0.95rem;
    color: var(--gn-ink) !important;
    text-decoration: none !important;
    padding: 0.25rem 0;
    border-bottom: 2px solid transparent;
}
.gn-links a:hover { color: var(--gn-primary) !important; }
.gn-links a.active {
    color: var(--gn-primary) !important;
    border-bottom-color: var(--gn-primary);
}
.gn-links a.gn-cta {
    background: var(--gn-primary);
    color: #FFFFFF !important;
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    border: none;
    font-weight: 600;
}
.gn-links a.gn-cta:hover { background: var(--gn-primary-dark); }

/* ---------- Page title (inner pages) ---------- */
.gn-pagehead {
    border-bottom: 1px solid var(--gn-line);
    padding-top: 2rem;
    padding-bottom: 1.4rem;
    margin-bottom: 1.8rem;
}
.gn-pagehead h1 {
    font-size: 1.9rem !important;
    font-weight: 700;
    color: var(--gn-primary) !important;
    margin: 0;
    padding: 0;
}
.gn-pagehead p {
    color: var(--gn-muted);
    margin: 0.3rem 0 0 0;
    font-size: 1rem;
}
.gn-crumb {
    font-size: 0.8rem;
    color: var(--gn-muted);
    margin-bottom: 0.4rem;
}
.gn-crumb a { text-decoration: none; }

/* ---------- Hero (fills the first screen) ---------- */
.st-key-hero {
     min-height: auto;
    justify-content: center;
    padding: 2rem 0 1rem 0;
}
.gn-hero-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: clamp(2.6rem, 4.2vw, 4.4rem);
    font-weight: 700;
    color: var(--gn-primary);
    line-height: 1.05;
    margin: 0 0 0.6rem 0;
}
.gn-hero-tag {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.45rem;
    color: var(--gn-primary);
    font-style: italic;
    margin-bottom: 1rem;
}
.gn-hero-text {
    font-size: 1.15rem;
    color: var(--gn-ink);
    line-height: 1.65;
    max-width: 40rem;
    margin-bottom: 1.6rem;
}
.gn-btn {
    display: inline-block;
    padding: 0.6rem 1.5rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
    text-decoration: none !important;
    margin: 0 0.6rem 0.6rem 0;
    border: 1.5px solid var(--gn-primary);
}
.gn-btn.primary { background: var(--gn-primary); color: #FFFFFF !important; }
.gn-btn.primary:hover { background: var(--gn-primary-dark); }
.gn-btn.outline { background: #FFFFFF; color: var(--gn-primary) !important; }
.gn-btn.outline:hover { background: var(--gn-primary-tint); }

.gn-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr));
    max-width: 40rem;
    gap: 1rem;
    margin: 1.8rem 0 1rem 0;
}
.gn-stat {
    padding: 0.9rem 1.1rem;
    border-radius: 6px;
    background: #FFFFFF;
    box-shadow: 0 2px 10px rgba(20, 40, 90, 0.10);
    text-align: center;
}
.gn-stat .num {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--gn-primary);
    line-height: 1.1;
}
.gn-stat .lbl {
    font-size: 0.78rem;
    color: var(--gn-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* ---------- Section headings (centred between rules) ---------- */
.gn-heading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin: 3.4rem 0 1.6rem 0;
}
.gn-heading::before, .gn-heading::after {
    content: "";
    width: 3rem;
    height: 2px;
    background: var(--gn-primary);
}
.gn-heading span {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--gn-ink);
}
@media (max-width: 700px) {
    .gn-heading span { font-size: 1.25rem; text-align: center; }
    .gn-heading::before, .gn-heading::after { width: 1.5rem; }
    .gn-partners { gap: 1rem; }
    .gn-partner { width: 44%; }
}
.gn-lead {
    text-align: center;
    max-width: 58rem;
    margin: -0.6rem auto 1.8rem auto !important;
    color: var(--gn-muted);
    font-size: 1.02rem;
    line-height: 1.6;
}
.gn-band {
    background: var(--gn-surface);
    padding-top: 0.2rem;
    padding-bottom: 2.6rem;
    margin-top: 3rem;
}
.gn-band .gn-heading { margin-top: 2.6rem; }

/* ---------- Feature tiles ---------- */
.gn-features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.1rem;
}
@media (max-width: 1100px) { .gn-features { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 700px) { .gn-features { grid-template-columns: 1fr; } }
.gn-feature {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    background: #FFFFFF;
    border: 1px solid var(--gn-line);
    border-radius: 6px;
    padding: 1rem 1.1rem;
}
.gn-feature svg { flex-shrink: 0; margin-top: 0.1rem; }
.gn-feature { padding: 1.2rem 1.3rem; }
.gn-feature b { display: block; font-size: 1.02rem; margin-bottom: 0.25rem; }
.gn-feature div { font-size: 0.92rem; color: var(--gn-muted); line-height: 1.55; }

/* ---------- Cards ---------- */
.gn-cards {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.4rem;
}
.gn-cards.three {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}
.gn-cards.three > .gn-card { width: 20rem; }
@media (max-width: 900px) {
    .gn-cards { grid-template-columns: 1fr; }
    .gn-cards.three > .gn-card { width: 100%; }
}
.gn-card {
    background: #FFFFFF;
    border-radius: 6px;
    box-shadow: 0 2px 12px rgba(20, 40, 90, 0.09);
    padding: 1.3rem 1.4rem;
}
.gn-card h4 {
    font-size: 1.08rem !important;
    font-weight: 600;
    margin: 0 0 0.4rem 0 !important;
    padding: 0 !important;
}
.gn-card p { font-size: 0.94rem; color: var(--gn-muted); line-height: 1.55; }
.gn-card img { width: 100%; border-radius: 4px; margin-top: 0.6rem; }
.gn-card .venue { font-size: 0.86rem; color: var(--gn-muted); margin-bottom: 0.4rem; }
.gn-card .more {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-decoration: none;
}
/* ---------- Steps ---------- */
.gn-steps {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.2rem;
}
@media (max-width: 1100px) { .gn-steps { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 700px) { .gn-steps { grid-template-columns: 1fr; } }
.gn-steps .gn-card { box-shadow: none; border: 1px solid var(--gn-line); }
.gn-step-num {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--gn-primary);
    line-height: 1;
    margin-bottom: 0.5rem;
}

/* ---------- Figure ---------- */
.gn-figure {
    background: #FFFFFF;
    border: 1px solid var(--gn-line);
    border-radius: 8px;
    padding: 1rem;
}
.gn-figure img { width: 100%; display: block; }
.gn-figcap {
    font-size: 0.86rem;
    color: var(--gn-muted);
    border-top: 1px solid var(--gn-line);
    padding-top: 0.6rem;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* ---------- FAQ ---------- */
.gn-faq { max-width: 80rem; margin: 0 auto; }
.gn-faq-row {
    display: grid;
    grid-template-columns: 1fr 1.6fr;
    gap: 2rem;
    padding: 1.1rem 0;
    border-bottom: 1px solid var(--gn-line);
}
@media (max-width: 900px) { .gn-faq-row { grid-template-columns: 1fr; gap: 0.4rem; } }
.gn-faq-q { font-weight: 600; font-size: 0.98rem; display: flex; gap: 0.5rem; }
.gn-faq-q span {
    flex-shrink: 0;
    width: 1.3rem; height: 1.3rem;
    border: 1.5px solid var(--gn-primary);
    border-radius: 50%;
    color: var(--gn-primary);
    font-size: 0.75rem;
    display: flex; align-items: center; justify-content: center;
}
.gn-faq-a { font-size: 0.95rem; color: var(--gn-muted); line-height: 1.6; }

/* ---------- Team ---------- */
.gn-person { text-align: center; }
.gn-avatar {
    width: 110px; height: 110px;
    border-radius: 50%;
    margin: 0 auto 0.8rem auto;
    background: var(--gn-primary-tint);
    color: var(--gn-primary);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 700;
    overflow: hidden;
}
.gn-avatar img { width: 100%; height: 100%; object-fit: cover; margin: 0; border-radius: 0; }
.gn-person .role { color: var(--gn-primary); font-size: 0.9rem; font-weight: 600; }
.gn-person .aff { color: var(--gn-muted); font-size: 0.88rem; }

/* ---------- Slideshow ---------- */
.gn-slideshow {
    margin-top: 0;
    border-radius: 8px;
    overflow: hidden;
    background: #FFFFFF;
    box-shadow: 0 4px 22px rgba(20, 40, 90, 0.12);
}
.gn-slides {
    position: relative;
    aspect-ratio: 16 / 10;
    background: #FFFFFF;
}
.gn-slide {
    position: absolute;
    inset: 0;
    opacity: 0;
    animation: gn-fade var(--gn-cycle) infinite;
}
.gn-slide img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}
.gn-slidebar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 1rem;
    border-top: 1px solid var(--gn-line);
}
.gn-captions { position: relative; height: 1.3rem; flex: 1; }
.gn-caption {
    position: absolute;
    inset: 0;
    opacity: 0;
    font-size: 0.85rem;
    color: var(--gn-muted);
    animation: gn-fade var(--gn-cycle) infinite;
}
.gn-dots { display: flex; gap: 0.4rem; }
.gn-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--gn-line);
    animation: gn-dot var(--gn-cycle) infinite;
}
@media (prefers-reduced-motion: reduce) {
    .gn-slide, .gn-caption, .gn-dot { animation: none; }
    .gn-slide:first-child, .gn-caption:first-child { opacity: 1; }
}

/* ---------- Form section cards ---------- */
[class*="st-key-card_"] {
    border: 1px solid var(--gn-line);
    border-radius: 6px;
    padding: 1.1rem 1.3rem 0.6rem 1.3rem;
    background: #FFFFFF;
}
.gn-section-head { display: flex; align-items: baseline; gap: 0.7rem; margin-bottom: 0.15rem; }
.gn-section-num {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.95rem; font-weight: 600; color: var(--gn-primary);
}
.gn-section-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.12rem; font-weight: 600; color: var(--gn-ink);
}
.gn-section-desc {
    font-size: 0.86rem; color: var(--gn-muted);
    margin: 0 0 0.6rem 0; padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--gn-line);
}
.gn-intro { font-size: 0.95rem; color: var(--gn-muted); margin: 0 0 0.9rem 0; }

/* ---------- Inputs ---------- */
[data-testid="stWidgetLabel"] p {
    font-size: 0.86rem !important; font-weight: 500; color: var(--gn-ink);
}
[data-testid="stNumberInput"] input, [data-baseweb="select"] > div {
    font-size: 0.92rem; min-height: 2.35rem;
}
[data-testid="stNumberInputContainer"], [data-baseweb="select"] > div {
    border-color: var(--gn-line) !important;
    background: var(--gn-surface) !important;
}
[data-testid="stNumberInputContainer"]:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--gn-primary) !important;
}
[data-testid="stForm"] { border: none; padding: 0; }

/* ---------- Buttons ---------- */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stDownloadButton"] > button {
    border-radius: 999px; font-weight: 600; padding: 0.55rem 1.4rem;
}
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
    background: var(--gn-primary); border: 1px solid var(--gn-primary);
}
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover {
    background: var(--gn-primary-dark); border-color: var(--gn-primary-dark);
}
[data-testid="stDownloadButton"] > button {
    border: 1.5px solid var(--gn-primary); color: var(--gn-primary); background: #FFFFFF;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--gn-primary-tint); color: var(--gn-primary-dark);
}

/* ---------- Result ---------- */
.gn-result {
    border: 1px solid var(--gn-line);
    border-left: 5px solid var(--gn-accent);
    background: var(--gn-accent-tint);
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    margin: 0 0 1rem 0;
}
.gn-result.positive { --gn-accent: var(--gn-positive); --gn-accent-tint: var(--gn-positive-tint); }
.gn-result.negative { --gn-accent: var(--gn-negative); --gn-accent-tint: var(--gn-negative-tint); }
.gn-result.pending { --gn-accent: var(--gn-line); --gn-accent-tint: var(--gn-surface); }
.gn-result.pending .gn-result-text { color: var(--gn-muted); font-size: 0.9rem; }
.gn-result-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--gn-muted);
}
.gn-result-value {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.1rem; font-weight: 600; color: var(--gn-accent);
    line-height: 1.2; margin: 0.15rem 0 0.3rem 0;
}
.gn-result-text { font-size: 0.97rem; color: var(--gn-ink); }
.gn-note {
    font-size: 0.85rem; color: var(--gn-muted);
    border-top: 1px solid var(--gn-line);
    padding-top: 0.6rem; margin-top: 0.4rem;
}
.gn-h {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.25rem; font-weight: 600; color: var(--gn-ink);
    margin: 1.2rem 0 0.5rem 0;
}
.gn-list { padding-left: 1.1rem; margin: 0.4rem 0 1rem 0; }
.gn-list li, .gn-side-text {
    font-size: 0.9rem !important; line-height: 1.5; margin-bottom: 0.3rem;
}
[class*="st-key-card_guidance"] { padding-bottom: 1rem; }
[class*="st-key-card_guidance"] .gn-section-title { font-size: 1rem; }

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    border: 1px solid var(--gn-line); border-radius: 6px;
    padding: 0.8rem 1rem; background: var(--gn-surface);
}
[data-testid="stMetricValue"] {
    font-family: 'Source Serif 4', Georgia, serif; color: var(--gn-primary);
}

/* ---------- Steps (batch) ---------- */
.gn-step {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--gn-primary);
}
.gn-step-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.05rem; font-weight: 600; margin: 0.1rem 0 0.3rem 0;
}
.gn-step-text { font-size: 0.88rem; color: var(--gn-muted); margin-bottom: 0.7rem; }

/* ---------- About ---------- */
.gn-prose p, .gn-prose li { font-size: 0.98rem; line-height: 1.65; }
.gn-spec { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.gn-spec, .gn-spec tr { border: none !important; }
.gn-spec td {
    padding: 0.5rem 0.2rem;
    border: none !important;
    border-bottom: 1px solid var(--gn-line) !important;
    vertical-align: top;
}
.gn-spec td:first-child { color: var(--gn-muted); width: 42%; }
.gn-spec td:last-child { font-weight: 500; }

/* ---------- Footer ---------- */
.gn-footer {
    margin-top: 3.5rem;
    background: #16264A;
    color: #C3D5DE;
    font-size: 0.87rem;
    line-height: 1.6;
}
.gn-footer-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 3rem;
    align-items: start;
    padding: 2.4rem 0 1.6rem 0;
}
@media (max-width: 900px) {
    .gn-footer-grid { grid-template-columns: 1fr; gap: 1.2rem; }
}
.gn-footer h5 {
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.74rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #FFFFFF;
    margin: 0 0 0.6rem 0; padding: 0;
}
.gn-footer-brand {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.3rem; font-weight: 600; color: #FFFFFF; margin-bottom: 0.4rem;
}
.gn-footer p { margin: 0 0 0.3rem 0; font-size: 0.87rem; }
.gn-footer .gn-lab { color: #FFFFFF; font-weight: 600; }
.gn-footer-lab { margin-top: 0.9rem; }
.gn-author { margin-bottom: 0.7rem; }
.gn-footer p.gn-author-name { color: #FFFFFF; font-weight: 600; margin: 0; }
.gn-footer p.gn-author-meta { margin: 0; font-size: 0.83rem; }
.gn-footer a { color: #A9C3F0; text-decoration: none; }
.gn-footer a:hover { text-decoration: underline; }
.gn-footer-links a { display: block; margin-bottom: 0.25rem; }
.gn-footer-bottom {
    border-top: 1px solid #2E4270;
    padding: 0.9rem 0 1rem 0;
    font-size: 0.8rem;
    color: #93AEBB;
}

/* ---------- Report upload (patient assessment) ---------- */
[class*="st-key-card_scan"] { padding-bottom: 1rem; margin-bottom: 1rem; }
[class*="st-key-card_scan"] [data-testid="stButton"] > button {
    border-radius: 999px; font-weight: 600;
    border: 1.5px solid var(--gn-primary); color: var(--gn-primary);
    background: #FFFFFF;
}
[class*="st-key-card_scan"] [data-testid="stButton"] > button:hover:enabled {
    background: var(--gn-primary-tint); color: var(--gn-primary-dark);
}
</style>
"""


# ============================================================
# HELPERS
# ============================================================

def image_data_uri(path):
    mime = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


@st.cache_data
def logo_emblem_base64():
    """Crop the emblem (without the wordmark) and return it as base64 PNG."""
    image = Image.open(LOGO_PATH).convert("RGB")
    w, h = image.size
    emblem = image.crop(
        (int(w * 0.19), int(h * 0.045), int(w * 0.81), int(h * 0.66))
    )
    emblem.thumbnail((160, 160))
    buffer = io.BytesIO()
    emblem.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def available_slides():
    return [
        (SLIDES_DIR / name, caption)
        for name, caption in SLIDES
        if (SLIDES_DIR / name).exists()
    ]


def slideshow_keyframes():
    n = max(len(available_slides()), 1)
    if n == 1:
        return (
            "@keyframes gn-fade {0%, 100% {opacity: 1}}"
            "@keyframes gn-dot {0%, 100% {background: #2F5597}}"
        )
    visible = 100 / n
    fade = min(4, visible / 4)
    return (
        "@keyframes gn-fade {"
        f"0% {{opacity: 0}} {fade:.2f}% {{opacity: 1}} "
        f"{visible:.2f}% {{opacity: 1}} {visible + fade:.2f}% {{opacity: 0}} "
        "100% {opacity: 0}}"
        "@keyframes gn-dot {"
        f"0% {{background: #2F5597}} {visible:.2f}% {{background: #2F5597}} "
        f"{visible + 0.01:.2f}% {{background: #DCE2EC}} "
        "100% {background: #DCE2EC}}"
    )


@st.cache_data
def slideshow_html():
    """Auto-advancing CSS slideshow (no JavaScript needed)."""
    slides = available_slides()
    if not slides:
        return ""

    cycle = len(slides) * SLIDE_SECONDS

    def delay(i):
        return f'style="animation-delay:{i * SLIDE_SECONDS}s"'

    images = "".join(
        f'<div class="gn-slide" {delay(i)}>'
        f'<img src="{image_data_uri(path)}" alt="{caption}"></div>'
        for i, (path, caption) in enumerate(slides)
    )
    captions = "".join(
        f'<div class="gn-caption" {delay(i)}>{caption}</div>'
        for i, (_, caption) in enumerate(slides)
    )
    dots = "".join(
        f'<span class="gn-dot" {delay(i)}></span>'
        for i in range(len(slides))
    )

    return (
        f'<div class="gn-slideshow" style="--gn-cycle:{cycle}s">'
        f'<div class="gn-slides">{images}</div>'
        f'<div class="gn-slidebar"><div class="gn-captions">{captions}</div>'
        f'<div class="gn-dots">{dots}</div></div></div>'
    )


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def page_link(page):
    return f"?page={page}"


def heading(title):
    html(f'<div class="gn-heading"><span>{title}</span></div>')


def page_header(title, subtitle):
    html(
        f'<div class="gn-pagehead">'
        f'<div class="gn-crumb"><a href="{page_link("home")}" '
        f'target="_self">Home</a> &rsaquo; {title}</div>'
        f"<h1>{title}</h1><p>{subtitle}</p></div>"
    )


def section_header(number, title, description):
    html(
        f'<div class="gn-section-head">'
        f'<span class="gn-section-num">{number}</span>'
        f'<span class="gn-section-title">{title}</span></div>'
        f'<p class="gn-section-desc">{description}</p>'
    )


def icon(path_d):
    return (
        '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
        'stroke="#2F5597" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round">{path_d}</svg>'
    )


ICONS = {
    "person": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2"/>'
             '<path d="M3 10h18M3 15h18M9 4v16"/>',
    "flow": '<circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/>'
            '<circle cx="12" cy="18" r="2"/><path d="M7 6h10M6 8l5 8M18 8l-5 8"/>',
    "drop": '<path d="M12 3s6 7 6 11a6 6 0 0 1-12 0c0-4 6-11 6-11z"/>',
    "download": '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
    "book": '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/>'
            '<path d="M4 19V5"/>',
}


# ============================================================
# NAVIGATION
# ============================================================

page = st.query_params.get("page", "home")
if page not in PAGES:
    page = "home"


def render_nav():
    links = "".join(
        f'<a href="{page_link(key)}" target="_self" '
        f'class="{"active" if key == page else ""}">{label}</a>'
        for key, label in PAGES.items()
    )
    logo = (
        f'<img src="data:image/png;base64,{logo_emblem_base64()}" alt="">'
        if LOGO_PATH.exists() else ""
    )
    html(
    f'<div class="gn-bleed gn-nav">'
    f'<a class="gn-brand" href="{page_link("home")}" target="_self">'
    f"{logo}<span>Glunova AI</span></a>"
    f'<div class="gn-links">{links}</div></div>'
    )


# ============================================================
# PAGE : HOME
# ============================================================

FEATURES = [
    ("person", "Individual patient assessment",
     "Provides GDM analysis report of a single patient."),
    ("table", "Cohort assessment",
     "Provides GDM analysis report of multiple patients."),
    ("drop", "Routine antenatal data",
     "Relies only on measurements already collected in antenatal care: "
     "history, blood pressure, fasting glucose and complete blood count."),
    ("flow", "Handles incomplete records",
     "Missing laboratory values are estimated from the other recorded "
     "parameters, as in the derivation study."),
    ("download", "Exportable reports",
     "Download individual or cohort assessments for the patient record "
     "or further epidemiological analysis."),
    ("book", "Transparent methodology",
     "The predictive model, data handling and clinical parameters are "
     "documented openly."),
]

STEPS = [
    ("1", "Record", "Enter the mother's demographic, obstetric, clinical "
     "and laboratory findings, or complete the cohort template."),
    ("2", "Standardise", "Values are checked, missing laboratory results "
     "are estimated and measurements are harmonised."),
    ("3", "Assess", "The Random Forest model compares the maternal profile "
     "with patterns learned from the study cohort."),
    ("4", "Report", "The outcome is shown as GDM predicted (YES) or GDM "
     "not predicted (NO), with a downloadable report."),
]

FAQS = [
    ("What is Glunova AI?",
     "Glunova AI is a clinical decision-support web server that assesses "
     "Gestational Diabetes Mellitus status from routinely collected "
     "antenatal data using a Random Forest model."),
    ("Who can use Glunova AI?",
     "Obstetric and maternal health researchers, clinicians and students. "
     "Access is free and does not require an account."),
    ("Which clinical data are required?",
     "Maternal age, gravidity, BMI, family history of diabetes, blood group, "
     "mean systolic and diastolic blood pressure, fasting plasma glucose and "
     "six complete blood count parameters."),
    ("Is patient data stored?",
     "No. Entered values, uploaded spreadsheets and report scans are "
     "processed on the server only to generate the assessment and are not "
     "saved by the application. Report scans are read with OCR software "
     "running on the same server; they are not sent to any other service."),
    ("Can the outcome be used for diagnosis?",
     "No. Glunova AI is intended for research and educational use only. "
     "GDM must be diagnosed according to established clinical criteria by "
     "a qualified healthcare professional."),
]


def partner_logo(partner):
    logo = PARTNERS_DIR / partner["logo"] if partner.get("logo") else None
    if logo and logo.exists():
        return (
            f'<div class="gn-partner"><img src="{image_data_uri(logo)}" '
            f'alt="{partner["name"]}" title="{partner["name"]}"></div>'
        )
    return (
        '<div class="gn-partner placeholder"><b>Hospital logo</b>'
        f'<code>assets/partners/{partner.get("logo") or "logo.png"}</code>'
        "</div>"
    )


def render_home():
    with st.container(key="hero"):
        hero_left, hero_right = st.columns(
            [1, 1.05], gap="large", vertical_alignment="center"
        )

        with hero_left:
            html(
                '<div class="gn-hero-title">Glunova AI</div>'
                '<div class="gn-hero-tag">Clinical decision support for '
                "Gestational Diabetes Mellitus</div>"
                '<div class="gn-hero-text">Glunova AI assesses maternal GDM '
                "status from routinely collected antenatal data: "
                "demographic and obstetric history, blood pressure, fasting "
                "plasma glucose and the complete blood count. Developed at "
                f"the {LAB_NAME}.</div>"
                ''
                '<div class="gn-stats">'
                '<div class="gn-stat"><div class="num">14</div>'
                '<div class="lbl">Antenatal parameters</div></div>'
                '<div class="gn-stat"><div class="num">3</div>'
                '<div class="lbl">Clinical domains</div></div>'
                '<div class="gn-stat"><div class="num">CBC</div>'
                '<div class="lbl">Routine laboratory tests</div></div>'
                '<div class="gn-stat"><div class="num">Open</div>'
                '<div class="lbl">Free access</div></div>'
                "</div>"
            )

        with hero_right:
            html(slideshow_html())

    # ---------------- Features ----------------
    heading("Features")
    tiles = "".join(
        f'<div class="gn-feature">{icon(ICONS[key])}'
        f"<div><b>{title}</b>{text}</div></div>"
        for key, title, text in FEATURES
    )
    html(f'<div class="gn-features">{tiles}</div>')

    # ---------------- How it works ----------------
    heading("How it works")
    html(
        '<div class="gn-steps">'
        + "".join(
            f'<div class="gn-card"><div class="gn-step-num">{num}</div>'
            f"<h4>{title}</h4><p>{text}</p></div>"
            for num, title, text in STEPS
        )
        + "</div>"
    )

           # ---------------- FAQ ----------------
    heading("Frequently Asked Questions")
    html(
        '<div class="gn-faq">'
        + "".join(
            f'<div class="gn-faq-row"><div class="gn-faq-q"><span>?</span>'
            f'{q}</div><div class="gn-faq-a">{a}</div></div>'
            for q, a in FAQS
        )
        + "</div>"
    )


# ============================================================
# PAGE : TEAM
# ============================================================

def initials(name):
    return "".join(part[0] for part in name.split()[:2]).upper()


def render_team():
    page_header(
        "Team",
        f"The research team behind Glunova AI at the {LAB_NAME}.",
    )

    cards = []
    for author in AUTHORS:
        photo = TEAM_DIR / author["photo"] if author.get("photo") else None
        if photo and photo.exists():
            avatar = f'<img src="{image_data_uri(photo)}" alt="">'
        else:
            avatar = initials(author["name"])
        email = (
            f'<p><a href="mailto:{author["email"]}">{author["email"]}</a></p>'
            if author.get("email") else ""
        )
        cards.append(
            f'<div class="gn-card gn-person"><div class="gn-avatar">{avatar}'
            f'</div><h4>{author["name"]}</h4>'
            f'<div class="role">{author.get("role", "")}</div>'
            f'<div class="aff">{author.get("affiliation", "")}</div>'
            f"{email}</div>"
        )
    html(f'<div class="gn-cards three">{"".join(cards)}</div>')

    heading("LAB")
    contact = [f"<h4>{LAB_NAME}</h4>"]
    for value in (LAB_INSTITUTION, LAB_ADDRESS):
        if value:
            contact.append(f"<p>{value}</p>")
    if LAB_EMAIL:
        contact.append(
            f'<p>Email: <a href="mailto:{LAB_EMAIL}">{LAB_EMAIL}</a></p>'
        )
    if LAB_WEBSITE:
        contact.append(
            f'<p>Website: <a href="{LAB_WEBSITE}" target="_blank">'
            f"{LAB_WEBSITE}</a></p>"
        )
    html(
        '<div class="gn-card" style="max-width:40rem;margin:0 auto;'
        f'text-align:center">{"".join(contact)}</div>'
    )

# ============================================================
# FOOTER
# ============================================================

def render_footer():
    citation_text = CITATION or (
        "If Glunova AI supports your research, please acknowledge the "
        "Integrative Omics and Molecular Modelling Lab."
    )

    html(
        '<div class="gn-bleed gn-footer"><div class="gn-footer-grid">'
        # Developer
        '<div><h5>Developer</h5>'
        '<p class="gn-lab">Integrative Omics and Molecular Modelling Lab</p>'
        '<p>Government College University, Faisalabad, Pakistan</p></div>'
        # Citation
        f'<div><h5>Citation</h5><p>{citation_text}</p></div>'
        # Disclaimer
        '<div><h5>Disclaimer</h5>'
        '<p>For research and educational use only. Not a substitute for '
        'professional clinical diagnosis.</p></div>'
        '</div>'
        # Bottom line
        '<div class="gn-footer-bottom">'
        '&copy; 2026 Integrative Omics and Molecular Modelling Lab. '
        'All rights reserved.</div>'
        '</div>'
    )

# ============================================================
# PAGE : SINGLE PREDICTION
# ============================================================

# ------------------------------------------------------------
# Input ranges: (minimum, maximum, decimals). Used by the form AND by the
# report reader, so a value read from a report can never fall outside what
# the form itself accepts.
# ------------------------------------------------------------

INPUT_RANGES = {
    "age": (10.0, 70.0, 0),
    "gravidity": (0.0, 20.0, 0),
    "bmi": (10.0, 80.0, 1),
    "systolic": (50.0, 250.0, 1),
    "diastolic": (30.0, 150.0, 1),
    "fasting_glucose": (20.0, 500.0, 1),
    "hb": (1.0, 30.0, 2),
    "rbc": (0.1, 15.0, 2),
    "wbc": (0.1, 100.0, 2),
    "mchc": (1.0, 60.0, 2),
    "lymphocytes": (0.0, 100.0, 2),
    "eosinophils": (0.0, 100.0, 2),
}

def scan_state():
    ss = st.session_state
    ss.setdefault("prefill", {})
    ss.setdefault("prefill_version", 0)
    ss.setdefault("scan_upload_version", 0)
    ss.setdefault("scan_summary", None)
    return ss


def widget_key(name):
    return f"in_{name}_{st.session_state.prefill_version}"


def number_field(name, label, placeholder, help=None):
    lo, hi, dec = INPUT_RANGES[name]
    return st.number_input(
        label,
        min_value=lo,
        max_value=hi,
        value=st.session_state.prefill.get(name),
        placeholder=placeholder,
        step=10.0 ** -dec,
        format=f"%.{dec}f",
        help=help,
        key=widget_key(name),
    )


def select_field(name, label, options):
    pre = st.session_state.prefill.get(name)
    return st.selectbox(
        label,
        options,
        index=options.index(pre) if pre in options else None,
        placeholder="Select",
        key=widget_key(name),
    )


@st.cache_resource
def ocr_ready():
    return ocr_available()


def apply_extraction(ex):
    """Store pre-fill values and the summary shown above the form."""
    summary = summarise_for_form(ex, INPUT_RANGES)
    ss = st.session_state
    ss.prefill = summary.pop("prefill")
    ss.prefill_version += 1
    ss.scan_summary = summary


def render_scan_card():
    ss = scan_state()

    with st.container(key="card_scan"):
        html(
            '<div class="gn-section-head"><span class="gn-section-title">'
            "Fill the form from reports</span></div>"
            "Upload the file here or manually enter the values in the sections below. Files read on this server are not saved.</p>"
        )

        if not ocr_ready():
            st.caption(
                "Photos and scanned PDFs cannot be read on this server "
                "(Tesseract OCR is not installed). PDFs downloaded from a "
                "laboratory system can still be read."
            )

        files = st.file_uploader(
            "Upload reports",
            type=list(FILE_TYPES),
            accept_multiple_files=True,
            key=f"scan_files_{ss.scan_upload_version}",
            label_visibility="collapsed",
        )

        b1, b2, _ = st.columns([1, 1, 1.3])
        with b1:
            read = st.button(
                "Read reports", disabled=not files, width="stretch"
            )
        with b2:
            clear = st.button(
                "Clear filled values",
                disabled=not ss.prefill and ss.scan_summary is None,
                width="stretch",
            )

        if clear:
            ss.prefill = {}
            ss.scan_summary = None
            ss.prefill_version += 1
            ss.scan_upload_version += 1
            st.rerun()

        if read and files:
            with st.spinner("Reading reports…"):
                ex = read_reports([(f.name, f.getvalue()) for f in files])
            apply_extraction(ex)

        summary = ss.scan_summary
        if summary is None:
            return

        n_filled = len(summary["rows"])
        if n_filled:
            st.info(
                f"{n_filled} of {len(FIELD_LABELS)} values were filled. "
                "Check each one against the original report before the "
                "assessment; the table shows where each value came from."
            )
            st.dataframe(
                pd.DataFrame(summary["rows"]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.warning(
                "No values could be read from these files. Please enter the "
                "findings in the form below."
            )

        if summary["attention"]:
            st.warning(
                "Needs attention:\n\n"
                + "\n".join(f"- {a}" for a in summary["attention"])
            )

        if summary["not_found"] and n_filled:
            st.caption(
                "Not filled, enter by hand: "
                + ", ".join(summary["not_found"]) + "."
            )

        with st.expander("Text read from the reports"):
            for name, method, text in summary["files"]:
                st.markdown(f"**{name}** ({method or 'not read'})")
                st.text(text.strip() or "(no text)")


def render_predict():
    page_header(
        "Patient Assessment",
        "Record the mother's antenatal findings below to assess her GDM "
        "status. All fields are required.",
    )

    form_col, side_col = st.columns([2.1, 1], gap="large")

    with form_col:

        render_scan_card()
        ss = st.session_state
        prefilled = bool(ss.prefill)

        with st.form("gdm_prediction_form", border=False):

            # ----------------------------------------------------
            # 1. DEMOGRAPHIC & OBSTETRIC
            # ----------------------------------------------------

            with st.container(key="card_demographic"):

                section_header(
                    "I.",
                    "Demographic & Obstetric History",
                    "Maternal characteristics recorded at antenatal booking.",
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    age = number_field(
                        "age", "Maternal age (years)", "e.g. 29"
                    )

                with c2:
                    gravidity = number_field(
                        "gravidity", "Gravidity", "e.g. 2",
                        help="Total number of pregnancies, including the "
                             "current one.",
                    )

                with c3:
                    bmi = number_field("bmi", "BMI (kg/m²)", "e.g. 24.5")

                c4, c5, _ = st.columns(3)

                with c4:
                    family_history = select_field(
                        "family_history",
                        "Family history of diabetes",
                        ["No", "Yes"],
                    )

                with c5:
                    blood_type = select_field(
                        "blood_type", "Blood group", ["A", "B", "AB", "O"]
                    )

            # ----------------------------------------------------
            # 2. CLINICAL & BIOCHEMICAL
            # ----------------------------------------------------

            with st.container(key="card_clinical"):

                section_header(
                    "II.",
                    "Clinical & Biochemical Measurements",
                    "Mean blood pressure readings and fasting plasma glucose.",
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    systolic = number_field(
                        "systolic", "Mean systolic BP (mmHg)", "e.g. 115.0"
                    )

                with c2:
                    diastolic = number_field(
                        "diastolic", "Mean diastolic BP (mmHg)", "e.g. 75.0"
                    )

                with c3:
                    fasting_glucose = number_field(
                        "fasting_glucose", "Fasting glucose (mg/dL)",
                        "e.g. 88.0",
                    )

            # ----------------------------------------------------
            # 3. HAEMATOLOGICAL
            # ----------------------------------------------------

            with st.container(key="card_haematological"):

                section_header(
                    "III.",
                    "Haematological Parameters",
                    "Values from the complete blood count (CBC).",
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    hb = number_field(
                        "hb", "Haemoglobin, Hb (g/dL)", "e.g. 11.50"
                    )

                with c2:
                    rbc = number_field("rbc", "RBC (millions/mL)", "e.g. 4.20")

                with c3:
                    wbc = number_field("wbc", "WBC (×10³/µL)", "e.g. 9.10")

                c4, c5, c6 = st.columns(3)

                with c4:
                    mchc = number_field(
                        "mchc", "MCHC (g/dL)", "e.g. 33.00",
                        help="Mean corpuscular haemoglobin concentration.",
                    )

                with c5:
                    lymphocytes = number_field(
                        "lymphocytes", "Lymphocytes (×10³/µL)", "e.g. 2.10",
                        help="Absolute lymphocyte count.",
                    )

                with c6:
                    eosinophils = number_field(
                        "eosinophils", "Eosinophils (×10³/µL)", "e.g. 0.15",
                        help="Absolute eosinophil count.",
                    )

            confirmed = True
            if prefilled:
                confirmed = st.checkbox(
                    "I have checked the values filled from the reports "
                    "against the original reports.",
                    key=widget_key("confirm"),
                )

            _, button_col, _ = st.columns([1, 1, 1])

            with button_col:
                predict = st.form_submit_button(
                    "Assess GDM Status",
                    type="primary",
                    width="stretch",
                )

    with side_col:

        if not predict:
            html(
                '<div class="gn-result pending">'
                '<div class="gn-result-label">Assessment outcome</div>'
                '<div class="gn-result-text">The outcome will appear here '
                "once the findings are submitted.</div></div>"
            )

        if predict:

            values = {
                "Maternal age": age,
                "Gravidity": gravidity,
                "Family History of Diabetes": 1 if family_history == "Yes" else 0,
                "Mean Diastolic BP": diastolic,
                "Mean Systolic BP": systolic,
                "Fasting Glucose (mg/dl)": fasting_glucose,
                "Hb (g/dl)": hb,
                "RBC (millions/ml)": rbc,
                "WBC (10^3/uL)": wbc,
                "MCHC (g/dl)": mchc,
                "Lymphocytes (Absolute count 10^3/uL)": lymphocytes,
                "Eosinophils (Absolute count 10^3/uL)": eosinophils,
                "BMI": bmi,
                "Blood Type": blood_type,
            }

            missing_values = [
                name for name, value in values.items()
                if value is None
            ]

            # Family history is encoded before the check, so test it directly.
            if family_history is None:
                missing_values.insert(2, "Family History of Diabetes")

            if not confirmed:

                st.warning(
                    "Please check the values filled from the reports and "
                    "tick the confirmation box before the assessment."
                )

            elif missing_values:

                st.warning(
                    "Please complete the following findings before the "
                    "assessment: " + ", ".join(missing_values) + "."
                )

            else:

                try:

                    patient = pd.DataFrame([values])
                    model_input = preprocess_for_model(patient)

                    prediction = int(model.predict(model_input)[0])

                    st.markdown(
                        '<div class="gn-h">Assessment Outcome</div>',
                        unsafe_allow_html=True,
                    )

                    if prediction == 1:
                        css_class, verdict, sentence = (
                            "positive",
                            "YES",
                            "The entered maternal clinical profile is "
                            "<b>predicted to be positive</b> for Gestational "
                            "Diabetes Mellitus.",
                        )
                    else:
                        css_class, verdict, sentence = (
                            "negative",
                            "NO",
                            "The entered maternal clinical profile is "
                            "<b>predicted to be negative</b> for Gestational "
                            "Diabetes Mellitus.",
                        )

                    st.markdown(
                        f"""
                        <div class="gn-result {css_class}">
                            <div class="gn-result-label">GDM predicted</div>
                            <div class="gn-result-value">{verdict}</div>
                            <div class="gn-result-text">{sentence}</div>
                            <div class="gn-note">
                                Research use only. This output should not
                                replace professional clinical diagnosis or
                                medical decision-making.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("Recorded clinical findings"):

                        show = patient.T.reset_index()
                        show.columns = ["Clinical parameter", "Value"]
                        show.loc[
                            show["Clinical parameter"] == "Family History of Diabetes",
                            "Value",
                        ] = family_history
                        show["Value"] = show["Value"].astype(str)

                        st.dataframe(
                            show,
                            hide_index=True,
                            width="stretch",
                            height=35 * (len(show) + 1) + 3,
                        )

                    # ------------------------------------------------
                    # DOWNLOAD
                    # ------------------------------------------------

                    result = patient.copy()
                    result["GDM Prediction"] = (
                        "Yes" if prediction == 1 else "No"
                    )
                    result["Entry method"] = (
                        "Filled from reports, checked by user"
                        if prefilled else "Manual entry"
                    )

                    st.download_button(
                        "Download Assessment Report (CSV)",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="Glunova_AI_GDM_Prediction.csv",
                        mime="text/csv",
                    )

                except Exception as e:

                    st.error(
                        "An error occurred during the assessment: "
                        + str(e)
                    )

        with st.container(key="card_guidance"):
            st.markdown(
                """
                <div class="gn-section-title">How to use</div>
                <ol class="gn-list">
                    <li>Optionally, upload the patient's reports and select
                        <b>Read reports</b> to fill the form.</li>
                    <li>Record or check all 14 antenatal findings in
                        sections I&ndash;III.</li>
                    <li>Use the units shown next to each field.</li>
                    <li>Select <b>Assess GDM Status</b> to obtain the
                        outcome.</li>
                    <li>Download the report for the patient record.</li>
                </ol>
                <div class="gn-section-title">Output</div>
                <p class="gn-side-text">
                    <b>YES</b> &mdash; GDM predicted<br>
                    <b>NO</b> &mdash; GDM not predicted
                </p>
                <div class="gn-section-title">Model</div>
                <p class="gn-side-text">
                    Random Forest model developed on routinely
                    collected antenatal data. To assess several mothers
                    at once, use <b>Cohort Assessment</b>.
                </p>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE : BATCH PREDICTION
# ============================================================

def render_batch():
    page_header(
        "Cohort Assessment",
        "Assess GDM status for an antenatal clinic list or study cohort "
        "from a single spreadsheet (CSV).",
    )

    # Exact columns accepted for the upload.
    batch_template = pd.DataFrame(
        columns=CLINICAL_FEATURES
    )

    step1, step2 = st.columns([1, 1.4], gap="large")

    with step1:
        with st.container(key="card_step1"):
            st.markdown(
                """
                <div class="gn-step">Step 1</div>
                <div class="gn-step-title">Download the data collection template</div>
                <div class="gn-step-text">
                    A spreadsheet (CSV) with one column for each of the
                    14 antenatal parameters. Enter one mother per row.
                    Blood Type: A, B, AB or O. Family History of
                    Diabetes: 1 (yes) or 0 (no).
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.download_button(
                label="Download Template",
                data=batch_template.to_csv(index=False).encode("utf-8"),
                file_name="Glunova_AI_Template.csv",
                mime="text/csv",
            )

    with step2:
        with st.container(key="card_step2"):
            st.markdown(
                """
                <div class="gn-step">Step 2</div>
                <div class="gn-step-title">Upload the completed spreadsheet</div>
                """,
                unsafe_allow_html=True,
            )

            uploaded = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key="batch_upload",
                label_visibility="collapsed",
            )

    if uploaded is not None:

        try:

            data = pd.read_csv(uploaded)
            data.columns = data.columns.str.strip()

            missing = [
                col for col in CLINICAL_FEATURES
                if col not in data.columns
            ]

            if missing:

                st.error(
                    "The uploaded spreadsheet is missing these "
                    "clinical parameters: " + ", ".join(missing)
                )

            else:

                model_input = preprocess_for_model(
                    data[CLINICAL_FEATURES]
                )

                pred = model.predict(model_input)

                results = data.copy()

                results["GDM Prediction"] = [
                    "Yes" if int(x) == 1 else "No"
                    for x in pred
                ]

                st.markdown(
                    '<div class="gn-h">Cohort Outcome</div>',
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric("Mothers assessed", len(results))

                with m2:
                    st.metric(
                        "GDM predicted: YES",
                        int((pred == 1).sum()),
                    )

                with m3:
                    st.metric(
                        "GDM predicted: NO",
                        int((pred == 0).sum()),
                    )

                st.dataframe(
                    results,
                    width="stretch",
                )

                st.download_button(
                    "Download Cohort Report (CSV)",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="Glunova_AI_GDM_Predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:

            st.error(
                "Unable to process the uploaded spreadsheet: " + str(e)
            )


# ============================================================
# PAGE : ABOUT
# ============================================================

def render_about():
    page_header(
        "About Glunova AI",
        "Purpose, methodology and clinical data dictionary.",
    )

    text_col, spec_col = st.columns([1.6, 1], gap="large")

    with text_col:
        html(
            '<div class="gn-prose">'
            '<div class="gn-h" style="margin-top:0.4rem">Purpose</div>'
            "<p><b>Glunova AI</b> is a clinical decision-support tool for "
            "the assessment of <b>Gestational Diabetes Mellitus (GDM)</b>. "
            "It evaluates the maternal demographic and obstetric history, "
            "anthropometry, blood pressure, fasting plasma glucose and "
            "complete blood count recorded during antenatal care.</p>"
            '<div class="gn-h">Assessment outcome</div>'
            "<p>For each mother the tool reports one of two outcomes: "
            "<b>YES</b> (GDM predicted) or <b>NO</b> (GDM not predicted). "
            "A risk score or probability is not reported.</p>"
            '<div class="gn-h">Intended use</div>'
            "<p>Glunova AI is intended for research and educational use. "
            "It does not replace diagnosis of GDM according to established "
            "clinical criteria, or the judgement of a qualified healthcare "
            "professional.</p></div>"
        )

    with spec_col:
        with st.container(key="card_spec"):
            html(
                '<div class="gn-section-title" style="margin-bottom:0.4rem">'
                "Methodological Summary</div>"
                '<table class="gn-spec">'
                "<tr><td>Predictive model</td><td>Random Forest (ensemble "
                "of decision trees)</td></tr>"
                "<tr><td>Outcome</td><td>GDM predicted / not predicted"
                "</td></tr>"
                "<tr><td>Clinical parameters</td><td>14 routine antenatal "
                "measures</td></tr>"
                "<tr><td>Clinical domains</td><td>Demographic &amp; "
                "obstetric, clinical &amp; biochemical, haematological"
                "</td></tr>"
                "<tr><td>Incomplete records</td><td>Missing laboratory "
                "values estimated by iterative imputation</td></tr>"
                "<tr><td>Class imbalance</td><td>SMOTEENN resampling "
                "during model development</td></tr>"
                "<tr><td>Reading reports</td><td>Optional OCR on the server "
                "(Tesseract) fills the form; the user checks every value "
                "before the assessment</td></tr>"
                "</table>"
            )

    html('<div class="gn-h">Clinical Data Dictionary</div>')

    reference = pd.DataFrame(
        {
            "Clinical parameter": CLINICAL_FEATURES,
            "Domain": [
                "Demographic & obstetric",
                "Demographic & obstetric",
                "Demographic & obstetric",
                "Clinical",
                "Clinical",
                "Biochemical",
                "Haematological",
                "Haematological",
                "Haematological",
                "Haematological",
                "Haematological",
                "Haematological",
                "Anthropometric",
                "Demographic & obstetric",
            ],
            "Definition": [
                "Maternal age at assessment (years)",
                "Total number of pregnancies, including the current one",
                "Family history of diabetes (Yes/No)",
                "Mean of recorded diastolic blood pressure readings (mmHg)",
                "Mean of recorded systolic blood pressure readings (mmHg)",
                "Fasting plasma glucose (mg/dL)",
                "Haemoglobin concentration (g/dL)",
                "Red blood cell count (millions/mL)",
                "White blood cell count (x10^3/uL)",
                "Mean corpuscular haemoglobin concentration (g/dL)",
                "Absolute lymphocyte count (x10^3/uL)",
                "Absolute eosinophil count (x10^3/uL)",
                "Body mass index (kg/m2)",
                "ABO blood group (A, B, AB, O)",
            ],
        }
    )

    st.dataframe(
        reference,
        hide_index=True,
        width="stretch",
        height=35 * (len(reference) + 1) + 3,
    )


# ============================================================
# RENDER
# ============================================================

st.markdown(
    CUSTOM_CSS.replace("</style>", slideshow_keyframes() + "\n</style>"),
    unsafe_allow_html=True,
)

render_nav()

{
    "home": render_home,
    "predict": render_predict,
    "batch": render_batch,
    "about": render_about,
    "team": render_team,
}[page]()

render_footer()
