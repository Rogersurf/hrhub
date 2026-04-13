import streamlit as st
import os
import json
from huggingface_hub import snapshot_download


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB – Evaluation & Metrics",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# DOWNLOAD DATASET (HF CORRETO)
# =========================================================
@st.cache_resource
def load_artifacts():
    return snapshot_download(
        repo_id="Rogersurf/hrhub-artifacts",
        repo_type="dataset"
    )

DATASET_PATH = load_artifacts()
RESULTS_PATH = os.path.join(DATASET_PATH, "results")

def show_image(filename):
    path = os.path.join(RESULTS_PATH, filename)
    if os.path.exists(path):
        st.image(
            path,
            use_column_width=True,
            width=900)
    else:
        st.warning(f"{filename} not found in dataset.")


# =========================================================
# HEADER
# =========================================================
st.title("📊 Evaluation & Metrics")
st.caption(
    "Frozen quantitative evaluation of the bilateral semantic matching system. "
    "All results shown here were computed offline and loaded from the dataset."
)

# =========================================================
# SECTION 1 — SCORE DISTRIBUTION
# =========================================================
st.markdown("---")
st.header("1️⃣ Score Distribution")

st.markdown("""
Global cosine similarity distribution between candidates and companies
in the shared SBERT embedding space.

**Interpretation:**
- Scores around **0.55–0.60** already indicate strong semantic alignment
- Absolute scores are less important than ranking
""")

score_fig = os.path.join(RESULTS_PATH, "score_distribution.png")
if os.path.exists(score_fig):
    st.image(score_fig,
        use_column_width=True,
        width=900)
else:
    st.warning("Score distribution figure not found in dataset.")

# =========================================================
# SECTION 2 — BASELINE COMPARISON
# =========================================================
st.markdown("---")
st.header("2️⃣ Baseline Comparison")

baseline_fig = os.path.join(RESULTS_PATH, "baseline_comparison_all_methods.png")
if os.path.exists(baseline_fig):
    st.image(baseline_fig,
            use_column_width=True,
            width=900)
else:
    st.warning("Baseline comparison figure not found in dataset.")

# =========================================================
# SECTION 3 — BILATERAL FAIRNESS
# =========================================================
st.markdown("---")
st.header("3️⃣ Bilateral Fairness")

fairness_json = os.path.join(RESULTS_PATH, "evaluation_metrics.json")

if os.path.exists(fairness_json):
    with open(fairness_json, "r") as f:
        metrics = json.load(f)

    st.metric(
        "Fairness Ratio",
        f"{metrics.get('bilateral_fairness', 'N/A')}"
    )
else:
    st.info(
        "Bilateral fairness was computed offline. "
        "Refer to the thesis report for full analysis."
    )

# =========================================================
# SECTION 4 — COVERAGE & SCALE
# =========================================================
st.markdown("---")
st.header("4️⃣ Coverage & Scale")

st.markdown("""
**System scale (offline preprocessing):**
- Candidates: **9,544**
- Companies: **24,473**
- Job postings: **123,849**
- Vocabulary coverage after enrichment: **96.1%**
""")

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "This page performs no recomputation. All metrics are loaded from "
    "the frozen HRHUB artifacts dataset."
)
