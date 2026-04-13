import streamlit as st
import os
import json
from huggingface_hub import snapshot_download

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB – Baseline Comparison",
    page_icon="🧪",
    layout="centered"
)

# =========================================================
# LOAD DATASET (HF WAY)
# =========================================================
@st.cache_resource
def load_artifacts():
    return snapshot_download(
        repo_id="Rogersurf/hrhub-artifacts",
        repo_type="dataset"
    )

DATASET_PATH = load_artifacts()
RESULTS_PATH = os.path.join(DATASET_PATH, "results")

BASELINE_JSON = os.path.join(RESULTS_PATH, "baseline_results_summary.json")
BASELINE_PNG = os.path.join(RESULTS_PATH, "baseline_comparison_all_methods.png")

# =========================================================
# HEADER
# =========================================================
st.title("🧪 Baseline Comparison")
st.caption(
    "Comparison between semantic embeddings (SBERT) and classical baselines. "
    "This section justifies the model choice used throughout the system."
)

# =========================================================
# SECTION 1 — WHY BASELINES
# =========================================================
st.markdown("""
### Why compare against baselines?

In applied machine learning, new methods must be validated against
**simpler and well-understood alternatives**.

We compare SBERT against:
- **TF-IDF + Cosine Similarity** (lexical baseline)
- **Keyword Overlap (Jaccard)** (set-based baseline)
""")

# =========================================================
# SECTION 2 — QUANTITATIVE SUMMARY
# =========================================================
st.markdown("---")
st.subheader("📊 Quantitative Comparison")

if os.path.exists(BASELINE_JSON):
    with open(BASELINE_JSON, "r") as f:
        metrics = json.load(f)

    for k, v in metrics.items():
        if isinstance(v, (int, float, str)):
            st.metric(k.replace("_", " ").title(), v)
        else:
            st.markdown(f"**{k.replace('_', ' ').title()}**")
            st.json(v)

else:
    st.info(
        "Baseline metrics were computed offline. "
        "Refer to the thesis report for full tables."
    )

# =========================================================
# SECTION 3 — VISUAL COMPARISON
# =========================================================
st.markdown("---")
st.subheader("📈 Visual Comparison")

if os.path.exists(BASELINE_PNG):
    st.image(
        BASELINE_PNG,
        caption="Performance comparison across TF-IDF, Jaccard, and SBERT",
        use_column_width=True,
        width=900
    )
else:
    st.warning("Baseline comparison figure not found in dataset.")

# =========================================================
# SECTION 4 — INTERPRETATION
# =========================================================
st.markdown("""
### Interpretation of Results

- **TF-IDF** performs well for exact term overlap but fails under
  vocabulary mismatch.
- **Jaccard similarity** is brittle and insensitive to context.
- **SBERT** achieves higher alignment and more stable rankings.

Stable ranking quality is critical in retrieval-based systems, where
relative order matters more than absolute score values.
""")

# =========================================================
# SECTION 5 — DESIGN DECISION
# =========================================================
st.markdown("""
### Final Model Choice

Based on empirical performance, robustness, and scalability,
**SBERT was selected as the production model**.

Classical baselines are retained as references for validation
and interpretability.
""")

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "All baseline results shown here were computed offline and "
    "loaded from the frozen HRHUB artifacts dataset."
)
