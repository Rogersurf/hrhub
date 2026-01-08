import streamlit as st
import numpy as np
import pandas as pd
import os
import json

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB – Baseline Comparison",
    page_icon="🧪",
    layout="centered"
)

# =========================================================
# PATHS
# =========================================================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_PATH = os.path.join(BASE_PATH, "data", "evaluation")

BASELINE_JSON = os.path.join(EVAL_PATH, "baseline_results_summary.json")
BASELINE_PNG = os.path.join(EVAL_PATH, "baseline_comparison_all_methods.png")

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

These baselines represent what a production system *could* use
if semantic models did not provide sufficient gains.
""")

# =========================================================
# SECTION 2 — LOAD RESULTS
# =========================================================
if not os.path.exists(BASELINE_JSON):
    st.warning("Baseline evaluation file not found. Showing conceptual comparison only.")
else:
    with open(BASELINE_JSON, "r") as f:
        baseline_results = json.load(f)

    df = pd.DataFrame(baseline_results)

    st.subheader("📊 Quantitative Comparison")
    st.dataframe(df)

# =========================================================
# SECTION 3 — VISUAL COMPARISON
# =========================================================
st.markdown("---")
st.subheader("📈 Visual Comparison")

if os.path.exists(BASELINE_PNG):
    st.image(
        BASELINE_PNG,
        caption="Performance comparison across TF-IDF, Jaccard, and SBERT",
    )
else:
    st.info("Visualization not available. See notebook for full plots.")

# =========================================================
# SECTION 4 — INTERPRETATION
# =========================================================
st.markdown("""
### Interpretation of Results

Key observations from the comparison:

- **TF-IDF** performs well for exact term overlap but fails under
  vocabulary mismatch (e.g., synonyms, paraphrases).
- **Jaccard similarity** is highly brittle and insensitive to context.
- **SBERT** consistently achieves higher alignment scores and
  smoother score distributions.

Most importantly, SBERT produces **stable rankings**, which is critical
in retrieval-based systems where relative order matters more than
absolute values.
""")

# =========================================================
# SECTION 5 — DESIGN DECISION
# =========================================================
st.markdown("""
### Final Model Choice

Based on:
- Quantitative improvement over baselines
- Robustness to vocabulary mismatch
- Scalability with pre-computed embeddings
- Compatibility with bilateral matching

**SBERT was selected as the production model**, while classical methods
remain as reference baselines for validation and interpretability.
""")

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "Baseline results shown here are derived from controlled experiments "
    "and synthetic validation cases described in the thesis report."
)
