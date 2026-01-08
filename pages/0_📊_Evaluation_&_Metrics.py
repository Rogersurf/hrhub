import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB – Evaluation & Metrics",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_PATH, "data", "v3", "processed")

CAND_EMB_PATH = os.path.join(DATA_PATH, "candidate_embeddings.npy")
COMP_EMB_PATH = os.path.join(DATA_PATH, "company_embeddings.npy")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_resource
def load_embeddings():
    return (
        np.load(CAND_EMB_PATH),
        np.load(COMP_EMB_PATH),
    )

candidate_embeddings, company_embeddings = load_embeddings()

# =========================================================
# HEADER
# =========================================================
st.title("📊 Evaluation & Metrics")
st.caption(
    "Quantitative validation of the bilateral semantic matching system. "
    "This page mirrors the evaluation logic described in the thesis report."
)

# =========================================================
# SECTION 1 — SCORE DISTRIBUTION
# =========================================================
st.markdown("---")
st.header("1️⃣ Score Distribution (Global)")

st.markdown("""
This histogram shows the **global distribution of cosine similarity scores**
between candidates and companies in the shared SBERT embedding space.

**Key interpretation:**
- Scores around **0.50–0.60** already indicate strong semantic alignment
- Scores above **0.70** are rare and exceptional
- Ranking is more meaningful than absolute thresholds
""")

@st.cache_data(show_spinner=False)
def compute_score_distribution(candidate_embeddings, company_embeddings, sample_size=300):
    scores = []
    n = min(sample_size, len(candidate_embeddings))

    for i in range(n):
        sims = cosine_similarity(
            candidate_embeddings[i].reshape(1, -1),
            company_embeddings
        )[0]
        scores.extend(sims)

    return np.array(scores)

with st.spinner("Computing score distribution..."):
    score_dist = compute_score_distribution(
        candidate_embeddings,
        company_embeddings
    )

hist = pd.Series(score_dist).value_counts(bins=40).sort_index()
st.bar_chart(hist)

c1, c2, c3 = st.columns(3)
c1.metric("Mean Score", f"{score_dist.mean():.3f}")
c2.metric("95th Percentile", f"{np.percentile(score_dist, 95):.3f}")
c3.metric("Max Observed", f"{score_dist.max():.3f}")

# =========================================================
# SECTION 2 — BILATERAL FAIRNESS
# =========================================================
st.markdown("---")
st.header("2️⃣ Bilateral Fairness")

st.markdown("""
Bilateral fairness evaluates whether the system behaves symmetrically
for **Candidate → Company** and **Company → Candidate** retrieval.

This is **not demographic fairness**, but **structural fairness** of the algorithm.
""")

@st.cache_data(show_spinner=False)
def compute_bilateral_fairness(candidate_embeddings, company_embeddings, top_k=10, sample_size=150):
    n_cand = min(sample_size, len(candidate_embeddings))
    n_comp = min(sample_size, len(company_embeddings))

    cand_scores, comp_scores = [], []

    for i in range(n_cand):
        sims = cosine_similarity(
            candidate_embeddings[i].reshape(1, -1),
            company_embeddings
        )[0]
        cand_scores.extend(np.sort(sims)[-top_k:])

    for j in range(n_comp):
        sims = cosine_similarity(
            company_embeddings[j].reshape(1, -1),
            candidate_embeddings[:n_cand]
        )[0]
        comp_scores.extend(np.sort(sims)[-top_k:])

    c_mean = float(np.mean(cand_scores))
    co_mean = float(np.mean(comp_scores))
    fairness = min(c_mean, co_mean) / max(c_mean, co_mean)

    return c_mean, co_mean, fairness

with st.spinner("Computing bilateral fairness..."):
    cand_mean, comp_mean, fairness = compute_bilateral_fairness(
        candidate_embeddings,
        company_embeddings
    )

f1, f2, f3 = st.columns(3)
f1.metric("Candidate → Company", f"{cand_mean:.3f}")
f2.metric("Company → Candidate", f"{comp_mean:.3f}")
f3.metric("Fairness Ratio", f"{fairness:.3f}")

if fairness >= 0.9:
    st.success("System is highly balanced")
elif fairness >= 0.6:
    st.info("System shows expected balance for Top-K retrieval")
else:
    st.warning("Potential asymmetry detected")

# =========================================================
# SECTION 3 — COVERAGE & SCALE
# =========================================================
st.markdown("---")
st.header("3️⃣ Coverage & Scale")

st.markdown("""
The enrichment pipeline leverages job postings as a **vocabulary bridge**
between candidates and companies.

This enables:
- semantic alignment
- high coverage
- scalable matching
""")

st.markdown("""
**Reported system scale (from preprocessing stage):**
- Candidates: **9,544**
- Companies: **24,473**
- Job postings: **123,849**
- Coverage after enrichment: **96.1%**
""")

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "All metrics shown here are derived from the same embeddings and matching "
    "logic used in the Candidate and Company views. No additional models are applied."
)
