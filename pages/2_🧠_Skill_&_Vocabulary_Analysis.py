import streamlit as st
from huggingface_hub import snapshot_download
import os
import json

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB – Skill & Vocabulary Analysis",
    page_icon="🧠",
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

# =========================================================
# HEADER
# =========================================================
st.title("🧠 Skill & Vocabulary Analysis")
st.caption(
    "How HRHUB resolves vocabulary mismatch between candidates and companies "
    "using job postings as semantic bridges."
)

# =========================================================
# SECTION 1 — THE PROBLEM
# =========================================================
st.markdown("""
### 1️⃣ The Vocabulary Mismatch Problem

Candidates and companies **describe similar skills using different language**.

Examples:
- Candidate: *Python, Pandas, Machine Learning*
- Company: *Data-driven systems, analytics pipelines, AI solutions*

Pure keyword matching fails under this mismatch.
""")

# =========================================================
# SECTION 2 — RAW VS ENRICHED (CONCEPTUAL)
# =========================================================
st.markdown("---")
st.header("2️⃣ Raw vs Enriched Skill Vocabulary")

st.markdown("""
**Before enrichment:**
- Sparse and inconsistent candidate-provided skills
- High synonym and abbreviation variance
- Severe cold-start for companies

**After enrichment (via job postings):**
- Shared vocabulary layer between candidates and companies
- Expanded skill coverage
- Consistent terminology across entities
""")

# =========================================================
# SECTION 3 — COVERAGE IMPACT (FROZEN METRICS)
# =========================================================
st.markdown("---")
st.header("3️⃣ Coverage Impact")

c1, c2, c3 = st.columns(3)
c1.metric("Candidates", "9,544")
c2.metric("Companies", "24,473")
c3.metric("Vocabulary Coverage", "96.1%")

st.markdown("""
Coverage is computed **offline during preprocessing**.
Job postings act as semantic bridges that propagate skills to companies,
dramatically reducing cold-start issues.
""")

# =========================================================
# SECTION 4 — WHY THIS MATTERS
# =========================================================
st.markdown("---")
st.header("4️⃣ Why Vocabulary Enrichment Matters")

st.markdown("""
- Improves recall without sacrificing precision
- Enables semantic embeddings to operate on richer inputs
- Stabilizes ranking quality across domains
- Essential for bilateral (candidate ↔ company) matching
""")

# =========================================================
# SECTION 5 — INTERPRETATION
# =========================================================
st.markdown("""
### 5️⃣ Interpretation

The combination of:
- vocabulary enrichment (job postings)
- semantic embeddings (SBERT)

explains the strong performance gains observed over lexical baselines.
""")

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "This page performs no recomputation. "
    "All statistics and conclusions are derived from offline preprocessing."
)
