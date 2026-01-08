import streamlit as st
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB – Embedding & Network Analysis",
    page_icon="🕸️"
)

# =========================================================
# PATHS (CORRIGIDOS)
# =========================================================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_PATH, "data")

# Visualizations (FINAL, STORYTELLING)
VIS_PATH = os.path.join(DATA_PATH, "visualizations")

# Results (v3 experimental outputs)
RES_PATH = os.path.join(DATA_PATH, "v3", "results")

TSNE_HTML = os.path.join(VIS_PATH, "tsne_embedding_space.html")
NETWORK_HTML = os.path.join(RES_PATH, "network_interactive.html")

# =========================================================
# HEADER
# =========================================================
st.title("🕸️ Embedding Space & Network Analysis")
st.caption(
    "Geometric and graph-based inspection of the shared semantic space "
    "used for bilateral candidate–company matching."
)

# =========================================================
# SECTION 1 — EMBEDDING SPACE
# =========================================================
st.markdown("""
### 1️⃣ Embedding Space Geometry

Candidates and companies are embedded into a **shared 384-dimensional
semantic vector space** using SBERT.

To make this space interpretable, a **t-SNE projection** is applied,
mapping the high-dimensional embeddings into two dimensions while
preserving local neighborhood structure.
""")

if os.path.exists(TSNE_HTML):
    st.markdown("#### 🔍 t-SNE Projection (Interactive)")
    with open(TSNE_HTML, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=550)
else:
    st.error(f"t-SNE file not found at: {TSNE_HTML}")

# =========================================================
# SECTION 2 — NETWORK GRAPH
# =========================================================
st.markdown("---")
st.header("2️⃣ Bipartite Matching Network")

st.markdown("""
The matching relationships can also be represented as a **bipartite graph**:

- One node set represents **candidates**
- One node set represents **companies**
- Edge weights correspond to **cosine similarity scores**

This view highlights hubs, dense regions, and structural asymmetries.
""")

if os.path.exists(NETWORK_HTML):
    st.markdown("#### 🌐 Candidate–Company Network (Interactive)")
    with open(NETWORK_HTML, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=550)
else:
    st.error(f"Network file not found at: {NETWORK_HTML}")

# =========================================================
# SECTION 3 — INTERPRETATION
# =========================================================
st.markdown("""
### 3️⃣ Interpretation

From the embedding and network analysis, we observe that:

- Semantically similar entities form **local geometric clusters**
- High-quality matches correspond to **short distances in embedding space**
- Network hubs often represent generalized profiles
- Sparse regions indicate niche or specialized skill sets

These observations confirm that the model learns **meaningful semantic structure**,
rather than relying on superficial keyword overlap.
""")

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "All visualizations shown on this page are generated offline during "
    "the preprocessing and evaluation stages and loaded as static HTML "
    "artifacts for reproducibility."
)
