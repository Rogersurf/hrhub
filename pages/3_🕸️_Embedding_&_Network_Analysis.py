import streamlit as st
import streamlit.components.v1 as components
import os
from huggingface_hub import snapshot_download

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB – Embedding & Network Analysis",
    page_icon="🕸️",
    layout="wide"
)

# =========================================================
# GLOBAL CSS (LIMIT WIDTH)
# =========================================================
st.markdown("""
<style>
.viz-container {
    width: 100%;
    max-width: 1400px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

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

TSNE_HTML = os.path.join(RESULTS_PATH, "tsne_embedding_space.html")
NETWORK_HTML = os.path.join(RESULTS_PATH, "network_interactive.html")

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

Candidates and companies are embedded into a **shared semantic vector space**
using SBERT.

To make this space interpretable, a **t-SNE projection** is applied,
mapping high-dimensional embeddings into two dimensions while
preserving local neighborhood structure.
""")

if os.path.exists(TSNE_HTML):
    st.markdown("#### 🔍 t-SNE Projection (Interactive)")
    with open(TSNE_HTML, "r", encoding="utf-8") as f:
        html_code = f.read()

    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    components.html(html_code, height=650)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("t-SNE visualization not found in dataset.")

# =========================================================
# SECTION 2 — NETWORK GRAPH
# =========================================================
st.markdown("---")
st.header("2️⃣ Bipartite Matching Network")

st.markdown("""
Matching relationships can be represented as a **bipartite graph**:

- One node set represents **candidates**
- One node set represents **companies**
- Edge weights correspond to **semantic similarity scores**

This view highlights hubs, dense regions, and structural asymmetries.
""")

if os.path.exists(NETWORK_HTML):
    st.markdown("#### 🌐 Candidate–Company Network (Interactive)")
    with open(NETWORK_HTML, "r", encoding="utf-8") as f:
        html_code = f.read()

    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    components.html(html_code, height=650)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Network visualization not found in dataset.")

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
    "All visualizations shown on this page are generated offline and "
    "loaded as static HTML artifacts from the HRHUB dataset."
)
