import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# FAIRNESS FUNCTION
# =========================================================
def compute_bilateral_fairness(
    candidate_embeddings,
    company_embeddings,
    top_k=10,
    sample_size=100
):
    n_cand = min(sample_size, len(candidate_embeddings))
    n_comp = min(sample_size, len(company_embeddings))

    cand_scores = []
    comp_scores = []

    # Candidate → Company
    for i in range(n_cand):
        sims = cosine_similarity(
            candidate_embeddings[i].reshape(1, -1),
            company_embeddings
        )[0]
        top = np.sort(sims)[-top_k:]
        cand_scores.extend(top)

    # Company → Candidate
    for j in range(n_comp):
        sims = cosine_similarity(
            company_embeddings[j].reshape(1, -1),
            candidate_embeddings[:n_cand]
        )[0]
        top = np.sort(sims)[-top_k:]
        comp_scores.extend(top)

    cand_mean = float(np.mean(cand_scores))
    comp_mean = float(np.mean(comp_scores))

    fairness = min(cand_mean, comp_mean) / max(cand_mean, comp_mean)

    return cand_mean, comp_mean, fairness


@st.cache_data(show_spinner=False)
def cached_fairness(candidate_embeddings, company_embeddings, top_k):
    return compute_bilateral_fairness(
        candidate_embeddings,
        company_embeddings,
        top_k=top_k,
        sample_size=100
    )

# =========================================================
# COMPUTES SCORE DISTRIBUTION
# =========================================================
@st.cache_data(show_spinner=False)
def compute_score_distribution(
    candidate_embeddings,
    company_embeddings,
    sample_size=200
):
    """
    Compute a global score distribution using random candidate samples
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    n = min(sample_size, len(candidate_embeddings))
    scores = []

    for i in range(n):
        sims = cosine_similarity(
            candidate_embeddings[i].reshape(1, -1),
            company_embeddings
        )[0]
        scores.extend(sims)

    return np.array(scores)

# =========================================================
# BUILD NETWORK GRAPH
# =========================================================
@st.cache_data(show_spinner=False)
def build_network_graph(
    candidate_embeddings,
    company_embeddings,
    candidates_meta,
    companies_meta,
    top_k=3,
    sample_size=15
):
    from pyvis.network import Network
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50"
    )

    n_cand = min(sample_size, len(candidate_embeddings))

    # Add candidate nodes
    for i in range(n_cand):
        label = f"Candidate {i}"
        net.add_node(
            f"cand_{i}",
            label=label,
            color="#667eea",
            shape="dot",
            size=18
        )

    # Add company nodes + edges
    for i in range(n_cand):
        sims = cosine_similarity(
            candidate_embeddings[i].reshape(1, -1),
            company_embeddings
        )[0]

        top_idx = np.argsort(sims)[-top_k:][::-1]

        for j in top_idx:
            company_name = companies_meta.iloc[j].get("name", f"Company {j}")

            net.add_node(
                f"comp_{j}",
                label=company_name,
                color="#2ecc71",
                shape="box",
                size=14
            )

            net.add_edge(
                f"cand_{i}",
                f"comp_{j}",
                value=float(sims[j]),
                title=f"Score: {sims[j]:.3f}"
            )

    return net

# =========================================================
# LLM-BASED MATCH EXPLANATION
# =========================================================
def explain_match_llm(candidate_row, company_row, score):
    """
    Post-hoc LLM explanation for a single match.
    Safe: does NOT affect ranking.
    """
    import os

    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        return {
            "summary": "LLM not enabled (no HF_TOKEN set).",
            "strengths": [],
            "gaps": [],
            "recommendation": "Enable LLM for detailed explanation."
        }

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=HF_TOKEN)

        prompt = f"""
You are an HR analyst.

Explain why the following candidate matches the company.

CANDIDATE:
Category: {candidate_row.get('Category','')}
Skills: {candidate_row.get('skills','')}
Objective: {candidate_row.get('career_objective','')}

COMPANY:
Name: {company_row.get('name','')}
Industry: {company_row.get('industries_list','')}
Required Skills: {company_row.get('required_skills','')}

MATCH SCORE: {score:.3f}

Return a concise explanation in JSON with keys:
- strengths (list)
- gaps (list)
- recommendation (string)
- summary (string)
"""

        response = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )

        content = response.choices[0].message.content

        import json
        start, end = content.find("{"), content.rfind("}") + 1
        return json.loads(content[start:end])

    except Exception as e:
        return {
            "summary": f"LLM error: {str(e)}",
            "strengths": [],
            "gaps": [],
            "recommendation": "Review manually."
        }


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB - Candidate View",
    page_icon="👤",
    layout="wide"
)

# =========================================================
# PATHS (V3 = REPORT CONSISTENT)
# =========================================================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_PATH, "data", "v3", "processed")

CAND_EMB_PATH = os.path.join(DATA_PATH, "candidate_embeddings.npy")
COMP_EMB_PATH = os.path.join(DATA_PATH, "company_embeddings.npy")
CAND_META_PATH = os.path.join(DATA_PATH, "candidates_metadata.pkl")
COMP_META_PATH = os.path.join(DATA_PATH, "companies_metadata.pkl")

# =========================================================
# LOAD CORE DATA
# =========================================================
@st.cache_resource
def load_core():
    return (
        np.load(CAND_EMB_PATH),
        np.load(COMP_EMB_PATH),
        pickle.load(open(CAND_META_PATH, "rb")),
        pickle.load(open(COMP_META_PATH, "rb")),
    )

candidate_embeddings, company_embeddings, candidates_meta, companies_meta = load_core()

# =========================================================
# HEADER
# =========================================================
st.title("👤 HRHUB – Candidate View")
st.caption("Semantic matching in a shared SBERT embedding space")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")

    candidate_id = st.number_input(
        "Candidate ID",
        min_value=0,
        max_value=len(candidate_embeddings) - 1,
        value=0,
        step=1
    )

    top_k = st.slider("Top-K Companies", 3, 20, 10)
    threshold = st.slider("Highlight score >", 0.4, 0.8, 0.6)

# =========================================================
# CANDIDATE PROFILE
# =========================================================
candidate = candidates_meta.iloc[candidate_id]

left, right = st.columns([1, 2])

with left:
    st.subheader("👤 Candidate Profile")

    st.markdown(f"**Category:** {candidate.get('Category', 'N/A')}")

    with st.expander("🧠 Skills", expanded=True):
        st.write(candidate.get("skills", "N/A"))

    with st.expander("🎯 Career Objective", expanded=True):
        st.write(candidate.get("career_objective", "N/A"))

# =========================================================
# MATCHING (REAL)
# =========================================================
cand_vec = candidate_embeddings[candidate_id].reshape(1, -1)
scores = cosine_similarity(cand_vec, company_embeddings)[0]

top_idx = np.argsort(scores)[-top_k:][::-1]
top_scores = scores[top_idx]

rows = []
for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
    company = companies_meta.iloc[idx]
    rows.append({
        "Rank": rank,
        "Company": company.get("name", "Unknown"),
        "Industry": company.get("industries_list", "N/A"),
        "Score": score
    })

df = pd.DataFrame(rows)

# =========================================================
# MATCH METRICS + TABLE
# =========================================================
with right:
    st.subheader("📊 Match Overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Best Match", f"{df.Score.max():.3f}")
    m2.metric("Average Score", f"{df.Score.mean():.3f}")
    m3.metric("Strong Matches", (df.Score > threshold).sum())

    st.subheader("🏢 Top Company Matches")

    def style_score(val):
        if val > threshold:
            return "color: green; font-weight: bold;"
        return ""

    st.dataframe(
        df.style.applymap(style_score, subset=["Score"]),
        use_container_width=True
    )

# =========================================================
# FAIRNESS PANEL
# =========================================================
st.markdown("---")
st.subheader("⚖️ Bilateral Fairness (Top-K)")

with st.expander("What does this mean?"):
    st.markdown("""
    **Bilateral Fairness** evaluates whether the system treats
    candidates and companies symmetrically.

    - Candidate → Company: mean Top-K similarity
    - Company → Candidate: mean Top-K similarity

    Values near **1.0** indicate a balanced system.
    Lower values are expected in retrieval-based systems.
    """)

with st.spinner("Computing fairness metrics..."):
    cand_mean, comp_mean, fairness = cached_fairness(
        candidate_embeddings,
        company_embeddings,
        top_k
    )

c1, c2, c3 = st.columns(3)
c1.metric("Candidate → Company", f"{cand_mean:.3f}")
c2.metric("Company → Candidate", f"{comp_mean:.3f}")
c3.metric("Fairness Ratio", f"{fairness:.3f}")

if fairness >= 0.9:
    st.success("✅ System is highly balanced")
elif fairness >= 0.6:
    st.info("ℹ️ System is reasonably balanced (expected for Top-K)")
else:
    st.warning("⚠️ Potential asymmetry detected")

# =========================================================
# SCORE DISTRIBUTION
# =========================================================
st.markdown("---")
st.subheader("📈 Score Distribution")

with st.expander("How to interpret this?", expanded=False):
    st.markdown("""
    This histogram shows the **distribution of cosine similarity scores**
    between candidates and companies.

    **Important interpretation:**
    - Scores above **0.6** are already considered **strong semantic matches**
    - Scores above **0.7** are **rare and exceptional**
    - The system is evaluated by **ranking**, not absolute thresholds
    """)

with st.spinner("Computing score distribution..."):
    score_dist = compute_score_distribution(
        candidate_embeddings,
        company_embeddings,
        sample_size=200
    )

# Histogram
hist_df = pd.DataFrame({"Similarity Score": score_dist})

st.bar_chart(
    hist_df["Similarity Score"].value_counts(bins=30).sort_index()
)

# Reference lines (textual)
c1, c2, c3 = st.columns(3)
c1.metric("Mean Score", f"{score_dist.mean():.3f}")
c2.metric("95th Percentile", f"{np.percentile(score_dist, 95):.3f}")
c3.metric("Max Observed", f"{score_dist.max():.3f}")

# =========================================================
# NETWORK GRAPH
# =========================================================
st.markdown("---")
st.subheader("🌐 Matching Network Graph")

with st.expander("What does this show?", expanded=False):
    st.markdown("""
    This network visualizes the **Top-K semantic relationships**
    between candidates and companies.

    - 🔵 Blue nodes: Candidates
    - 🟢 Green nodes: Companies
    - Edges represent strong semantic matches

    The graph helps detect:
    - Structural bias
    - Over-dominant companies
    - Diversity of matches
    """)

with st.spinner("Building network graph..."):
    net = build_network_graph(
        candidate_embeddings,
        company_embeddings,
        candidates_meta,
        companies_meta,
        top_k=3,
        sample_size=12
    )

html_path = os.path.join(BASE_PATH, "data", "v3", "results", "network_temp.html")
net.write_html(html_path)

import streamlit.components.v1 as components
components.html(
    open(html_path, "r").read(),
    height=620,
    scrolling=True
)

# =========================================================
# LLM EXPLAINABILITY (TOP-1)
# =========================================================
st.markdown("---")
st.subheader("🤖 Match Explanation (LLM)")

with st.expander("Why is this company a good match?", expanded=True):
    top_company_idx = top_idx[0]
    top_company = companies_meta.iloc[top_company_idx]
    top_score = top_scores[0]

    if st.button("Generate AI Explanation"):
        with st.spinner("LLM analyzing match..."):
            explanation = explain_match_llm(
                candidate,
                top_company,
                top_score
            )

        st.markdown(f"**Summary:** {explanation.get('summary','')}")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### ✅ Strengths")
            for s in explanation.get("strengths", []):
                st.write(f"- {s}")

        with c2:
            st.markdown("### ⚠️ Gaps")
            for g in explanation.get("gaps", []):
                st.write(f"- {g}")

        st.markdown(
            f"### 🧭 Recommendation\n**{explanation.get('recommendation','')}**"
        )

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "ℹ️ Scores are cosine similarity. In SBERT embedding spaces, "
    "values around 0.55–0.60 already indicate strong alignment. "
    "Ranking matters more than absolute values."
)
