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
# SCORE DISTRIBUTION
# =========================================================
@st.cache_data(show_spinner=False)
def compute_score_distribution(
    candidate_embeddings,
    company_embeddings,
    sample_size=200
):
    n = min(sample_size, len(company_embeddings))
    scores = []

    for i in range(n):
        sims = cosine_similarity(
            company_embeddings[i].reshape(1, -1),
            candidate_embeddings
        )[0]
        scores.extend(sims)

    return np.array(scores)

# =========================================================
# NETWORK GRAPH
# =========================================================
@st.cache_resource(show_spinner=False)
def build_network_graph(
    company_embeddings,
    candidate_embeddings,
    companies_meta,
    candidates_meta,
    top_k=3,
    sample_size=15
):
    from pyvis.network import Network

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50"
    )

    n_comp = min(sample_size, len(company_embeddings))

    # Company nodes
    for i in range(n_comp):
        label = companies_meta.iloc[i].get("name", f"Company {i}")
        net.add_node(
            f"comp_{i}",
            label=label,
            color="#2ecc71",
            shape="box",
            size=18
        )

    # Candidate nodes + edges
    for i in range(n_comp):
        sims = cosine_similarity(
            company_embeddings[i].reshape(1, -1),
            candidate_embeddings
        )[0]

        top_idx = np.argsort(sims)[-top_k:][::-1]

        for j in top_idx:
            net.add_node(
                f"cand_{j}",
                label=f"Candidate {j}",
                color="#667eea",
                shape="dot",
                size=14
            )

            net.add_edge(
                f"comp_{i}",
                f"cand_{j}",
                value=float(sims[j]),
                title=f"Score: {sims[j]:.3f}"
            )

    return net

# =========================================================
# LLM EXPLANATION
# =========================================================
def explain_match_llm(company_row, candidate_row, score):
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
        import json

        client = InferenceClient(token=HF_TOKEN)

        prompt = f"""
You are an HR analyst.

Explain why the following candidate matches the company.

COMPANY:
Name: {company_row.get('name','')}
Industry: {company_row.get('industries_list','')}
Required Skills: {company_row.get('required_skills','')}

CANDIDATE:
Category: {candidate_row.get('Category','')}
Skills: {candidate_row.get('skills','')}
Objective: {candidate_row.get('career_objective','')}

MATCH SCORE: {score:.3f}

Return a concise explanation in JSON with keys:
- strengths
- gaps
- recommendation
- summary
"""

        response = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )

        content = response.choices[0].message.content
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
    page_title="HRHUB - Company View",
    page_icon="🏢",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_PATH, "data", "v3", "processed")

CAND_EMB_PATH = os.path.join(DATA_PATH, "candidate_embeddings.npy")
COMP_EMB_PATH = os.path.join(DATA_PATH, "company_embeddings.npy")
CAND_META_PATH = os.path.join(DATA_PATH, "candidates_metadata.pkl")
COMP_META_PATH = os.path.join(DATA_PATH, "companies_metadata.pkl")

# =========================================================
# LOAD DATA
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
st.title("🏢 HRHUB – Company View")
st.caption("Semantic matching in a shared SBERT embedding space")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")

    company_id = st.number_input(
        "Company ID",
        min_value=0,
        max_value=len(company_embeddings) - 1,
        value=0,
        step=1
    )

    top_k = st.slider("Top-K Candidates", 3, 20, 10)
    threshold = st.slider("Highlight score >", 0.4, 0.8, 0.6)

# =========================================================
# COMPANY PROFILE
# =========================================================
company = companies_meta.iloc[company_id]

left, right = st.columns([1, 2])

with left:
    st.subheader("🏢 Company Profile")

    st.markdown(f"**Name:** {company.get('name','Unknown')}")

    with st.expander("🏭 Industry", expanded=True):
        st.write(company.get("industries_list","N/A"))

    with st.expander("🧠 Required Skills", expanded=True):
        st.write(company.get("required_skills","N/A"))

# =========================================================
# MATCHING
# =========================================================
comp_vec = company_embeddings[company_id].reshape(1, -1)
scores = cosine_similarity(comp_vec, candidate_embeddings)[0]

top_idx = np.argsort(scores)[-top_k:][::-1]
top_scores = scores[top_idx]

rows = []
for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
    candidate = candidates_meta.iloc[idx]
    rows.append({
        "Rank": rank,
        "Category": candidate.get("Category","N/A"),
        "Score": score
    })

df = pd.DataFrame(rows)

# =========================================================
# MATCH METRICS
# =========================================================
with right:
    st.subheader("📊 Match Overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Best Match", f"{df.Score.max():.3f}")
    m2.metric("Average Score", f"{df.Score.mean():.3f}")
    m3.metric("Strong Matches", (df.Score > threshold).sum())

    st.subheader("👤 Top Candidate Matches")

    def style_score(val):
        return "color: green; font-weight: bold;" if val > threshold else ""

    st.dataframe(
        df.style.applymap(style_score, subset=["Score"]),
        use_container_width=True
    )

# =========================================================
# FAIRNESS
# =========================================================
st.markdown("---")
st.subheader("⚖️ Bilateral Fairness (Top-K)")

cand_mean, comp_mean, fairness = cached_fairness(
    candidate_embeddings,
    company_embeddings,
    top_k
)

c1, c2, c3 = st.columns(3)
c1.metric("Candidate → Company", f"{cand_mean:.3f}")
c2.metric("Company → Candidate", f"{comp_mean:.3f}")
c3.metric("Fairness Ratio", f"{fairness:.3f}")

# =========================================================
# SCORE DISTRIBUTION
# =========================================================
st.markdown("---")
st.subheader("📈 Score Distribution")

score_dist = compute_score_distribution(
    candidate_embeddings,
    company_embeddings
)

st.bar_chart(pd.Series(score_dist).value_counts(bins=30).sort_index())

# =========================================================
# NETWORK GRAPH
# =========================================================
st.markdown("---")
st.subheader("🌐 Matching Network Graph")

net = build_network_graph(
    company_embeddings,
    candidate_embeddings,
    companies_meta,
    candidates_meta
)

html_path = os.path.join(BASE_PATH, "data", "v3", "results", "network_company.html")
os.makedirs(os.path.dirname(html_path), exist_ok=True)
net.write_html(html_path)

import streamlit.components.v1 as components
components.html(open(html_path).read(), height=620, scrolling=True)

# =========================================================
# LLM EXPLANATION
# =========================================================
st.markdown("---")
st.subheader("🤖 Match Explanation (LLM)")

with st.expander("Why is this candidate a good match?", expanded=True):
    top_candidate = candidates_meta.iloc[top_idx[0]]
    top_score = top_scores[0]

    if st.button("Generate AI Explanation"):
        explanation = explain_match_llm(
            company,
            top_candidate,
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
