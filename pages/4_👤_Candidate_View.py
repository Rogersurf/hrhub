import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import json
import re

from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download, InferenceClient

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB - Candidate View",
    page_icon="👤",
    layout="wide"
)

# =========================================================
# HF ARTIFACT CONFIG (UNCHANGED LOGIC, CORRECT SOURCE)
# =========================================================
DATASET_REPO = "Rogersurf/hrhub-artifacts"
PROCESSED_DIR = "processed"

# =========================================================
# LOAD DATA (HF ARTIFACTS – NO LOCAL FS)
# =========================================================
@st.cache_resource(show_spinner=True)
def load_core():
    cand_emb_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{PROCESSED_DIR}/candidate_embeddings.npy",
        repo_type="dataset"
    )
    comp_emb_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{PROCESSED_DIR}/company_embeddings.npy",
        repo_type="dataset"
    )
    cand_meta_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{PROCESSED_DIR}/candidates_metadata.pkl",
        repo_type="dataset"
    )
    comp_meta_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{PROCESSED_DIR}/companies_metadata.pkl",
        repo_type="dataset"
    )

    candidate_embeddings = np.load(cand_emb_path)
    company_embeddings = np.load(comp_emb_path)
    candidates_meta = pickle.load(open(cand_meta_path, "rb"))
    companies_meta = pickle.load(open(comp_meta_path, "rb"))

    return (
        candidate_embeddings,
        company_embeddings,
        candidates_meta,
        companies_meta
    )

candidate_embeddings, company_embeddings, candidates_meta, companies_meta = load_core()

# =========================================================
# FAIRNESS FUNCTION (UNCHANGED)
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
# SCORE DISTRIBUTION (UNCHANGED)
# =========================================================
@st.cache_data(show_spinner=False)
def compute_score_distribution(
    candidate_embeddings,
    company_embeddings,
    sample_size=200
):
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
# NETWORK GRAPH (UNCHANGED)
# =========================================================
@st.cache_resource(show_spinner=False)
def build_network_graph(
    candidate_embeddings,
    company_embeddings,
    candidates_meta,
    companies_meta,
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

    n_cand = min(sample_size, len(candidate_embeddings))

    for i in range(n_cand):
        net.add_node(
            f"cand_{i}",
            label=f"Candidate {i}",
            color="#667eea",
            shape="dot",
            size=18
        )

    for i in range(n_cand):
        sims = cosine_similarity(
            candidate_embeddings[i].reshape(1, -1),
            company_embeddings
        )[0]

        top_idx = np.argsort(sims)[-top_k:][::-1]

        for j in top_idx:
            label = companies_meta.iloc[j].get("name", f"Company {j}")

            net.add_node(
                f"comp_{j}",
                label=label,
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
# LLM CLIENT (UNCHANGED)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_llm_client():
    token = os.getenv("HF_TOKEN")
    if not token:
        return None
    return InferenceClient(token=token)

# =========================================================
# LLM EXPLANATION (REVISED – NOW RETURNS STRUCTURED OUTPUT)
# =========================================================
def explain_match_llm(candidate_row, company_row, score):
    """
    Generates a structured match explanation using Groq LLM.
    Returns a dictionary with keys: summary, strengths, gaps, recommendation.
    """
    import os
    from groq import Groq

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""
You are an HR analyst. Explain why the following candidate matches the company.
Provide your answer in valid JSON format with the exact structure below.

CANDIDATE:
Category: {candidate_row.get('Category', '')}
Skills: {candidate_row.get('skills', '')}
Objective: {candidate_row.get('career_objective', '')}

COMPANY:
Name: {company_row.get('name', '')}
Industry: {company_row.get('industries_list', '')}
Required Skills: {company_row.get('required_skills', '')}

MATCH SCORE: {score:.3f}

Return ONLY a JSON object like this:
{{
  "summary": "A concise paragraph summarizing the overall match.",
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "gaps": ["Gap 1", "Gap 2"],
  "recommendation": "A clear next-step recommendation for the candidate."
}}
Do not include any extra text before or after the JSON. The response must be parseable by json.loads().
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )

        content = response.choices[0].message.content.strip()

        # Attempt to extract JSON from the response (handles occasional markdown wrapping)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: take from first '{' to last '}'
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
            else:
                json_str = content

        data = json.loads(json_str)

        # Ensure all expected keys are present
        return {
            "summary": data.get("summary", "No summary provided."),
            "strengths": data.get("strengths", []),
            "gaps": data.get("gaps", []),
            "recommendation": data.get("recommendation", "No recommendation.")
        }

    except Exception as e:
        # Fallback in case of parsing or API errors
        return {
            "summary": f"LLM ERROR: {str(e)}",
            "strengths": [],
            "gaps": [],
            "recommendation": ""
        }

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

    st.markdown(f"**Category:** {candidate.get('Category','N/A')}")

    with st.expander("🧠 Skills", expanded=True):
        st.write(candidate.get("skills","N/A"))

    with st.expander("🎯 Career Objective", expanded=True):
        st.write(candidate.get("career_objective","N/A"))

# =========================================================
# MATCHING
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
        "Company": company.get("name","Unknown"),
        "Industry": company.get("industries_list","N/A"),
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
    candidate_embeddings,
    company_embeddings,
    candidates_meta,
    companies_meta
)

net.save_graph("network_candidate.html")
import streamlit.components.v1 as components
components.html(open("network_candidate.html").read(), height=620, scrolling=True)

# =========================================================
# LLM EXPLANATION (UI REMAINS UNCHANGED)
# =========================================================
st.markdown("---")
st.subheader("🤖 Match Explanation (LLM)")

with st.expander("Why is this company a good match?", expanded=True):
    if st.button("Generate AI Explanation"):
        explanation = explain_match_llm(
            candidate,
            companies_meta.iloc[top_idx[0]],
            top_scores[0]
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