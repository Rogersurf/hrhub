import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import json
import re

from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB - Company View",
    page_icon="🏢",
    layout="wide"
)

# =========================================================
# HF ARTIFACT CONFIG (SAME AS CANDIDATE VIEW)
# =========================================================
DATASET_REPO = "Rogersurf/hrhub-artifacts"
PROCESSED_DIR = "processed"

# =========================================================
# LOAD DATA (HF ARTIFACTS – SAME STANDARD)
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
# FAIRNESS (UNCHANGED)
# =========================================================
def compute_bilateral_fairness(candidate_embeddings, company_embeddings, top_k=10, sample_size=100):
    n_cand = min(sample_size, len(candidate_embeddings))
    n_comp = min(sample_size, len(company_embeddings))

    cand_scores, comp_scores = [], []

    for i in range(n_cand):
        sims = cosine_similarity(candidate_embeddings[i].reshape(1, -1), company_embeddings)[0]
        cand_scores.extend(np.sort(sims)[-top_k:])

    for j in range(n_comp):
        sims = cosine_similarity(company_embeddings[j].reshape(1, -1), candidate_embeddings[:n_cand])[0]
        comp_scores.extend(np.sort(sims)[-top_k:])

    cand_mean = float(np.mean(cand_scores))
    comp_mean = float(np.mean(comp_scores))
    fairness = min(cand_mean, comp_mean) / max(cand_mean, comp_mean)

    return cand_mean, comp_mean, fairness


@st.cache_data(show_spinner=False)
def cached_fairness(candidate_embeddings, company_embeddings, top_k):
    return compute_bilateral_fairness(candidate_embeddings, company_embeddings, top_k)

# =========================================================
# LLM EXPLANATION (GROQ VERSION – STRUCTURED JSON)
# =========================================================
def explain_match_llm(company_row, candidate_row, score):
    """
    Generate a structured match explanation using Groq LLM.
    Returns a dictionary with keys: summary, strengths, gaps, recommendation.
    """
    import os
    from groq import Groq

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""
You are an HR analyst.

Explain why the following candidate is a good match for the company.

COMPANY:
Name: {company_row.get('name','')}
Industry: {company_row.get('industries_list','')}
Required Skills: {company_row.get('required_skills','')}

CANDIDATE:
Category: {candidate_row.get('Category','')}
Skills: {candidate_row.get('skills','')}
Career Objective: {candidate_row.get('career_objective','')}

MATCH SCORE: {score:.3f}

Return ONLY a valid JSON object with the following structure:
{{
  "summary": "A concise paragraph summarizing the match.",
  "strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "gaps": ["Gap 1", "Gap 2"],
  "recommendation": "A clear next-step recommendation."
}}
Do not include any text outside the JSON. The response must be parseable by json.loads().
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # Active Groq model as of 2026
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from possible markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end] if start != -1 and end > start else "{}"

        data = json.loads(json_str)

        return {
            "summary": data.get("summary", "No summary provided."),
            "strengths": data.get("strengths", []),
            "gaps": data.get("gaps", []),
            "recommendation": data.get("recommendation", "No recommendation.")
        }

    except Exception as e:
        return {
            "summary": f"LLM ERROR: {str(e)}",
            "strengths": [],
            "gaps": [],
            "recommendation": ""
        }

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
    cand = candidates_meta.iloc[idx]
    rows.append({
        "Rank": rank,
        "Category": cand.get("Category","N/A"),
        "Score": score
    })

df = pd.DataFrame(rows)

# =========================================================
# METRICS + TABLE
# =========================================================
with right:
    st.subheader("📊 Match Overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Best Match", f"{df.Score.max():.3f}")
    m2.metric("Average Score", f"{df.Score.mean():.3f}")
    m3.metric("Strong Matches", int((df.Score > threshold).sum()))

    def style_score(v):
        return "color: green; font-weight: bold;" if v > threshold else ""

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
# LLM EXPLANATION (UI)
# =========================================================
st.markdown("---")
st.subheader("🤖 Match Explanation (LLM)")

with st.expander("Why is this candidate a good match?", expanded=True):
    if st.button("Generate AI Explanation"):
        explanation = explain_match_llm(
            company,
            candidates_meta.iloc[top_idx[0]],
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