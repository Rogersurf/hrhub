from huggingface_hub import hf_hub_download
import numpy as np
import pickle
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_production_artifacts():
    base = "processed"

    cand_emb_path = hf_hub_download(
        repo_id="Rogersurf/hrhub-artifacts",
        filename=f"{base}/candidate_embeddings.npy",
        repo_type="dataset"
    )

    comp_emb_path = hf_hub_download(
        repo_id="Rogersurf/hrhub-artifacts",
        filename=f"{base}/company_embeddings.npy",
        repo_type="dataset"
    )

    cand_meta_path = hf_hub_download(
        repo_id="Rogersurf/hrhub-artifacts",
        filename=f"{base}/candidates_metadata.pkl",
        repo_type="dataset"
    )

    comp_meta_path = hf_hub_download(
        repo_id="Rogersurf/hrhub-artifacts",
        filename=f"{base}/companies_metadata.pkl",
        repo_type="dataset"
    )

    candidate_embeddings = np.load(cand_emb_path)
    company_embeddings = np.load(comp_emb_path)

    with open(cand_meta_path, "rb") as f:
        candidates_meta = pickle.load(f)

    with open(comp_meta_path, "rb") as f:
        companies_meta = pickle.load(f)

    return (
        candidate_embeddings,
        company_embeddings,
        candidates_meta,
        companies_meta,
    )
