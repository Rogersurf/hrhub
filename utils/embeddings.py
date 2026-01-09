from huggingface_hub import hf_hub_download
import numpy as np
import pickle
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_production_artifacts():
    cand_emb_path = hf_hub_download(
        repo_id="Rogersurf/hrhub-artifacts",
        filename="candidate_embeddings.npy",
        repo_type="dataset"
    )

    comp_emb_path = hf_hub_download(
        repo_id="Rogersurf/hrhub-artifacts",
        filename="company_embeddings.npy",
        repo_type="dataset"
    )

    cand_meta_path = hf_hub_download(
        repo_id="Rogersurf/hrhub-artifacts",
        filename="candidates_metadata.pkl",
        repo_type="dataset"
    )

    comp_meta_path = hf_hub_download(
        repo_id="Rogersurf/hrhub-artifacts",
        filename="companies_metadata.pkl",
        repo_type="dataset"
    )

    return (
        np.load(cand_emb_path),
        np.load(comp_emb_path),
        pickle.load(open(cand_meta_path, "rb")),
        pickle.load(open(comp_meta_path, "rb")),
    )
