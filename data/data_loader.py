import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings():
    """Load pre-computed embeddings and metadata."""
    
    # Load embeddings
    candidate_embeddings = np.load('data/processed/candidate_embeddings.npy')
    company_embeddings = np.load('data/processed/company_embeddings.npy')
    
    # Load metadata
    with open('data/processed/candidates_processed.pkl', 'rb') as f:
        candidates_df = pickle.load(f)
    
    with open('data/processed/companies_processed.pkl', 'rb') as f:
        companies_df = pickle.load(f)
    
    return candidate_embeddings, company_embeddings, candidates_df, companies_df

def find_top_matches(candidate_idx, candidate_embeddings, company_embeddings, companies_df, top_k=10):
    """Find top K company matches for a candidate."""
    
    # Get candidate embedding
    candidate_vec = candidate_embeddings[candidate_idx].reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity(candidate_vec, company_embeddings)[0]
    
    # Get top K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results
    matches = []
    for idx in top_indices:
        matches.append({
            'company_id': idx,
            'company_name': companies_df.iloc[idx].get('name', f'Company {idx}'),
            'job_title': companies_df.iloc[idx].get('title', 'N/A'),
            'score': float(similarities[idx])
        })
    
    return matches