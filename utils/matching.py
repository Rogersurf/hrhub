"""
Matching algorithms for HRHUB.
Contains cosine similarity and matching logic.
"""

import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(
    candidate_embedding: np.ndarray,
    company_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between candidate and all companies.
    
    Args:
        candidate_embedding: Single candidate vector (384,)
        company_embeddings: All company vectors (N, 384)
    
    Returns:
        Similarity scores array (N,)
    """
    
    # Reshape candidate to (1, 384) for sklearn
    candidate_reshaped = candidate_embedding.reshape(1, -1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(candidate_reshaped, company_embeddings)
    
    # Return as 1D array
    return similarities.flatten()


def find_top_matches(
    candidate_embedding: np.ndarray,
    company_embeddings: np.ndarray,
    top_k: int = 10,
    min_score: float = 0.0
) -> List[Tuple[int, float]]:
    """
    Find top K company matches for a candidate.
    
    Args:
        candidate_embedding: Candidate vector
        company_embeddings: All company vectors
        top_k: Number of top matches to return
        min_score: Minimum similarity score threshold
    
    Returns:
        List of (company_index, similarity_score) tuples
    """
    
    # Compute all similarities
    similarities = compute_similarity(candidate_embedding, company_embeddings)
    
    # Filter by minimum score
    valid_indices = np.where(similarities >= min_score)[0]
    valid_scores = similarities[valid_indices]
    
    # Sort by score (descending)
    sorted_idx = np.argsort(valid_scores)[::-1]
    
    # Get top K
    top_indices = valid_indices[sorted_idx][:top_k]
    top_scores = valid_scores[sorted_idx][:top_k]
    
    # Return as list of tuples
    return list(zip(top_indices.tolist(), top_scores.tolist()))


def compute_match_strength(score: float) -> str:
    """
    Convert similarity score to human-readable strength.
    
    Args:
        score: Similarity score (0-1)
    
    Returns:
        Match strength label
    """
    
    if score >= 0.8:
        return "🔥 Excellent"
    elif score >= 0.7:
        return "✨ Very Good"
    elif score >= 0.6:
        return "👍 Good"
    elif score >= 0.5:
        return "✓ Fair"
    else:
        return "⚠ Weak"
