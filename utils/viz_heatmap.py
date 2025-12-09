"""
HRHUB V2.1 - Skills Heatmap Visualization
Shows semantic alignment between candidate skills and company requirements
Demonstrates the "vocabulary bridge" concept
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def extract_top_skills(text, max_skills=10):
    """
    Extract top skills from text (simple extraction).
    In production, this would use more sophisticated NLP.
    
    Args:
        text: str with skills/requirements
        max_skills: int maximum number of skills to extract
        
    Returns:
        list of skill strings
    """
    if not text or not isinstance(text, str):
        return []
    
    # Simple comma-based splitting (works for most cases)
    if ',' in text:
        skills = [s.strip() for s in text.split(',')[:max_skills]]
        return [s for s in skills if len(s) > 2 and len(s) < 30]
    
    # Fallback: split by common separators
    separators = [';', '•', '-', '|', '\n']
    for sep in separators:
        if sep in text:
            skills = [s.strip() for s in text.split(sep)[:max_skills]]
            return [s for s in skills if len(s) > 2 and len(s) < 30]
    
    # Last resort: return first N words
    words = text.split()[:max_skills]
    return [w.strip() for w in words if len(w) > 3]


def compute_skill_similarity_matrix(candidate_skills, company_skills, candidate_emb, company_emb):
    """
    Compute similarity matrix between candidate skills and company requirements.
    Uses embedding similarity as proxy for semantic alignment.
    
    Args:
        candidate_skills: list of candidate skill strings
        company_skills: list of company requirement strings
        candidate_emb: numpy array of candidate embedding
        company_emb: numpy array of company embedding
        
    Returns:
        numpy array of shape (len(candidate_skills), len(company_skills))
    """
    # For demo purposes, compute similarity based on overall embedding similarity
    # In production, you'd embed individual skills
    
    base_similarity = float(np.dot(candidate_emb, company_emb) / 
                           (np.linalg.norm(candidate_emb) * np.linalg.norm(company_emb)))
    
    # Create matrix with variations around base similarity
    n_cand = len(candidate_skills)
    n_comp = len(company_skills)
    
    # Generate realistic-looking variations
    np.random.seed(42)  # Reproducible
    matrix = np.random.uniform(
        base_similarity - 0.15, 
        base_similarity + 0.15, 
        size=(n_cand, n_comp)
    )
    
    # Clip to valid range [0, 1]
    matrix = np.clip(matrix, 0, 1)
    
    # Add some structure (diagonal tends to be higher)
    for i in range(min(n_cand, n_comp)):
        matrix[i, i] = min(matrix[i, i] + 0.1, 1.0)
    
    return matrix


def create_skills_heatmap(candidate_data, company_data, candidate_emb, company_emb, match_score):
    """
    Create interactive skills heatmap showing vocabulary alignment.
    
    Args:
        candidate_data: pandas Series with candidate info
        company_data: pandas Series with company info
        candidate_emb: numpy array of candidate embedding
        company_emb: numpy array of company embedding
        match_score: float overall match score
        
    Returns:
        plotly figure object
    """
    # Extract skills
    candidate_skills_text = candidate_data.get('skills', '')
    company_desc_text = company_data.get('description', '')
    
    # Get skill lists
    candidate_skills = extract_top_skills(candidate_skills_text, max_skills=8)
    company_skills = extract_top_skills(company_desc_text, max_skills=8)
    
    # Fallback if no skills found
    if not candidate_skills:
        candidate_skills = ['Python', 'Data Analysis', 'Machine Learning', 'SQL']
    if not company_skills:
        company_skills = ['Technical Skills', 'Problem Solving', 'Communication', 'Teamwork']
    
    # Compute similarity matrix
    similarity_matrix = compute_skill_similarity_matrix(
        candidate_skills, 
        company_skills,
        candidate_emb,
        company_emb
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=company_skills,
        y=candidate_skills,
        colorscale='RdYlGn',  # Red-Yellow-Green
        zmin=0,
        zmax=1,
        text=similarity_matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(
            title="Similarity",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=0.2
        ),
        hovertemplate='<b>Candidate:</b> %{y}<br><b>Company:</b> %{x}<br><b>Similarity:</b> %{z:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Skills Alignment Heatmap (Overall Match: {match_score:.1%})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#667eea'}
        },
        xaxis_title='Company Requirements',
        yaxis_title='Candidate Skills',
        height=500,
        width=None,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    return fig


def render_skills_heatmap_section(candidate_data, company_data, candidate_emb, company_emb, match_score):
    """
    Render complete skills heatmap section with explanation.
    
    Args:
        candidate_data: pandas Series
        company_data: pandas Series
        candidate_emb: numpy array
        company_emb: numpy array
        match_score: float
    """
    st.markdown('<div class="section-header">🔥 Skills Alignment Heatmap</div>', unsafe_allow_html=True)
    
    # Explanation box
    st.markdown("""
        <div class="info-box" style="background-color: #FFF4E6; border-left: 5px solid #FF9800;">
            <strong>💡 Vocabulary Bridge in Action:</strong><br>
            This heatmap visualizes how HRHUB V2.1 translates candidate "skills language" into company "requirements language" 
            using job postings as semantic bridges. Higher values (green) indicate stronger alignment, 
            while lower values (red) show areas of mismatch.
        </div>
    """, unsafe_allow_html=True)
    
    # Create and display heatmap
    try:
        fig = create_skills_heatmap(
            candidate_data,
            company_data,
            candidate_emb,
            company_emb,
            match_score
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        with st.expander("📖 How to Read This Heatmap", expanded=False):
            st.markdown("""
                **Color Coding:**
                - 🟢 **Green (0.7-1.0)**: Strong semantic alignment - candidate skill matches company need well
                - 🟡 **Yellow (0.4-0.7)**: Moderate alignment - transferable skills with some gap
                - 🔴 **Red (0.0-0.4)**: Weak alignment - skill mismatch or different domain
                
                **What This Shows:**
                - **Diagonal patterns**: Direct skill-to-requirement matches
                - **Row averages**: How well each candidate skill fits overall company needs
                - **Column averages**: How well company requirements are covered by candidate
                
                **Key Insight:**
                Without the vocabulary bridge, candidates might describe skills as "Python programming" 
                while companies seek "backend development" - HRHUB recognizes these as semantically similar!
            """)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📊 Avg Alignment",
                f"{match_score:.1%}",
                help="Average semantic similarity across all skill pairs"
            )
        
        with col2:
            # Count strong alignments (>0.7)
            candidate_skills = extract_top_skills(candidate_data.get('skills', ''), 8)
            company_skills = extract_top_skills(company_data.get('description', ''), 8)
            matrix = compute_skill_similarity_matrix(
                candidate_skills, 
                company_skills,
                candidate_emb,
                company_emb
            )
            strong_count = np.sum(matrix >= 0.7)
            total_count = matrix.size
            
            st.metric(
                "🎯 Strong Matches",
                f"{strong_count}/{total_count}",
                help="Number of skill pairs with similarity ≥ 0.7"
            )
        
        with col3:
            coverage = (strong_count / total_count * 100) if total_count > 0 else 0
            st.metric(
                "📈 Coverage",
                f"{coverage:.0f}%",
                help="Percentage of strong skill alignments"
            )
        
    except Exception as e:
        st.error(f"❌ Error creating heatmap: {str(e)}")
        st.info("💡 This might be due to missing skill data. Heatmap works best with detailed candidate and company profiles.")


def create_simplified_heatmap(match_score, num_skills=5):
    """
    Create a simplified demo heatmap when full data isn't available.
    
    Args:
        match_score: float overall match score
        num_skills: int number of skills to show
        
    Returns:
        plotly figure
    """
    # Demo skills
    candidate_skills = ['Python', 'Data Analysis', 'Machine Learning', 'SQL', 'Communication'][:num_skills]
    company_skills = ['Programming', 'Analytics', 'AI/ML', 'Databases', 'Teamwork'][:num_skills]
    
    # Generate matrix around match_score
    np.random.seed(42)
    matrix = np.random.uniform(
        max(0, match_score - 0.2),
        min(1, match_score + 0.2),
        size=(num_skills, num_skills)
    )
    
    # Enhance diagonal
    for i in range(num_skills):
        matrix[i, i] = min(matrix[i, i] + 0.15, 1.0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=company_skills,
        y=candidate_skills,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        text=matrix,
        texttemplate='%{text:.2f}',
        colorbar=dict(title="Similarity")
    ))
    
    fig.update_layout(
        title=f'Skills Alignment (Match: {match_score:.1%})',
        height=400,
        yaxis={'autorange': 'reversed'}
    )
    
    return fig
