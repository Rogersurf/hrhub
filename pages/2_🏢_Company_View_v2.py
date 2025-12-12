"""
HRHUB V2.1 - Company View
Dynamic company-to-candidate matching interface
"""

import streamlit as st
import sys
from pathlib import Path
import re

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import *
from data.data_loader import (
    load_embeddings,
    # find_top_matches_company  # Function doesn't exist yet - using embedded version below
)
from hrhub_project.utils.display_v3 import (
    display_company_profile_basic,
    display_candidate_card_basic,
    display_match_table_candidates,
    display_stats_overview_company
)
from utils.visualization import create_network_graph
from utils.viz_heatmap import render_skills_heatmap_section
import streamlit.components.v1 as components
import numpy as np


def configure_page():
    """Configure Streamlit page settings and custom CSS."""
    
    st.set_page_config(
        page_title="HRHUB - Company View",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #667eea;
            margin-bottom: 0;
        }
        
        .sub-title {
            font-size: 1rem;
            text-align: center;
            color: #666;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }
        
        /* Section headers */
        .section-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 1.3rem;
            font-weight: bold;
        }
        
        /* Info boxes */
        .info-box {
            background-color: #FFF4E6;
            border-left: 5px solid #FF9800;
            padding: 12px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Success box */
        .success-box {
            background-color: #D4EDDA;
            border-left: 5px solid #28A745;
            padding: 12px;
            border-radius: 5px;
            margin: 10px 0;
            color: #155724;
        }
        
        /* Warning box */
        .warning-box {
            background-color: #FFF3CD;
            border-left: 5px solid #FFC107;
            padding: 12px;
            border-radius: 5px;
            margin: 10px 0;
            color: #856404;
        }
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: #F8F9FA;
            border: 2px solid #E0E0E0;
            padding: 12px;
            border-radius: 8px;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #F0F2F6;
            border-radius: 5px;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Input field styling */
        .stTextInput > div > div > input {
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)


def validate_company_input(input_str):
    """
    Validate company input (ID or search term).
    Returns: (is_valid, company_id, error_message)
    """
    if not input_str:
        return False, None, "Please enter a company ID or name"
    
    input_clean = input_str.strip()
    
    # Check if it's a numeric ID
    if input_clean.isdigit():
        company_id = int(input_clean)
        return True, company_id, None
    
    # Otherwise treat as search term (we'll search by name)
    return True, input_clean, None


def find_company_by_name(companies_df, search_term):
    """
    Find company by name (case-insensitive partial match).
    Returns: (found, company_id, company_name)
    """
    search_lower = search_term.lower()
    
    # Search in company names
    if 'name' in companies_df.columns:
        matches = companies_df[companies_df['name'].str.lower().str.contains(search_lower, na=False)]
        
        if len(matches) > 0:
            # Return first match
            company_id = matches.index[0]
            company_name = matches.iloc[0]['name']
            return True, company_id, company_name
    
    return False, None, None


def find_top_candidate_matches(company_id, company_embeddings, candidate_embeddings, candidates_df, top_k=10):
    """
    Find top candidate matches for a company (reverse of candidate matching).
    """
    # Get company embedding
    company_emb = company_embeddings[company_id].reshape(1, -1)
    
    # Calculate cosine similarity with all candidates
    # Normalize embeddings
    company_norm = company_emb / np.linalg.norm(company_emb)
    candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = np.dot(candidate_norms, company_norm.T).flatten()
    
    # Get top K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Format results
    matches = []
    for idx in top_indices:
        matches.append({
            'candidate_id': int(idx),
            'score': float(similarities[idx])
        })
    
    return matches


def render_sidebar():
    """Render sidebar with controls and information."""
    
    with st.sidebar:
        # Logo/Title
        st.markdown("### 🏢 Company Matching")
        st.markdown("---")
        
        # Settings section
        st.markdown("### ⚙️ Settings")
        
        # Number of matches
        top_k = st.slider(
            "Number of Matches",
            min_value=5,
            max_value=20,
            value=DEFAULT_TOP_K,
            step=5,
            help="Select how many top candidates to display"
        )
        
        # Minimum score threshold
        min_score = st.slider(
            "Minimum Match Score",
            min_value=0.0,
            max_value=1.0,
            value=MIN_SIMILARITY_SCORE,
            step=0.05,
            help="Filter candidates below this similarity score"
        )
        
        st.markdown("---")
        
        # View mode selection
        st.markdown("### 👀 View Mode")
        view_mode = st.radio(
            "Select view:",
            ["📊 Overview", "🔍 Detailed Cards", "📈 Table View"],
            help="Choose how to display candidate matches"
        )
        
        st.markdown("---")
        
        # Information section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
                **Company View** helps you discover top talent based on:
                
                - 🤖 **NLP Embeddings**: 384-dimensional semantic space
                - 📊 **Cosine Similarity**: Scale-invariant matching
                - 🌉 **Job Postings Bridge**: Vocabulary alignment
                
                **How it works:**
                1. Enter company ID or search by name
                2. System finds top candidate matches
                3. Explore candidates with scores and skills
                4. Visualize talent network via graph
            """)
        
        with st.expander("📚 Input Format", expanded=False):
            st.markdown("""
                **Valid formats:**
                - `9418` → Company ID 9418
                - `30989` → Company ID 30989
                - `Anblicks` → Search by name
                - `iO Associates` → Partial name search
                
                **Search tips:**
                - Case-insensitive
                - Partial matches work
                - Returns first match found
            """)
        
        with st.expander("📊 Coverage Info", expanded=False):
            st.markdown("""
                **Company Coverage:**
                - 🟢 **30,000 companies** with job postings
                - 🟡 **120,000 companies** via collaborative filtering
                - 📈 **5x coverage expansion** through skill inference
                
                Companies without job postings inherit skills from similar companies.
            """)
        
        st.markdown("---")
        
        # Back to home button
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("app.py")
        
        # Version info
        st.caption(f"Version: {VERSION}")
        st.caption("© 2024 HRHUB Team")
        
        return top_k, min_score, view_mode


def get_network_graph_data_company(company_id, matches, companies_df):
    """Generate network graph data from matches (company perspective)."""
    nodes = []
    edges = []
    
    # Add company node (red/orange)
    company_name = companies_df.iloc[company_id].get('name', f'Company {company_id}')
    if len(company_name) > 30:
        company_name = company_name[:27] + '...'
    
    nodes.append({
        'id': f'COMP{company_id}',
        'label': company_name,
        'color': '#ff6b6b',
        'shape': 'box',
        'size': 30
    })
    
    # Add candidate nodes (green) and edges
    for cand_id, score, cand_data in matches:
        nodes.append({
            'id': f'C{cand_id}',
            'label': f'Candidate #{cand_id}',
            'color': '#4ade80',
            'shape': 'dot',
            'size': 20
        })
        
        edges.append({
            'from': f'COMP{company_id}',
            'to': f'C{cand_id}',
            'value': float(score) * 10,
            'title': f'Match Score: {score:.3f}'
        })
    
    return {'nodes': nodes, 'edges': edges}


def render_network_section(company_id: int, matches, companies_df):
    """Render interactive network visualization section."""
    
    st.markdown('<div class="section-header">🕸️ Talent Network</div>', unsafe_allow_html=True)
    
    # Explanation box
    st.markdown("""
        <div class="info-box">
            <strong>💡 What this shows:</strong> Talent network reveals skill alignment and candidate clustering. 
            Thicker edges indicate stronger semantic match between company requirements and candidate skills.
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Generating interactive network graph..."):
        # Get graph data
        graph_data = get_network_graph_data_company(company_id, matches, companies_df)
        
        # Create HTML graph
        html_content = create_network_graph(
            nodes=graph_data['nodes'],
            edges=graph_data['edges'],
            height="600px"
        )
        
        # Display in Streamlit
        components.html(html_content, height=620, scrolling=False)
    
    # Graph instructions
    with st.expander("📖 Graph Controls", expanded=False):
        st.markdown("""
            **How to interact:**
            
            - 🖱️ **Drag nodes**: Click and drag to reposition
            - 🔍 **Zoom**: Scroll to zoom in/out
            - 👆 **Pan**: Click background and drag to pan
            - 🎯 **Hover**: Hover over nodes/edges for details
            
            **Legend:**
            - 🔴 **Red square**: Your company
            - 🟢 **Green circles**: Matched candidates
            - **Line thickness**: Match strength (thicker = better)
        """)


def render_matches_section(matches, view_mode: str):
    """Render candidate matches section with different view modes."""
    
    st.markdown('<div class="section-header">🎯 Candidate Matches</div>', unsafe_allow_html=True)
    
    if view_mode == "📊 Overview" or view_mode == "📈 Table View":
        # Table view - use display function
        display_match_table_candidates(matches)
        
    elif view_mode == "🔍 Detailed Cards":
        # Card view - use display function
        for rank, (cand_id, score, cand_data) in enumerate(matches, 1):
            display_candidate_card_basic(cand_data, cand_id, score, rank)


def main():
    """Main application entry point."""
    
    # Configure page
    configure_page()
    
    # Render header
    st.markdown('<h1 class="main-title">🏢 Company View</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Discover top talent for your company</p>', unsafe_allow_html=True)
    
    # Render sidebar and get settings
    top_k, min_score, view_mode = render_sidebar()
    
    st.markdown("---")
    
    # Load embeddings (cache in session state)
    if 'embeddings_loaded' not in st.session_state:
        with st.spinner("📄 Loading embeddings and data..."):
            try:
                cand_emb, comp_emb, cand_df, comp_df = load_embeddings()
                st.session_state.embeddings_loaded = True
                st.session_state.candidate_embeddings = cand_emb
                st.session_state.company_embeddings = comp_emb
                st.session_state.candidates_df = cand_df
                st.session_state.companies_df = comp_df
                
                st.markdown("""
                    <div class="success-box">
                        ✅ Data loaded successfully! Ready to find talent.
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.stop()
    
    # Company input section
    st.markdown("### 🔍 Enter Company ID or Name")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        company_input = st.text_input(
            "Company ID or Name",
            value="9418",
            max_chars=100,
            help="Enter company ID (e.g., 9418) or search by name (e.g., Anblicks)",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🚀 Find Candidates", use_container_width=True, type="primary")
    
    # Validate input
    is_valid, company_id_or_search, error_msg = validate_company_input(company_input)
    
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")
        st.stop()
    
    # Determine if it's ID or search
    if isinstance(company_id_or_search, int):
        # Direct ID
        company_id = company_id_or_search
        
        # Check if company exists
        if company_id >= len(st.session_state.companies_df):
            st.error(f"❌ Company ID {company_id} not found. Maximum ID: {len(st.session_state.companies_df) - 1}")
            st.stop()
        
        company = st.session_state.companies_df.iloc[company_id]
        company_name = company.get('name', f'Company {company_id}')
        
    else:
        # Search by name
        found, company_id, company_name = find_company_by_name(st.session_state.companies_df, company_id_or_search)
        
        if not found:
            st.error(f"❌ No company found matching: '{company_id_or_search}'")
            st.info("💡 **Tip:** Try searching with partial name or use company ID directly")
            st.stop()
        
        company = st.session_state.companies_df.iloc[company_id]
        st.success(f"✅ Found: **{company_name}** (ID: {company_id})")
    
    # Show company info
    st.markdown(f"""
        <div class="info-box">
            <strong>Selected:</strong> {company_name} (ID: {company_id}) | 
            <strong>Total companies in system:</strong> {len(st.session_state.companies_df):,}
        </div>
    """, unsafe_allow_html=True)
    
    # Check if company has job postings
    has_postings = company.get('has_job_postings', False) if 'has_job_postings' in company else True
    
    if not has_postings:
        st.markdown("""
            <div class="warning-box">
                ℹ️ <strong>Note:</strong> This company uses <strong>collaborative filtering</strong> 
                (skills inherited from similar companies). Matching still works but may be less precise than companies with direct job postings.
            </div>
        """, unsafe_allow_html=True)
    
    # Find matches
    with st.spinner("🔄 Finding top candidate matches..."):
        matches_list = find_top_candidate_matches(
            company_id,
            st.session_state.company_embeddings,
            st.session_state.candidate_embeddings,
            st.session_state.candidates_df,
            top_k
        )
    
    # Format matches for display
    matches = [
        (m['candidate_id'], m['score'], st.session_state.candidates_df.iloc[m['candidate_id']])
        for m in matches_list
    ]
    
    # Filter by minimum score
    matches = [(cid, score, cdata) for cid, score, cdata in matches if score >= min_score]
    
    if not matches:
        st.warning(f"⚠️ No candidates found above {min_score:.0%} threshold. Try lowering the minimum score in the sidebar.")
        st.stop()
    
    st.markdown("---")
    
    # Display statistics using display function
    display_stats_overview_company(company, matches)
    
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Company profile section
        st.markdown('<div class="section-header">🏢 Company Profile</div>', unsafe_allow_html=True)
        
        # Use basic display function
        display_company_profile_basic(company, company_id)
    
    with col2:
        # Matches section
        render_matches_section(matches, view_mode)
    
    st.markdown("---")
    
    # Skills Heatmap (show for top candidate match)
    if len(matches) > 0:
        top_cand_id, top_cand_score, top_cand_data = matches[0]
        
        st.markdown("### 🔥 Skills Analysis - Top Candidate")
        render_skills_heatmap_section(
            top_cand_data,
            company,
            st.session_state.candidate_embeddings[top_cand_id],
            st.session_state.company_embeddings[company_id],
            top_cand_score
        )
    
    st.markdown("---")
    
    # Network visualization (full width)
    render_network_section(company_id, matches, st.session_state.companies_df)
    
    st.markdown("---")
    
    # Technical info expander
    with st.expander("🔧 Technical Details", expanded=False):
        st.markdown(f"""
            **Current Configuration:**
            - Company ID: {company_id}
            - Company Name: {company_name}
            - Embedding Dimension: {EMBEDDING_DIMENSION}
            - Similarity Metric: Cosine Similarity
            - Top K Matches: {top_k}
            - Minimum Score: {min_score:.0%}
            - Candidates Available: {len(st.session_state.candidates_df):,}
            - Companies in System: {len(st.session_state.companies_df):,}
            
            **Algorithm:**
            1. Load pre-computed company embedding
            2. Calculate cosine similarity with all candidate embeddings
            3. Rank candidates by similarity score
            4. Return top-K matches above threshold
            
            **Coverage Strategy:**
            - Companies WITH job postings: Direct semantic matching
            - Companies WITHOUT postings: Collaborative filtering (inherit from similar companies)
            - Total coverage: 150K companies (5x expansion from 30K base)
        """)


if __name__ == "__main__":
    main()
