"""
HRHUB V2.1 - Candidate View
Dynamic candidate matching interface with customizable parameters
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
    find_top_matches
)
from hrhub_project.utils.display_v3 import (
    display_candidate_profile,
    display_company_card,
    display_match_table,
    display_stats_overview
)
from utils.visualization import create_network_graph
from utils.viz_heatmap import render_skills_heatmap_section
import streamlit.components.v1 as components


def configure_page():
    """Configure Streamlit page settings and custom CSS."""
    
    st.set_page_config(
        page_title="HRHUB - Candidate View",
        page_icon="👤",
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
            background-color: #E7F3FF;
            border-left: 5px solid #667eea;
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


def validate_candidate_input(input_str):
    """
    Validate candidate input format (e.g., C33, J34).
    Returns: (is_valid, candidate_id, error_message)
    """
    if not input_str:
        return False, None, "Please enter a candidate ID"
    
    # Pattern: Letter followed by numbers
    pattern = r'^([A-Z])(\d+)$'
    match = re.match(pattern, input_str.upper().strip())
    
    if not match:
        return False, None, "Invalid format. Use format like: C33, J34, A1, etc."
    
    letter, number = match.groups()
    candidate_id = int(number)
    
    return True, candidate_id, None


def render_sidebar():
    """Render sidebar with controls and information."""
    
    with st.sidebar:
        # Logo/Title
        st.markdown("### 👤 Candidate Matching")
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
            help="Select how many top companies to display"
        )
        
        # Minimum score threshold
        min_score = st.slider(
            "Minimum Match Score",
            min_value=0.0,
            max_value=1.0,
            value=MIN_SIMILARITY_SCORE,
            step=0.05,
            help="Filter companies below this similarity score"
        )
        
        st.markdown("---")
        
        # View mode selection
        st.markdown("### 👀 View Mode")
        view_mode = st.radio(
            "Select view:",
            ["📊 Overview", "🔍 Detailed Cards", "📈 Table View"],
            help="Choose how to display company matches"
        )
        
        st.markdown("---")
        
        # Information section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
                **Candidate View** helps you find your ideal company matches based on:
                
                - 🤖 **NLP Embeddings**: 384-dimensional semantic space
                - 📊 **Cosine Similarity**: Scale-invariant matching
                - 🌉 **Job Postings Bridge**: Vocabulary alignment
                
                **How it works:**
                1. Enter your candidate ID (e.g., C33, J34)
                2. System finds top company matches
                3. Explore matches with scores and details
                4. Visualize connections via network graph
            """)
        
        with st.expander("📚 Input Format", expanded=False):
            st.markdown("""
                **Valid formats:**
                - `C33` → Candidate 33
                - `J34` → Candidate 34
                - `A1` → Candidate 1
                
                **Pattern:** Single letter + number
            """)
        
        st.markdown("---")
        
        # Back to home button
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("app.py")
        
        # Version info
        st.caption(f"Version: {VERSION}")
        st.caption("© 2024 HRHUB Team")
        
        return top_k, min_score, view_mode


def get_network_graph_data(candidate_id, matches):
    """Generate network graph data from matches."""
    nodes = []
    edges = []
    
    # Add candidate node (green)
    nodes.append({
        'id': f'C{candidate_id}',
        'label': f'Candidate #{candidate_id}',
        'color': '#4ade80',
        'shape': 'dot',
        'size': 30
    })
    
    # Add company nodes (red) and edges
    for comp_id, score, comp_data in matches:
        # Get company name (truncate if too long)
        comp_name = comp_data.get('name', f'Company {comp_id}')
        if len(comp_name) > 30:
            comp_name = comp_name[:27] + '...'
        
        nodes.append({
            'id': f'COMP{comp_id}',
            'label': comp_name,
            'color': '#ff6b6b',
            'shape': 'box',
            'size': 20
        })
        
        edges.append({
            'from': f'C{candidate_id}',
            'to': f'COMP{comp_id}',
            'value': float(score) * 10,
            'title': f'Match Score: {score:.3f}'
        })
    
    return {'nodes': nodes, 'edges': edges}


def render_network_section(candidate_id: int, matches):
    """Render interactive network visualization section."""
    
    st.markdown('<div class="section-header">🕸️ Network Visualization</div>', unsafe_allow_html=True)
    
    # Explanation box
    st.markdown("""
        <div class="info-box">
            <strong>💡 What this shows:</strong> Network graph reveals skill clustering and career pathways. 
            Thicker edges indicate stronger semantic similarity between candidate skills and company requirements.
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Generating interactive network graph..."):
        # Get graph data
        graph_data = get_network_graph_data(candidate_id, matches)
        
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
            - 🟢 **Green circle**: Your candidate profile
            - 🔴 **Red squares**: Matched companies
            - **Line thickness**: Match strength (thicker = better)
        """)


def render_matches_section(matches, view_mode: str):
    """Render company matches section with different view modes."""
    
    st.markdown('<div class="section-header">🎯 Company Matches</div>', unsafe_allow_html=True)
    
    if view_mode == "📊 Overview":
        # Table view
        display_match_table(matches)
        
    elif view_mode == "🔍 Detailed Cards":
        # Card view - detailed
        for rank, (comp_id, score, comp_data) in enumerate(matches, 1):
            display_company_card(comp_data, score, rank)
            
    elif view_mode == "📈 Table View":
        # Compact table
        display_match_table(matches)


def main():
    """Main application entry point."""
    
    # Configure page
    configure_page()
    
    # Render header
    st.markdown('<h1 class="main-title">👤 Candidate View</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Find your perfect company matches</p>', unsafe_allow_html=True)
    
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
                        ✅ Data loaded successfully! Ready to match.
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.stop()
    
    # Candidate input section
    st.markdown("### 🔍 Enter Candidate ID")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        candidate_input = st.text_input(
            "Candidate ID",
            value="C33",
            max_chars=10,
            help="Enter candidate ID (e.g., C33, J34, A1)",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🚀 Find Matches", use_container_width=True, type="primary")
    
    # Validate input
    is_valid, candidate_id, error_msg = validate_candidate_input(candidate_input)
    
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")
        st.info("💡 **Tip:** Use format like C33, J34, or A1")
        st.stop()
    
    # Check if candidate exists
    if candidate_id >= len(st.session_state.candidates_df):
        st.error(f"❌ Candidate ID {candidate_id} not found. Maximum ID: {len(st.session_state.candidates_df) - 1}")
        st.stop()
    
    # Load candidate data
    candidate = st.session_state.candidates_df.iloc[candidate_id]
    
    # Show candidate info
    st.markdown(f"""
        <div class="info-box">
            <strong>Selected:</strong> Candidate #{candidate_id} | 
            <strong>Total candidates in system:</strong> {len(st.session_state.candidates_df):,}
        </div>
    """, unsafe_allow_html=True)
    
    # Find matches
    with st.spinner("🔄 Finding top matches..."):
        matches_list = find_top_matches(
            candidate_id,
            st.session_state.candidate_embeddings,
            st.session_state.company_embeddings,
            st.session_state.companies_df,
            top_k
        )
    
    # Format matches for display
    matches = [
        (m['company_id'], m['score'], st.session_state.companies_df.iloc[m['company_id']])
        for m in matches_list
    ]
    
    # Filter by minimum score
    matches = [(cid, score, cdata) for cid, score, cdata in matches if score >= min_score]
    
    if not matches:
        st.warning(f"⚠️ No matches found above {min_score:.0%} threshold. Try lowering the minimum score in the sidebar.")
        st.stop()
    
    st.markdown("---")
    
    # Display statistics overview
    display_stats_overview(candidate, matches)
    
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Candidate profile section
        st.markdown('<div class="section-header">👤 Candidate Profile</div>', unsafe_allow_html=True)
        display_candidate_profile(candidate)
    
    with col2:
        # Matches section
        render_matches_section(matches, view_mode)
    
    st.markdown("---")
    
    # Skills Heatmap (show for top match)
    if len(matches) > 0:
        top_match_id, top_match_score, top_match_data = matches[0]
        
        st.markdown("### 🔥 Skills Analysis - Top Match")
        render_skills_heatmap_section(
            candidate,
            top_match_data,
            st.session_state.candidate_embeddings[candidate_id],
            st.session_state.company_embeddings[top_match_id],
            top_match_score
        )
    
    st.markdown("---")
    
    # Network visualization (full width)
    render_network_section(candidate_id, matches)
    
    st.markdown("---")
    
    # Technical info expander
    with st.expander("🔧 Technical Details", expanded=False):
        st.markdown(f"""
            **Current Configuration:**
            - Candidate ID: {candidate_id}
            - Embedding Dimension: {EMBEDDING_DIMENSION}
            - Similarity Metric: Cosine Similarity
            - Top K Matches: {top_k}
            - Minimum Score: {min_score:.0%}
            - Candidates Loaded: {len(st.session_state.candidates_df):,}
            - Companies Loaded: {len(st.session_state.companies_df):,}
            
            **Algorithm:**
            1. Load pre-computed embeddings (.npy files)
            2. Calculate cosine similarity between candidate and all companies
            3. Rank companies by similarity score
            4. Return top-K matches above threshold
            
            **Performance:**
            - Query time: <100ms (sub-second matching)
            - Smart caching: 3-second embedding load (from 5 minutes)
        """)


if __name__ == "__main__":
    main()
