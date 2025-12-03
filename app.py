"""
HRHUB - Bilateral HR Matching System
Main Streamlit Application

A professional HR matching system that connects candidates with companies
using NLP embeddings and cosine similarity matching.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import *
from data.data_loader import (
    load_embeddings,
    find_top_matches
)
from utils.display import (
    display_candidate_profile,
    display_company_card,
    display_match_table,
    display_stats_overview
)
from utils.visualization import create_network_graph
import streamlit.components.v1 as components


def configure_page():
    """Configure Streamlit page settings and custom CSS."""
    
    st.set_page_config(
        page_title="HRHUB - HR Matching",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #0066CC;
            margin-bottom: 0;
        }
        
        .sub-title {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-top: 0;
            margin-bottom: 2rem;
        }
        
        /* Section headers */
        .section-header {
            background: linear-gradient(90deg, #0066CC 0%, #00BFFF 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        /* Info boxes */
        .info-box {
            background-color: #E7F3FF;
            border-left: 5px solid #0066CC;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Metric cards */
        div[data-testid="metric-container"] {
            background-color: #F8F9FA;
            border: 2px solid #E0E0E0;
            padding: 15px;
            border-radius: 10px;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #F0F2F6;
            border-radius: 5px;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render application header."""
    
    st.markdown(f'<h1 class="main-title">{APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{APP_SUBTITLE}</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with controls and information."""
    
    with st.sidebar:
        st.image("https://via.placeholder.com/250x80/0066CC/FFFFFF?text=HRHUB", width=250)
        
        st.markdown("---")
        
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
        with st.expander("ℹ️ About HRHUB", expanded=False):
            st.markdown("""
                **HRHUB** is a bilateral HR matching system that uses:
                
                - 🤖 **NLP Embeddings**: Sentence transformers (384 dimensions)
                - 📏 **Cosine Similarity**: Scale-invariant matching
                - 🌉 **Job Postings Bridge**: Aligns candidate and company language
                
                **Key Innovation:**
                Companies enriched with job posting data speak the same 
                "skills language" as candidates!
            """)
        
        with st.expander("📚 How to Use", expanded=False):
            st.markdown("""
                1. **View Candidate Profile**: See the candidate's skills and background
                2. **Explore Matches**: Review top company matches with scores
                3. **Network Graph**: Visualize connections interactively
                4. **Company Details**: Click to see full company information
            """)
        
        st.markdown("---")
        
        # Version info
        st.caption(f"Version: {VERSION}")
        st.caption("© 2024 HRHUB Team")
        
        return top_k, min_score, view_mode


def get_network_graph_data(candidate_id, matches):
    """Generate network graph data from matches."""
    nodes = []
    edges = []
    
    # Add candidate node
    nodes.append({
        'id': f'C{candidate_id}',
        'label': f'Candidate #{candidate_id}',
        'color': '#4ade80',
        'shape': 'dot',
        'size': 30
    })
    
    # Add company nodes and edges
    for comp_id, score, comp_data in matches:
        nodes.append({
            'id': f'COMP{comp_id}',
            'label': comp_data.get('name', f'Company {comp_id}')[:30],
            'color': '#ff6b6b',
            'shape': 'box',
            'size': 20
        })
        
        edges.append({
            'from': f'C{candidate_id}',
            'to': f'COMP{comp_id}',
            'value': float(score) * 10,
            'title': f'{score:.3f}'
        })
    
    return {'nodes': nodes, 'edges': edges}


def render_network_section(candidate_id: int, matches):
    """Render interactive network visualization section."""
    
    st.markdown('<div class="section-header">🕸️ Network Visualization</div>', unsafe_allow_html=True)
    
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
            **How to interact with the graph:**
            
            - 🖱️ **Drag nodes**: Click and drag to reposition
            - 🔍 **Zoom**: Scroll to zoom in/out
            - 👆 **Pan**: Click background and drag to pan
            - 🎯 **Hover**: Hover over nodes and edges for details
            
            **Legend:**
            - 🟢 **Green circles**: Candidates
            - 🔴 **Red squares**: Companies
            - **Line thickness**: Match strength (thicker = better match)
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
    render_header()
    
    # Render sidebar and get settings
    top_k, min_score, view_mode = render_sidebar()
    
    # Main content area
    st.markdown("---")
    
    # Load embeddings (cache in session state)
    if 'embeddings_loaded' not in st.session_state:
        with st.spinner("🔄 Loading embeddings and data..."):
            cand_emb, comp_emb, cand_df, comp_df = load_embeddings()
            st.session_state.embeddings_loaded = True
            st.session_state.candidate_embeddings = cand_emb
            st.session_state.company_embeddings = comp_emb
            st.session_state.candidates_df = cand_df
            st.session_state.companies_df = comp_df
            st.success("✅ Data loaded successfully!")
    
    # Load candidate data
    candidate_id = DEMO_CANDIDATE_ID
    candidate = st.session_state.candidates_df.iloc[candidate_id]
    
    # Load company matches
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
        st.warning(f"No matches found above {min_score:.0%} threshold. Try lowering the minimum score.")
        return
    
    # Display statistics overview
    display_stats_overview(candidate, matches)
    
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
    
    # Network visualization (full width)
    render_network_section(candidate_id, matches)
    
    st.markdown("---")
    
    # Technical info expander
    with st.expander("🔧 Technical Details", expanded=False):
        st.markdown(f"""
            **Current Configuration:**
            - Embedding Dimension: {EMBEDDING_DIMENSION}
            - Similarity Metric: Cosine Similarity
            - Top K Matches: {top_k}
            - Minimum Score: {min_score:.0%}
            - Candidates Loaded: {len(st.session_state.candidates_df):,}
            - Companies Loaded: {len(st.session_state.companies_df):,}
            
            **Algorithm:**
            1. Load pre-computed embeddings (.npy files)
            2. Calculate cosine similarity
            3. Rank companies by similarity score
            4. Return top-K matches
        """)


if __name__ == "__main__":
    main()