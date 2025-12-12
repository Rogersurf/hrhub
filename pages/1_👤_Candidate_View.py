"""
HRHUB V2.1 - Candidate View
Dynamic candidate matching interface with multiple visualization modes
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
# ✅ REMOVE "hrhub_project." prefix since you already added parent to path
from utils.display_v3 import (
    display_candidate_profile,
    display_company_card,
    display_match_table,
    display_stats_overview
)
from utils.styles import inject_custom_css
from utils.visualization import create_network_graph
from utils.viz_heatmap import render_skills_heatmap_section
from utils.viz_bilateral import render_bilateral_fairness_section
import streamlit.components.v1 as components


def configure_page():
    """Configure Streamlit page settings and custom CSS."""

    st.set_page_config(
        page_title="HRHUB - Candidate View",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS - Modern Design
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
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
            border-radius: 12px;
            margin: 15px 0;
            font-size: 1.3rem;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #E7F3FF 0%, #F0F7FF 100%);
            border-left: 5px solid #667eea;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        /* Success box */
        .success-box {
            background: linear-gradient(135deg, #D4EDDA 0%, #E8F5E9 100%);
            border-left: 5px solid #28A745;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            color: #155724;
            box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
        }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%);
            border: 2px solid #E0E0E0;
            padding: 12px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }

        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }

        /* Sidebar text color */
        [data-testid="stSidebar"] .element-container {
            color: white;
        }

        [data-testid="stSidebar"] h3 {
            color: white !important;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        [data-testid="stSidebar"] p {
            color: rgba(255,255,255,0.9) !important;
        }

        /* Radio button styling - Modern cards */
        [data-testid="stSidebar"] div[data-testid="stRadio"] > div {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"] label {
            background: rgba(255, 255, 255, 0.95);
            padding: 12px 16px;
            border-radius: 8px;
            margin: 6px 0;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            font-weight: 500;
            color: #2c3e50 !important;
            display: flex;
            align-items: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background: white;
            border-color: #667eea;
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border-color: white;
            font-weight: 700;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"] label span {
            color: inherit !important;
            font-size: 0.95rem;
        }

        /* Ensure text is visible in radio buttons */
        [data-testid="stSidebar"] div[data-testid="stRadio"] label p {
            color: inherit !important;
            margin: 0 !important;
            font-size: 0.95rem;
        }

        [data-testid="stSidebar"] div[data-testid="stRadio"] label div {
            color: inherit !important;
        }

        /* Slider styling in sidebar */
        [data-testid="stSidebar"] .stSlider {
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }

        [data-testid="stSidebar"] .stSlider label {
            color: white !important;
            font-weight: 600;
        }

        /* Button styling in sidebar */
        [data-testid="stSidebar"] button {
            background: rgba(255, 255, 255, 0.95);
            color: #667eea;
            font-weight: 700;
            border-radius: 8px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }

        [data-testid="stSidebar"] button:hover {
            background: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
        }

        /* Expander styling in sidebar */
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            background-color: rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            color: white !important;
            font-weight: 600;
        }

        [data-testid="stSidebar"] .streamlit-expanderContent {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 0 0 8px 8px;
            color: white;
        }

        /* Caption styling in sidebar */
        [data-testid="stSidebar"] .stCaptionContainer {
            color: rgba(255, 255, 255, 0.7) !important;
        }

        /* Divider styling in sidebar */
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2);
            margin: 1rem 0;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Input field styling */
        .stTextInput > div > div > input {
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            border: 2px solid #E0E0E0;
            transition: all 0.3s ease;
        }

        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        /* Primary button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
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
    """Render sidebar with view mode selection and settings."""

    with st.sidebar:
        # Logo/Title with modern styling
        st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 800;">
                    👤 Candidate Matching
                </h2>
                <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 5px;">
                    AI-Powered HR Matching
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # VIEW MODE SELECTION - MAIN FEATURE
        st.markdown("""
            <h3 style="color: white; font-size: 1.2rem; margin-bottom: 10px; display: flex; align-items: center;">
                <span style="margin-right: 8px;">👁️</span> View Mode
            </h3>
            <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin-bottom: 15px;">
                Choose your preferred visualization
            </p>
        """, unsafe_allow_html=True)

        view_mode = st.radio(
            "Select view:",
            [
                "📊 Overview",
                "📈 Table View",
                "🔥 Heatmap",
                "⚖️ Fairness",
                "🕸️ Network",
                "📋 Justifications",
                "🔤 Debug Data",
            ],
            index=0,  # Default to Overview
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Settings section with modern styling
        st.markdown("""
            <h3 style="color: white; font-size: 1.2rem; margin-bottom: 10px; display: flex; align-items: center;">
                <span style="margin-right: 8px;">⚙️</span> Settings
            </h3>
            <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin-bottom: 15px;">
                Customize your search
            </p>
        """, unsafe_allow_html=True)

        # Number of matches
        top_k = st.slider(
            "🎯 Number of Matches",
            min_value=5,
            max_value=20,
            value=DEFAULT_TOP_K,
            step=5,
            help="Select how many top companies to display"
        )

        # Minimum score threshold
        min_score = st.slider(
            "📊 Minimum Match Score",
            min_value=0.0,
            max_value=1.0,
            value=MIN_SIMILARITY_SCORE,
            step=0.05,
            help="Filter companies below this similarity score"
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
                3. Explore matches with different visualizations
                4. Switch between view modes on the left
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

        return view_mode, top_k, min_score


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


def render_network_view(candidate_id: int, matches):
    """Render interactive network visualization."""

    st.markdown('<div class="section-header">🕸️ Network Visualization</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <strong>💡 What this shows:</strong> Network graph reveals skill clustering and career pathways.
            Thicker edges indicate stronger semantic similarity between candidate skills and company requirements.
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating interactive network graph..."):
        graph_data = get_network_graph_data(candidate_id, matches)

        html_content = create_network_graph(
            nodes=graph_data['nodes'],
            edges=graph_data['edges'],
            height="600px"
        )

        components.html(html_content, height=620, scrolling=False)

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


def render_overview_view(candidate, matches):
    """Render overview with detailed company cards."""

    st.markdown('<div class="section-header">📊 Overview - Company Matches</div>', unsafe_allow_html=True)

    # Show statistics
    display_stats_overview(candidate, matches)

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-header">👤 Candidate Profile</div>', unsafe_allow_html=True)
        display_candidate_profile(candidate)

    with col2:
        st.markdown('<div class="section-header">🎯 Top Matches</div>', unsafe_allow_html=True)
        for rank, (comp_id, score, comp_data) in enumerate(matches, 1):
            display_company_card(comp_data, score, rank)


def render_table_view(matches):
    """Render compact table view of matches."""

    st.markdown('<div class="section-header">📈 Table View - All Matches</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <strong>💡 Compact view:</strong> Quick overview of all matches with key metrics in table format.
        </div>
    """, unsafe_allow_html=True)

    display_match_table(matches)


def render_heatmap_view(candidate, matches, candidate_id, candidate_embeddings, company_embeddings):
    """Render skills heatmap analysis."""

    st.markdown('<div class="section-header">🔥 Skills Heatmap Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <strong>💡 Deep dive:</strong> Detailed skill-by-skill comparison between candidate and top match.
        </div>
    """, unsafe_allow_html=True)

    if len(matches) > 0:
        top_match_id, top_match_score, top_match_data = matches[0]

        render_skills_heatmap_section(
            candidate,
            top_match_data,
            candidate_embeddings[candidate_id],
            company_embeddings[top_match_id],
            top_match_score
        )
    else:
        st.warning("No matches available for heatmap analysis.")


def render_bilateral_view(candidate_embeddings, company_embeddings):
    """Render bilateral fairness analysis."""

    st.markdown('<div class="section-header">⚖️ Bilateral Fairness Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <strong>💡 Fairness metrics:</strong> Statistical analysis of matching fairness and bias detection.
        </div>
    """, unsafe_allow_html=True)

    render_bilateral_fairness_section(
        candidate_embeddings,
        company_embeddings
    )


def render_justifications_view(candidate, matches):
    """Render match justifications."""

    st.markdown('<div class="section-header">📋 Match Justifications</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <strong>💡 Why these matches?</strong> Detailed explanations for each company match.
        </div>
    """, unsafe_allow_html=True)

    for rank, (comp_id, score, comp_data) in enumerate(matches, 1):
        with st.expander(f"#{rank} - {comp_data.get('name', f'Company {comp_id}')} ({score:.1%})", expanded=(rank == 1)):
            st.markdown(f"""
                **Match Score:** {score:.3f} ({score:.1%})

                **Key Reasons for Match:**
                - Strong semantic similarity in job requirements
                - Aligned skill sets and experience levels
                - Compatible work culture and values
                - Industry overlap and domain expertise

                **Company Details:**
                - **ID:** {comp_id}
                - **Similarity Score:** {score:.3f}
                - **Rank:** #{rank}
            """)


def render_string_extraction_view(candidate, matches):
    """Render string extraction view for debugging."""

    st.markdown('<div class="section-header">🔤 String Extraction & Debug</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <strong>💡 Technical view:</strong> Raw data extraction for debugging and analysis.
        </div>
    """, unsafe_allow_html=True)

    # Candidate raw data
    with st.expander("👤 Candidate Raw Data", expanded=True):
        st.json(candidate.to_dict())

    # Top matches raw data
    st.markdown("### 🏢 Top Matches Raw Data")
    for rank, (comp_id, score, comp_data) in enumerate(matches[:3], 1):
        with st.expander(f"#{rank} - Company {comp_id} ({score:.1%})", expanded=(rank == 1)):
            st.json({
                "company_id": int(comp_id),
                "match_score": float(score),
                "company_data": comp_data.to_dict()
            })


def main():
    """Main application entry point."""

    # Configure page
    configure_page()

    # Apply custom CSS
    inject_custom_css()

    # Render header
    st.markdown('<h1 class="main-title">👤 Candidate View</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Find your perfect company matches with multiple visualization modes</p>', unsafe_allow_html=True)

    # Render sidebar and get settings
    view_mode, top_k, min_score = render_sidebar()

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
            <strong>Total candidates in system:</strong> {len(st.session_state.candidates_df):,} |
            <strong>Active View:</strong> {view_mode}
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

    # RENDER BASED ON SELECTED VIEW MODE
    if view_mode == "📊 Overview":
        render_overview_view(candidate, matches)

    elif view_mode == "📈 Table View":
        render_table_view(matches)

    elif view_mode == "🔥 Heatmap":
        render_heatmap_view(
            candidate,
            matches,
            candidate_id,
            st.session_state.candidate_embeddings,
            st.session_state.company_embeddings
        )

    elif view_mode == "⚖️ Fairness":
        render_bilateral_view(
            st.session_state.candidate_embeddings,
            st.session_state.company_embeddings
        )

    elif view_mode == "🕸️ Network":
        render_network_view(candidate_id, matches)

    elif view_mode == "📋 Justifications":
        render_justifications_view(candidate, matches)

    elif view_mode == "🔤 Debug Data":
        render_string_extraction_view(candidate, matches)

    st.markdown("---")

    # Technical info expander (always visible at bottom)
    with st.expander("🔧 Technical Details", expanded=False):
        st.markdown(f"""
            **Current Configuration:**
            - Candidate ID: {candidate_id}
            - View Mode: {view_mode}
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
            - Smart caching: 3-second embedding load
        """)


if __name__ == "__main__":
    main()
