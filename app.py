"""
HRHUB V2.1 - Bilateral HR Matching System
HOME PAGE (Landing Only – Navigation via Sidebar)
"""

import streamlit as st
from utils.styles import inject_custom_css

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="HRHUB -Home",
    page_icon="🎯",
    layout="wide"
)

def main():
    # SIDEBAR
    with st.sidebar:
        st.markdown("### 🔎 Navigation")
        st.markdown("Select **Candidate** or **Company** view")
    

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🎯 HRHUB</h1>
        <p>Bilateral HR Matching System • NLP Embeddings & Semantic Similarity</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <h2>👤 Candidate View</h2>
            <p>Find your perfect company match</p>
            <ul>
                <li>🎯 Top-K company matches</li>
                <li>📊 Semantic similarity scores</li>
                <li>⚖️ Fairness & evaluation</li>
                <li>🤖 LLM explainability</li>
            </ul>
            <p><em>Select this view from the sidebar</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h2>🏢 Company View</h2>
            <p>Discover top matching candidates</p>
            <ul>
                <li>🎯 Top-K candidates</li>
                <li>📊 Skill alignment scores</li>
                <li>🌐 Network visualization</li>
                <li>🤖 LLM explainability</li>
            </ul>
            <p><em>Select this view from the sidebar</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center;font-size:0.7rem;color:#999;margin-top:1rem'>"
        "🎓 Master's Thesis – Business Data Science · Aalborg University"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    inject_custom_css()
    main()
