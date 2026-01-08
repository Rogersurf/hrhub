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
    page_title="HRHUB V2.1",
    page_icon="🎯",
    layout="wide"
)

# =========================================================
# CSS – mantém seu visual (sem esconder sidebar)
# =========================================================
st.markdown("""
<style>
.main .block-container {
    padding: 0.8rem 1.2rem !important;
    max-width: 100% !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    margin-bottom: 1rem;
    color: white;
}

.hero h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
}

.hero p {
    font-size: 0.9rem;
    opacity: 0.9;
}

/* Cards */
.card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border: 1px solid #eaeaea;
}

.card h2 {
    text-align: center;
    font-size: 1.2rem;
    margin-bottom: 0.4rem;
}

.card p {
    text-align: center;
    font-size: 0.8rem;
    color: #555;
}

.card ul {
    font-size: 0.75rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🎯 HRHUB V2.1</h1>
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
    main()
    inject_custom_css()
