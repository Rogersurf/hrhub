"""
HRHUB V2.1 - Bilateral HR Matching System
HOME PAGE - Single Viewport Design (No Scrolling)
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="HRHUB V2.1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-compact CSS - fits everything in viewport
st.markdown("""
    <style>
    /* Force single viewport */
    .main .block-container {
        padding: 0.5rem 1rem !important;
        max-width: 100% !important;
    }
    
    [data-testid="stSidebar"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Hero - minimal */
    .hero {
        text-align: center;
        padding: 0.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        margin-bottom: 0.8rem;
        color: white;
    }
    
    .hero h1 {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0 0 0.2rem 0;
    }
    
    .hero p {
        font-size: 0.85rem;
        margin: 0;
        opacity: 0.9;
    }
    
    /* Cards container */
    .cards {
        display: flex;
        gap: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .card {
        flex: 1;
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e8e8e8;
        transition: all 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    .card-icon {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .card-icon svg {
        width: 45px;
        height: 45px;
    }
    
    .card h2 {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        text-align: center;
        color: #2c3e50;
    }
    
    .card p {
        font-size: 0.75rem;
        color: #666;
        text-align: center;
        margin: 0 0 0.5rem 0;
        line-height: 1.3;
    }
    
    .card ul {
        margin: 0;
        padding-left: 1.2rem;
        font-size: 0.7rem;
        color: #555;
    }
    
    .card li {
        margin: 0.2rem 0;
    }
    
    /* Innovation */
    .innovation {
        background: linear-gradient(120deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 6px;
        padding: 0.6rem;
        margin-bottom: 0.8rem;
        border-left: 3px solid #667eea;
    }
    
    .innovation h3 {
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        color: #2c3e50;
    }
    
    .innovation p {
        font-size: 0.7rem;
        color: #555;
        margin: 0;
        line-height: 1.4;
    }
    
    /* Stats */
    .stats {
        display: flex;
        gap: 0.6rem;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    
    .stat {
        text-align: center;
        padding: 0.4rem 0.6rem;
        background: white;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
    }
    
    .stat-num {
        font-size: 1.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.65rem;
        color: #666;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        height: 36px;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 6px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        transition: all 0.2s;
        box-shadow: 0 2px 6px rgba(102, 126, 234, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(102, 126, 234, 0.35);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 0.3rem;
        font-size: 0.65rem;
        color: #999;
        border-top: 1px solid #eee;
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
    
    # Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="card">
                <div class="card-icon">
                    <svg viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="11" fill="url(#g1)"/>
                        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="white"/>
                        <defs>
                            <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#667eea"/>
                                <stop offset="100%" style="stop-color:#764ba2"/>
                            </linearGradient>
                        </defs>
                    </svg>
                </div>
                <h2>Candidate View</h2>
                <p>Find your perfect company match based on skills and experience</p>
                <ul>
                    <li>🎯 Top 10 company matches</li>
                    <li>📊 Semantic similarity scores</li>
                    <li>🕸️ Network visualization</li>
                    <li>📥 Export results</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Candidate View", key="cand"):
            st.switch_page("pages/1_👤_Candidate_View.py")
    
    with col2:
        st.markdown("""
            <div class="card">
                <div class="card-icon">
                    <svg viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="11" fill="url(#g2)"/>
                        <path d="M12 7V3H2v18h20V7H12zM6 19H4v-2h2v2zm0-4H4v-2h2v2zm0-4H4V9h2v2zm0-4H4V5h2v2zm4 12H8v-2h2v2zm0-4H8v-2h2v2zm0-4H8V9h2v2zm0-4H8V5h2v2zm10 12h-8v-2h2v-2h-2v-2h2v-2h-2V9h8v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2z" fill="white"/>
                        <defs>
                            <linearGradient id="g2" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#667eea"/>
                                <stop offset="100%" style="stop-color:#764ba2"/>
                            </linearGradient>
                        </defs>
                    </svg>
                </div>
                <h2>Company View</h2>
                <p>Discover top talent matching your company's needs</p>
                <ul>
                    <li>🎯 Top 10 candidate matches</li>
                    <li>📊 Skill alignment scores</li>
                    <li>🕸️ Talent network mapping</li>
                    <li>📥 Export candidates</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Company View", key="comp"):
            st.switch_page("pages/2_🏢_Company_View.py")
    
    # Innovation
    st.markdown("""
        <div class="innovation">
            <h3>💡 Key Innovation: Vocabulary Bridge</h3>
            <p>Traditional HR systems fail because candidates and companies speak different "languages." 
            HRHUB V2.1 uses job postings as translation bridges, converting both into a shared semantic space. 
            Collaborative filtering extends coverage from 30K to 150K companies.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats
    st.markdown("""
        <div class="stats">
            <div class="stat">
                <div class="stat-num">9.5K</div>
                <div class="stat-label">Candidates</div>
            </div>
            <div class="stat">
                <div class="stat-num">150K</div>
                <div class="stat-label">Companies</div>
            </div>
            <div class="stat">
                <div class="stat-num">384</div>
                <div class="stat-label">Dimensions</div>
            </div>
            <div class="stat">
                <div class="stat-num">&lt;100ms</div>
                <div class="stat-label">Query Time</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="footer">
            🎓 Master's Thesis - Business Data Science | Aalborg University | December 2024
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
