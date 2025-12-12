"""
HRHUB V2.1 - Centralized CSS Styles
Inject CSS into Streamlit pages
"""

import streamlit as st


def inject_custom_css():
    """
    Inject all custom CSS into Streamlit page.
    Call this at the start of each page.
    """
    st.markdown("""
        <style>
        /* Section Headers */
        .section-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 1.3rem;
            font-weight: bold;
            text-align: center;
        }

        /* Info Boxes */
        .info-box {
            background-color: #E7F3FF;
            border-left: 5px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            line-height: 1.6;
        }

        .info-box-blue {
            background-color: #E7F3FF;
            border-left: 5px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }

        .info-box-orange {
            background-color: #FFF4E6;
            border-left: 5px solid #FF9800;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }

        .info-box-green {
            background-color: #D4EDDA;
            border-left: 5px solid #28A745;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }

        /* Success Box */
        .success-box {
            background-color: #D4EDDA;
            border-left: 5px solid #28A745;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            color: #155724;
        }

        /* Warning Box */
        .warning-box {
            background-color: #FFF3CD;
            border-left: 5px solid #FFC107;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            color: #856404;
        }

        /* Algorithm Flow */
        .algorithm-flow {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #667eea;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }

        .flow-step {
            display: flex;
            align-items: center;
            margin: 15px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }

        .flow-step:hover {
            transform: translateX(5px);
        }

        .flow-step-number {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2rem;
            margin-right: 15px;
            flex-shrink: 0;
        }

        .flow-step-content {
            flex: 1;
        }

        .flow-arrow {
            font-size: 2rem;
            margin: 5px 0;
            text-align: center;
            color: #667eea;
        }

        /* Bilateral Badges */
        .bilateral-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 5px;
        }

        .bilateral-badge-good {
            background: #D4EDDA;
            color: #155724;
            border: 1px solid #28A745;
        }

        .bilateral-badge-fair {
            background: #FFF3CD;
            color: #856404;
            border: 1px solid #FFC107;
        }

        .bilateral-badge-poor {
            background: #F8D7DA;
            color: #721C24;
            border: 1px solid #DC3545;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
