"""
Display utilities for HRHUB Streamlit UI.
Contains formatted display components for candidates and companies.
"""

import streamlit as st
import pandas as pd
import ast
from typing import Dict, Any, List, Tuple


def display_candidate_profile(candidate):
    """
    Display comprehensive candidate profile in Streamlit.
    
    Args:
        candidate: Pandas Series with candidate data
    """
    
    st.markdown("### 👤 Candidate Profile")
    st.markdown("---")
    
    # Career Objective
    with st.expander("🎯 Career Objective", expanded=True):
        st.write(candidate.get('career_objective', 'Not provided'))
    
    # Skills
    with st.expander("💻 Skills & Expertise", expanded=True):
        try:
            skills = ast.literal_eval(candidate.get('skills', '[]'))
            if skills:
                # Display as tags
                skills_html = " ".join([f'<span style="background-color: #0066CC; color: white; padding: 5px 10px; border-radius: 15px; margin: 3px; display: inline-block;">{skill}</span>' for skill in skills[:15]])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.write("No skills listed")
        except:
            st.write(candidate.get('skills', 'No skills listed'))
    
    # Education
    with st.expander("🎓 Education"):
        try:
            institutions = ast.literal_eval(candidate.get('educational_institution_name', '[]'))
            degrees = ast.literal_eval(candidate.get('degree_names', '[]'))
            majors = ast.literal_eval(candidate.get('major_field_of_studies', '[]'))
            years = ast.literal_eval(candidate.get('passing_years', '[]'))
            
            if institutions and any(institutions):
                for i in range(len(institutions)):
                    degree = degrees[i] if i < len(degrees) else 'N/A'
                    major = majors[i] if i < len(majors) else 'N/A'
                    year = years[i] if i < len(years) else 'N/A'
                    
                    st.write(f"**{degree}** in {major}")
                    st.write(f"📍 {institutions[i]}")
                    st.write(f"📅 {year}")
                    if i < len(institutions) - 1:
                        st.write("---")
            else:
                st.write("No education information provided")
        except:
            st.write("No education information provided")
    
    # Work Experience
    with st.expander("💼 Work Experience"):
        try:
            companies = ast.literal_eval(candidate.get('professional_company_names', '[]'))
            positions = ast.literal_eval(candidate.get('positions', '[]'))
            starts = ast.literal_eval(candidate.get('start_dates', '[]'))
            ends = ast.literal_eval(candidate.get('end_dates', '[]'))
            
            if companies and any(companies):
                for i in range(len(companies)):
                    position = positions[i] if i < len(positions) else 'N/A'
                    start = starts[i] if i < len(starts) else 'N/A'
                    end = ends[i] if i < len(ends) else 'N/A'
                    
                    st.write(f"**{position}** at {companies[i]}")
                    st.write(f"📅 {start} - {end}")
                    if i < len(companies) - 1:
                        st.write("---")
                
                # Show responsibilities
                responsibilities = candidate.get('responsibilities', '')
                if responsibilities:
                    st.markdown("**Key Responsibilities:**")
                    st.text(responsibilities)
            else:
                st.write("No work experience listed")
        except:
            st.write("No work experience listed")


def display_company_card(
    company_data,
    similarity_score: float,
    rank: int
):
    """
    Display company information as a card.
    
    Args:
        company_data: Pandas Series with company data
        similarity_score: Match score
        rank: Ranking position
    """
    
    with st.container():
        # Header with rank and score
        col1, col2, col3 = st.columns([1, 4, 2])
        
        with col1:
            st.markdown(f"### #{rank}")
        
        with col2:
            st.markdown(f"### 🏢 {company_data.get('name', 'Unknown Company')}")
        
        with col3:
            # Color-coded score
            if similarity_score >= 0.7:
                color = "#00FF00"  # Green
                label = "Excellent"
            elif similarity_score >= 0.6:
                color = "#FFD700"  # Gold
                label = "Very Good"
            elif similarity_score >= 0.5:
                color = "#FFA500"  # Orange
                label = "Good"
            else:
                color = "#FF6347"  # Red
                label = "Fair"
            
            st.markdown(
                f'<div style="text-align: center; padding: 10px; background-color: {color}20; border: 2px solid {color}; border-radius: 10px;">'
                f'<span style="font-size: 24px; font-weight: bold; color: {color};">{similarity_score:.1%}</span><br>'
                f'<span style="font-size: 12px;">{label} Match</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Company details
        st.markdown(f"**Company ID:** {company_data.name}")
        
        # Description
        description = company_data.get('description', company_data.get('text', 'No description available'))
        if len(str(description)) > 200:
            description = str(description)[:200] + "..."
        st.markdown(f"**About:** {description}")
        
        st.markdown("---")


def display_match_table(
    matches: List[Tuple[int, float, Any]],
    show_top_n: int = 10
):
    """
    Display match results as a formatted table.
    
    Args:
        matches: List of (company_id, score, company_data) tuples
        show_top_n: Number of matches to display
    """
    
    st.markdown(f"### 🎯 Top {show_top_n} Company Matches")
    st.markdown("---")
    
    # Prepare data for table
    table_data = []
    
    for rank, (comp_id, score, comp_data) in enumerate(matches[:show_top_n], 1):
        table_data.append({
            'Rank': f"#{rank}",
            'Company ID': comp_id,
            'Score': f"{score:.1%}",
            'Match Quality': '🔥 Excellent' if score >= 0.7 else '✨ Very Good' if score >= 0.6 else '👍 Good' if score >= 0.5 else '⭐ Fair'
        })
    
    # Display as dataframe
    df = pd.DataFrame(table_data)
    
    # Style the dataframe
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn(width="small"),
            "Score": st.column_config.TextColumn(width="small"),
            "Company ID": st.column_config.TextColumn(width="medium"),
            "Match Quality": st.column_config.TextColumn(width="medium")
        }
    )
    
    st.info("💡 **Tip:** Scores above 0.6 indicate strong alignment between candidate skills and company requirements!")


def display_stats_overview(
    candidate_data,
    matches: List[Tuple[int, float, Any]]
):
    """
    Display overview statistics about the matching results.
    
    Args:
        candidate_data: Candidate information
        matches: List of matches
    """
    
    st.markdown("### 📊 Matching Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Matches",
            len(matches),
            help="Number of companies analyzed"
        )
    
    with col2:
        avg_score = sum(score for _, score, _ in matches) / len(matches) if matches else 0
        st.metric(
            "Average Score",
            f"{avg_score:.1%}",
            help="Average similarity score"
        )
    
    with col3:
        excellent = sum(1 for _, score, _ in matches if score >= 0.7)
        st.metric(
            "Excellent Matches",
            excellent,
            help="Matches with score ≥ 70%"
        )
    
    with col4:
        best_score = max((score for _, score, _ in matches), default=0)
        st.metric(
            "Best Match",
            f"{best_score:.1%}",
            help="Highest similarity score"
        )
    
    st.markdown("---")