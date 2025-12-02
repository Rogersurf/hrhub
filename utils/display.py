"""
Display utilities for HRHUB Streamlit UI.
Contains formatted display components for candidates and companies.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple


def display_candidate_profile(candidate: Dict[str, Any]):
    """
    Display comprehensive candidate profile in Streamlit.
    
    Args:
        candidate: Dictionary with candidate data
    """
    
    st.markdown("### 👤 Candidate Profile")
    st.markdown("---")
    
    # Basic Info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Name:** {candidate.get('name', 'N/A')}")
        st.markdown(f"**Desired Position:** {candidate.get('job_position_name', 'N/A')}")
        
    with col2:
        st.metric("Match Score", f"{candidate.get('matched_score', 0):.2%}")
    
    # Career Objective
    with st.expander("🎯 Career Objective", expanded=True):
        st.write(candidate.get('career_objective', 'Not provided'))
    
    # Skills
    with st.expander("💻 Skills & Expertise", expanded=True):
        skills = candidate.get('skills', [])
        if skills:
            # Display as tags
            skills_html = " ".join([f'<span style="background-color: #0066CC; color: white; padding: 5px 10px; border-radius: 15px; margin: 3px; display: inline-block;">{skill}</span>' for skill in skills[:15]])
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.write("No skills listed")
    
    # Education
    with st.expander("🎓 Education"):
        edu_data = {
            'Institution': candidate.get('educational_institution_name', []),
            'Degree': candidate.get('degree_names', []),
            'Major': candidate.get('major_field_of_studies', []),
            'Year': candidate.get('passing_years', []),
            'GPA': candidate.get('educational_results', [])
        }
        
        if any(edu_data.values()):
            df_edu = pd.DataFrame(edu_data)
            st.dataframe(df_edu, use_container_width=True, hide_index=True)
        else:
            st.write("No education information provided")
    
    # Work Experience
    with st.expander("💼 Work Experience"):
        exp_data = {
            'Company': candidate.get('professional_company_names', []),
            'Position': candidate.get('positions', []),
            'Location': candidate.get('locations', []),
            'Start': candidate.get('start_dates', []),
            'End': candidate.get('end_dates', [])
        }
        
        if any(exp_data.values()):
            df_exp = pd.DataFrame(exp_data)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)
            
            # Show responsibilities
            responsibilities = candidate.get('responsibilities', '')
            if responsibilities:
                st.markdown("**Key Responsibilities:**")
                st.text(responsibilities)
        else:
            st.write("No work experience listed")
    
    # Languages
    with st.expander("🌍 Languages"):
        languages = candidate.get('languages', [])
        proficiency = candidate.get('proficiency_levels', [])
        
        if languages:
            for lang, prof in zip(languages, proficiency):
                st.write(f"• **{lang}** - {prof}")
        else:
            st.write("No languages listed")
    
    # Certifications
    with st.expander("🏅 Certifications"):
        providers = candidate.get('certification_providers', [])
        skills = candidate.get('certification_skills', [])
        
        if providers:
            for provider, skill in zip(providers, skills):
                st.write(f"• **{skill}** by {provider}")
        else:
            st.write("No certifications listed")


def display_company_card(
    company_data: Dict[str, Any],
    similarity_score: float,
    rank: int
):
    """
    Display company information as a card.
    
    Args:
        company_data: Dictionary with company data
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**📍 Location**")
            location = f"{company_data.get('city', '')}, {company_data.get('state', '')}, {company_data.get('country', '')}"
            st.write(location)
        
        with col2:
            st.markdown(f"**👥 Size**")
            st.write(company_data.get('employee_count', 'N/A'))
        
        with col3:
            st.markdown(f"**🏭 Industry**")
            industries = company_data.get('industries_list', 'N/A')
            st.write(industries.split(',')[0] if ',' in str(industries) else industries)
        
        # Description
        description = company_data.get('description', 'No description available')
        st.markdown(f"**About:** {description}")
        
        # Required skills
        required_skills = company_data.get('required_skills', '')
        if required_skills:
            st.markdown("**🔧 Required Skills:**")
            skills_list = [s.strip() for s in str(required_skills).split('|')[:8]]
            skills_html = " ".join([f'<span style="background-color: #CC0000; color: white; padding: 5px 10px; border-radius: 15px; margin: 3px; display: inline-block; font-size: 12px;">{skill}</span>' for skill in skills_list])
            st.markdown(skills_html, unsafe_allow_html=True)
        
        # Job postings
        job_titles = company_data.get('posted_job_titles', '')
        if job_titles:
            st.markdown(f"**💼 Open Positions:** {job_titles}")
        
        st.markdown("---")


def display_match_table(
    matches: List[Tuple[int, float, Dict[str, Any]]],
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
        # Get key skills (first 3)
        skills = comp_data.get('required_skills', 'N/A')
        if skills and skills != 'N/A':
            skills_list = [s.strip() for s in str(skills).split('|')[:3]]
            skills_display = ', '.join(skills_list)
        else:
            skills_display = 'N/A'
        
        table_data.append({
            'Rank': f"#{rank}",
            'Company': comp_data.get('name', 'N/A'),
            'Score': f"{score:.1%}",
            'Location': f"{comp_data.get('city', 'N/A')}, {comp_data.get('state', 'N/A')}",
            'Top Skills': skills_display,
            'Employees': comp_data.get('employee_count', 'N/A')
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
            "Company": st.column_config.TextColumn(width="medium"),
            "Location": st.column_config.TextColumn(width="medium"),
            "Top Skills": st.column_config.TextColumn(width="large"),
            "Employees": st.column_config.TextColumn(width="small")
        }
    )
    
    st.info("💡 **Tip:** Scores above 0.6 indicate strong alignment between candidate skills and company requirements!")


def display_stats_overview(
    candidate_data: Dict[str, Any],
    matches: List[Tuple[int, float, Dict[str, Any]]]
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
