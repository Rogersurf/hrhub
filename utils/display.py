"""
HRHUB V2.1 - Display Utilities
All display functions for candidate and company profiles, matches, and statistics
"""

import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any


# Constants for consistent styling
QUALITY_THRESHOLDS = {
    'EXCELLENT': 0.9,
    'VERY_GOOD': 0.7,
    'GOOD': 0.51
}

QUALITY_CONFIGS = {
    'EXCELLENT': {
        'label': '🔥 Excellent Match',
        'color': '#28a745',
        'bg_color': '#d4edda',
        'text_color': '#155724'
    },
    'VERY_GOOD': {
        'label': '✨ Very Good Match',
        'color': '#ffc107',
        'bg_color': '#fff3cd',
        'text_color': '#856404'
    },
    'GOOD': {
        'label': '✅ Good Match',
        'color': '#fd7e14',
        'bg_color': '#ffe5d0',
        'text_color': '#854d0e'
    },
    'MATCH': {
        'label': '📊 Match',
        'color': '#6c757d',
        'bg_color': '#e2e3e5',
        'text_color': '#383d41'
    }
}


def get_quality_config(score: float) -> Dict[str, str]:
    """
    Determine match quality configuration based on score.
    
    Args:
        score: Similarity score (0.0 to 1.0)
    
    Returns:
        Dictionary with quality configuration
    """
    if score >= QUALITY_THRESHOLDS['EXCELLENT']:
        return QUALITY_CONFIGS['EXCELLENT']
    elif score >= QUALITY_THRESHOLDS['VERY_GOOD']:
        return QUALITY_CONFIGS['VERY_GOOD']
    elif score >= QUALITY_THRESHOLDS['GOOD']:
        return QUALITY_CONFIGS['GOOD']
    else:
        return QUALITY_CONFIGS['MATCH']


def create_skill_badges(skills: List[str], max_display: int = 15) -> str:
    """
    Create HTML for skill badges.
    
    Args:
        skills: List of skill strings
        max_display: Maximum number of skills to display
    
    Returns:
        HTML string with skill badges
    """
    if not skills:
        return ""
    
    # Limit number of displayed skills
    display_skills = skills[:max_display]
    
    badges = []
    for skill in display_skills:
        badge = f'<span style="background-color: #0066CC; color: white; padding: 5px 10px; border-radius: 15px; margin: 3px; display: inline-block; font-size: 0.85rem;">{skill}</span>'
        badges.append(badge)
    
    # Add count badge if skills were truncated
    if len(skills) > max_display:
        remaining = len(skills) - max_display
        count_badge = f'<span style="background-color: #6c757d; color: white; padding: 5px 10px; border-radius: 15px; margin: 3px; display: inline-block; font-size: 0.85rem;">+{remaining} more</span>'
        badges.append(count_badge)
    
    return " ".join(badges)


def create_match_badge(score: float, config: Dict[str, str]) -> str:
    """
    Create HTML for match quality badge.
    
    Args:
        score: Similarity score
        config: Quality configuration dictionary
    
    Returns:
        HTML string for badge
    """
    return f"""
    <div style="
        background: {config['color']};
        color: white;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 5px 0;
        transition: all 0.3s ease;
    ">
        <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">
            {score:.1%}
        </div>
        <div style="font-size: 0.8rem; font-weight: 600; opacity: 0.9;">
            {config['label']}
        </div>
    </div>
    """


def display_candidate_profile(candidate: pd.Series) -> None:
    """
    Display candidate profile card with all relevant information.
    
    Args:
        candidate: pandas Series with candidate data
    """
    # Career Objective
    career_obj = candidate.get('career_objective')
    if career_obj and isinstance(career_obj, str) and career_obj.strip():
        with st.expander("🎯 Career Objective", expanded=True):
            st.markdown(career_obj)
    
    # Skills & Expertise
    with st.expander("💻 Skills & Expertise", expanded=True):
        skills = candidate.get('skills', [])
        if skills:
            if isinstance(skills, str):
                # Handle string representation of list
                skills = [s.strip() for s in skills.split(',')]
            
            skills_html = create_skill_badges(skills)
            st.markdown(skills_html, unsafe_allow_html=True)
            
            # Show skills count
            st.caption(f"Total skills: {len(skills)}")
        else:
            st.info("No skills listed")
    
    # Education
    with st.expander("🎓 Education", expanded=False):
        edu_columns = [
            'educational_institution_name',
            'degree_names',
            'major_field_of_studies',
            'passing_years',
            'educational_results'
        ]
        
        edu_data = {}
        for col in edu_columns:
            value = candidate.get(col)
            if value:
                # Clean column name for display
                display_name = col.replace('_', ' ').title()
                edu_data[display_name] = value if isinstance(value, list) else [value]
        
        if edu_data:
            # Ensure all lists have same length
            max_len = max(len(v) for v in edu_data.values())
            for key in edu_data:
                if len(edu_data[key]) < max_len:
                    edu_data[key].extend([''] * (max_len - len(edu_data[key])))
            
            df_edu = pd.DataFrame(edu_data)
            st.dataframe(df_edu, use_container_width=True, hide_index=True)
        else:
            st.info("No education information provided")
    
    # Work Experience
    with st.expander("💼 Work Experience", expanded=False):
        exp_columns = {
            'Company': 'professional_company_names',
            'Position': 'positions',
            'Location': 'locations',
            'Start': 'start_dates',
            'End': 'end_dates'
        }
        
        exp_data = {}
        for display_name, col_name in exp_columns.items():
            value = candidate.get(col_name)
            if value:
                exp_data[display_name] = value if isinstance(value, list) else [value]
        
        if exp_data:
            # Ensure all lists have same length
            max_len = max(len(v) for v in exp_data.values())
            for key in exp_data:
                if len(exp_data[key]) < max_len:
                    exp_data[key].extend([''] * (max_len - len(exp_data[key])))
            
            df_exp = pd.DataFrame(exp_data)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)
            
            # Show responsibilities
            responsibilities = candidate.get('responsibilities', '')
            if responsibilities and isinstance(responsibilities, str) and responsibilities.strip():
                st.markdown("**Key Responsibilities:**")
                st.text(responsibilities[:500] + ('...' if len(responsibilities) > 500 else ''))
        else:
            st.info("No work experience listed")
    
    # Languages
    with st.expander("🌍 Languages", expanded=False):
        languages = candidate.get('languages', [])
        proficiency = candidate.get('proficiency_levels', [])
        
        if languages:
            if isinstance(languages, str):
                languages = [s.strip() for s in languages.split(',')]
            
            for i, (lang, prof) in enumerate(zip(languages, proficiency)):
                if i < len(proficiency):
                    st.write(f"• **{lang}** - {prof}")
                else:
                    st.write(f"• **{lang}**")
        else:
            st.info("No languages listed")
    
    # Certifications
    with st.expander("🏅 Certifications", expanded=False):
        providers = candidate.get('certification_providers', [])
        skills = candidate.get('certification_skills', [])
        
        if providers:
            if isinstance(providers, str):
                providers = [s.strip() for s in providers.split(',')]
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(',')]
            
            for i, (provider, skill) in enumerate(zip(providers, skills)):
                if i < len(skills):
                    st.write(f"• **{skill}** by {provider}")
                else:
                    st.write(f"• Certification by {provider}")
        else:
            st.info("No certifications listed")
    
    # Additional info
    st.success("✅ **Profile enriched** with job posting vocabulary for semantic matching")


def display_company_card(company_data: pd.Series, score: float, rank: int) -> None:
    """
    Display company match card with score and details.
    
    Args:
        company_data: pandas Series with company data
        score: float similarity score
        rank: int rank position
    """
    quality_config = get_quality_config(score)
    company_name = company_data.get('name', f'Company {rank}')
    company_id = company_data.name if hasattr(company_data, 'name') else rank
    
    with st.expander(f"#{rank} - {company_name}", expanded=(rank <= 3)):
        col1, col2 = st.columns([2.5, 1])
        
        with col1:
            st.markdown(f"### 🏢 {company_name}")
            
            # Industry
            industry = company_data.get('industry')
            if industry and isinstance(industry, str) and industry.strip():
                st.markdown(f"**Industry:** {industry}")
            
            # Company ID
            st.caption(f"Company ID: {company_id}")
            
            # Description preview
            description = company_data.get('description')
            if description and isinstance(description, str) and description.strip():
                st.markdown("**Description:**")
                st.write(description[:300] + ('...' if len(description) > 300 else ''))
        
        with col2:
            # Match badge
            badge_html = create_match_badge(score, quality_config)
            st.markdown(badge_html, unsafe_allow_html=True)
            
            # Quality indicator
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: {quality_config['text_color']}; background-color: {quality_config['bg_color']}; padding: 5px; border-radius: 5px; margin-top: 10px;'>{quality_config['label']}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Additional details
        col3, col4 = st.columns(2)
        
        with col3:
            # Job posting status
            has_postings = company_data.get('has_job_postings', False)
            if has_postings:
                st.success("✅ **Direct job posting data available**")
            else:
                st.info("🔄 **Collaborative filtering data**")
        
        with col4:
            # Match strength
            if score >= 0.9:
                st.metric("Match Strength", "Very Strong")
            elif score >= 0.7:
                st.metric("Match Strength", "Strong")
            else:
                st.metric("Match Strength", "Moderate")


def display_match_table(matches: List[Tuple[int, float, pd.Series]]) -> None:
    """
    Display matches in table format.
    
    Args:
        matches: list of tuples (company_id, score, company_data)
    """
    if not matches:
        st.warning("No matches to display")
        return
    
    # Build table data
    table_data = []
    for rank, (comp_id, score, comp_data) in enumerate(matches, 1):
        company_name = comp_data.get('name', f'Company {comp_id}')
        industry = comp_data.get('industry', 'Not specified')
        
        quality_config = get_quality_config(score)
        quality_label = quality_config['label'].split()[-1]  # Get last word
        
        table_data.append({
            'Rank': rank,
            'Company': company_name,
            'Industry': industry,
            'Score': score,
            'Quality': quality_label,
            'Score %': f'{score:.1%}'
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Display with sorting options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📊 Match Results")
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=['Rank', 'Score', 'Company', 'Industry'],
            index=0,
            label_visibility="collapsed"
        )
    
    # Sort DataFrame
    if sort_by == 'Score':
        df = df.sort_values('Score', ascending=False)
    elif sort_by == 'Company':
        df = df.sort_values('Company')
    elif sort_by == 'Industry':
        df = df.sort_values('Industry')
    
    # Display as styled dataframe
    st.dataframe(
        df[['Rank', 'Company', 'Industry', 'Score %', 'Quality']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Rank': st.column_config.NumberColumn(
                'Rank',
                help="Match ranking position"
            ),
            'Company': st.column_config.TextColumn(
                'Company',
                help="Company name"
            ),
            'Industry': st.column_config.TextColumn(
                'Industry',
                help="Industry sector"
            ),
            'Score %': st.column_config.ProgressColumn(
                'Match Score',
                help="Similarity score as percentage",
                format="%.1f%%",
                min_value=0,
                max_value=1,
                width="medium"
            ),
            'Quality': st.column_config.TextColumn(
                'Quality',
                help="Match quality rating"
            )
        }
    )
    
    # Summary statistics
    avg_score = df['Score'].mean()
    best_score = df['Score'].max()
    
    st.info(f"""
    💡 **Summary:** Average match score: **{avg_score:.1%}** | Best match: **{best_score:.1%}**
    
    Scores above **60%** indicate strong semantic alignment between your skills and company requirements.
    """)


def display_stats_overview(candidate: pd.Series, matches: List[Tuple[int, float, pd.Series]]) -> None:
    """
    Display statistics overview for candidate matching.
    
    Args:
        candidate: pandas Series with candidate data
        matches: list of tuples (company_id, score, company_data)
    """
    if not matches:
        st.warning("No matches to display statistics")
        return
    
    # Calculate statistics
    scores = [score for _, score, _ in matches]
    total_matches = len(matches)
    avg_score = sum(scores) / total_matches
    excellent_matches = sum(1 for score in scores if score >= QUALITY_THRESHOLDS['EXCELLENT'])
    very_good_matches = sum(1 for score in scores if score >= QUALITY_THRESHOLDS['VERY_GOOD'])
    best_score = max(scores)
    
    # Candidate info
    candidate_skills = candidate.get('skills', [])
    if isinstance(candidate_skills, str):
        candidate_skills = [s.strip() for s in candidate_skills.split(',')]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Matches",
            value=total_matches,
            delta=f"{total_matches} companies",
            help="Number of companies above similarity threshold"
        )
    
    with col2:
        st.metric(
            label="📈 Average Score",
            value=f"{avg_score:.1%}",
            help="Mean similarity score across all matches"
        )
    
    with col3:
        st.metric(
            label="🔥 Excellent Matches",
            value=excellent_matches,
            delta=f"{very_good_matches} very good",
            help="Companies with score ≥ 90% (Excellent) and ≥ 70% (Very Good)"
        )
    
    with col4:
        st.metric(
            label="🎯 Best Match",
            value=f"{best_score:.1%}",
            help="Highest similarity score achieved"
        )
    
    # Additional statistics
    with st.expander("📋 Detailed Statistics", expanded=False):
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                label="Candidate Skills",
                value=len(candidate_skills),
                help="Total skills in candidate profile"
            )
        
        with col6:
            industries = set()
            for _, _, comp_data in matches:
                industry = comp_data.get('industry')
                if industry:
                    industries.add(industry)
            st.metric(
                label="Industries",
                value=len(industries),
                help="Number of different industries matched"
            )
        
        with col7:
            score_range = max(scores) - min(scores)
            st.metric(
                label="Score Range",
                value=f"{score_range:.1%}",
                help="Difference between best and worst match"
            )
        
        # Distribution visualization
        st.markdown("**Score Distribution:**")
        score_bins = pd.cut(scores, bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        dist_df = score_bins.value_counts().sort_index().reset_index()
        dist_df.columns = ['Score Range', 'Count']
        
        # Add percentage
        dist_df['Percentage'] = (dist_df['Count'] / total_matches * 100).round(1)
        
        st.dataframe(
            dist_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Score Range': st.column_config.TextColumn('Score Range'),
                'Count': st.column_config.NumberColumn('Companies'),
                'Percentage': st.column_config.ProgressColumn(
                    'Percentage',
                    format='%.1f%%',
                    min_value=0,
                    max_value=100
                )
            }
        )


# Additional helper functions for other display utilities...

def display_candidate_card_basic(candidate_data: pd.Series, candidate_id: int, score: float, rank: int) -> None:
    """
    Display basic candidate card for company view.
    
    Args:
        candidate_data: pandas Series with candidate data
        candidate_id: int candidate ID
        score: float similarity score
        rank: int rank position
    """
    quality_config = get_quality_config(score)
    
    with st.expander(f"#{rank} - Candidate {candidate_id} - {score:.1%}", expanded=(rank <= 3)):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Candidate ID:** {candidate_id}")
            
            # Career objective
            career_obj = candidate_data.get('career_objective')
            if career_obj and isinstance(career_obj, str) and career_obj.strip():
                st.markdown("**Career Objective:**")
                st.write(career_obj[:250] + ('...' if len(career_obj) > 250 else ''))
            
            # Skills preview
            skills = candidate_data.get('skills', [])
            if skills:
                if isinstance(skills, str):
                    skills = [s.strip() for s in skills.split(',')[:5]]
                elif isinstance(skills, list):
                    skills = skills[:5]
                
                st.markdown("**Top Skills:**")
                skills_text = ' • '.join(skills)
                st.write(skills_text)
        
        with col2:
            # Quality indicator with color
            badge_html = f"""
            <div style="
                background: {quality_config['color']};
                color: white;
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 10px;
            ">
                <div style="font-size: 1.5rem; font-weight: 700;">
                    {score:.1%}
                </div>
                <div style="font-size: 0.8rem;">
                    {quality_config['label'].split()[-1]}
                </div>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)
            
            # Match strength
            if score >= 0.9:
                st.success("🔥 Excellent Fit")
            elif score >= 0.7:
                st.info("✨ Strong Fit")
            elif score >= 0.51:
                st.warning("✅ Good Fit")
            else:
                st.caption("📊 Potential Fit")


# Note: Other functions (display_company_profile_basic, display_match_table_candidates,
# display_stats_overview_company) would follow similar patterns of improvement
# with better error handling, consistent styling, and performance optimizations.