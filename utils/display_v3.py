"""
HRHUB V2.1 - Display Utilities
All display functions for candidate and company profiles, matches, and stats
"""

import streamlit as st
import pandas as pd


def display_candidate_profile(candidate):
    """
    Display candidate profile card with all relevant information.
    
    Args:
        candidate: pandas Series with candidate data
    """
    # Career Objective
    if 'career_objective' in candidate and candidate['career_objective']:
        with st.expander("🎯 Career Objective", expanded=True):
            st.write(candidate['career_objective'])
    
    # Skills & Expertise
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
    
    # Additional info box
    st.info("💡 **Profile enriched** with job posting vocabulary for semantic matching")


def display_company_card(company_data, score, rank):
    """
    Display company match card with score and details (matching reference design).

    Args:
        company_data: pandas Series with company data
        score: float similarity score
        rank: int rank position
    """
    # Determine match quality and color (updated thresholds)
    if score >= 0.9:
        quality = "Excellent Match"
        badge_bg = "#28a745"  # Green
        badge_text = "Excellent Match"
    elif score >= 0.7:
        quality = "Very Good Match"
        badge_bg = "#ffc107"  # Yellow
        badge_text = "Very Good Match"
    elif score >= 0.51:
        quality = "Good Match"
        badge_bg = "#fd7e14"  # Orange
        badge_text = "Good Match"
    else:
        quality = "Match"
        badge_bg = "#6c757d"  # Gray
        badge_text = "Match"

    # Get company name
    company_name = company_data.get('name', f'Company {rank}')
    company_id = company_data.name if hasattr(company_data, 'name') else rank

    # Card expander (best of both worlds design)
    with st.expander(f"#{rank} - {company_name}", expanded=(rank <= 3)):
        # Layout: Company info on left, Badge on right
        col1, col2 = st.columns([2.5, 1])

        with col1:
            st.markdown(f"### 🏢 {company_name}")

            # Industry inline
            if 'industry' in company_data and company_data['industry']:
                st.markdown(f"**Industry:** {company_data['industry']}")

            st.markdown(f"*Company ID: {company_id}*")

        with col2:
            # Large colored badge (clean, modern design)
            badge_html = f"""
            <div style="
                background: {badge_bg};
                color: white;
                padding: 20px;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                margin: 5px 0;
            ">
                <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">
                    {score:.1%}
                </div>
                <div style="font-size: 0.8rem; font-weight: 600; opacity: 0.9;">
                    {badge_text}
                </div>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown("---")

        # Description/About (full width)
        if 'description' in company_data and company_data['description']:
            desc = company_data['description']
            if isinstance(desc, str) and len(desc) > 0:
                st.markdown("**📄 About:**")
                st.write(desc[:350] + ('...' if len(desc) > 350 else ''))

        # Job postings indicator
        if 'has_job_postings' in company_data:
            if company_data['has_job_postings']:
                st.info("✅ Direct job posting data available")
            else:
                st.caption("🔄 Data from collaborative filtering")


def display_match_table(matches):
    """
    Display matches in table format.
    
    Args:
        matches: list of tuples (company_id, score, company_data)
    """
    if len(matches) == 0:
        st.warning("No matches to display")
        return
    
    # Build table data
    table_data = []
    for rank, (comp_id, score, comp_data) in enumerate(matches, 1):
        company_name = comp_data.get('name', f'Company {comp_id}')
        industry = comp_data.get('industry', 'N/A')

        # Match quality (updated thresholds)
        if score >= 0.9:
            quality = "🔥 Excellent"
        elif score >= 0.7:
            quality = "✨ Very Good"
        elif score >= 0.51:
            quality = "✅ Good"
        else:
            quality = "📊 Match"
        
        table_data.append({
            'Rank': f'#{rank}',
            'Company': company_name,
            'Industry': industry,
            'Score': f'{score:.1%}',
            'Quality': quality
        })
    
    # Display as dataframe
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Add info tip
    st.info("💡 **Tip:** Scores above 0.6 indicate strong semantic alignment between your skills and company requirements!")


def display_stats_overview(candidate, matches):
    """
    Display statistics overview for candidate matching.

    Args:
        candidate: pandas Series with candidate data
        matches: list of tuples (company_id, score, company_data)
    """
    if len(matches) == 0:
        st.warning("No matches to display statistics")
        return

    # Calculate stats (updated thresholds)
    total_matches = len(matches)
    avg_score = sum(score for _, score, _ in matches) / total_matches
    excellent_matches = sum(1 for _, score, _ in matches if score >= 0.9)  # Updated to 90%
    best_score = max(score for _, score, _ in matches)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Matches",
            total_matches,
            help="Number of companies above minimum threshold"
        )
    
    with col2:
        st.metric(
            "📈 Average Score",
            f"{avg_score:.1%}",
            help="Mean similarity score across all matches"
        )
    
    with col3:
        st.metric(
            "🔥 Excellent Matches",
            excellent_matches,
            help="Companies with score ≥ 90%"
        )
    
    with col4:
        st.metric(
            "🎯 Best Match",
            f"{best_score:.1%}",
            help="Highest similarity score achieved"
        )


def display_candidate_card_basic(candidate_data, candidate_id, score, rank):
    """
    Display basic candidate card for company view.

    Args:
        candidate_data: pandas Series with candidate data
        candidate_id: int candidate ID
        score: float similarity score
        rank: int rank position
    """
    # Determine match quality (updated thresholds)
    if score >= 0.9:
        quality = "🔥 Excellent"
        color = "green"
    elif score >= 0.7:
        quality = "✨ Very Good"
        color = "blue"
    elif score >= 0.51:
        quality = "✅ Good"
        color = "orange"
    else:
        quality = "📊 Match"
        color = "gray"
    
    # Card expander
    with st.expander(f"#{rank} - Candidate {candidate_id} - {score:.1%}", expanded=(rank <= 3)):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Candidate ID:** {candidate_id}")
            st.markdown(f"**Match Score:** {score:.1%}")
            
            # Career objective
            if 'career_objective' in candidate_data and candidate_data['career_objective']:
                obj = candidate_data['career_objective']
                if isinstance(obj, str) and len(obj) > 0:
                    st.markdown("**Career Objective:**")
                    st.write(obj[:200] + ('...' if len(obj) > 200 else ''))
            
            # Skills
            if 'skills' in candidate_data and candidate_data['skills']:
                skills = candidate_data['skills']
                if isinstance(skills, str) and len(skills) > 0:
                    st.markdown("**Skills:**")
                    # Show first few skills as badges
                    if ',' in skills:
                        skills_list = [s.strip() for s in skills.split(',')[:8]]
                        st.markdown(' • '.join(skills_list))
                    else:
                        st.write(skills[:200] + ('...' if len(skills) > 200 else ''))
            
            # Experience
            if 'experience' in candidate_data and candidate_data['experience']:
                exp = candidate_data['experience']
                if isinstance(exp, str) and len(exp) > 0:
                    st.markdown("**Experience:**")
                    st.write(exp[:150] + ('...' if len(exp) > 150 else ''))
        
        with col2:
            # Match quality badge
            if color == "green":
                st.success(quality)
            elif color == "blue":
                st.info(quality)
            else:
                st.warning(quality)


def display_company_profile_basic(company_data, company_id):
    """
    Display basic company profile card.
    
    Args:
        company_data: pandas Series with company data
        company_id: int company ID
    """
    st.markdown(f"**Company ID:** {company_id}")
    
    # Name
    if 'name' in company_data and company_data['name']:
        st.markdown(f"**Name:** {company_data['name']}")
    
    # Industry
    if 'industry' in company_data and company_data['industry']:
        st.markdown(f"**Industry:** {company_data['industry']}")
    
    # Description
    if 'description' in company_data and company_data['description']:
        desc = company_data['description']
        if isinstance(desc, str) and len(desc) > 0:
            with st.expander("📄 Company Description", expanded=False):
                st.write(desc[:500] + ('...' if len(desc) > 500 else ''))
    
    # Job posting status
    has_postings = company_data.get('has_job_postings', True)
    
    st.markdown("---")
    
    if has_postings:
        st.success("✅ **Has job postings** (direct semantic data)")
    else:
        st.info("🔄 **Collaborative filtering** (skills inherited from similar companies)")
    
    st.caption("💡 Company profile enriched with job posting vocabulary")


def display_match_table_candidates(matches):
    """
    Display candidate matches in table format (for company view).
    
    Args:
        matches: list of tuples (candidate_id, score, candidate_data)
    """
    if len(matches) == 0:
        st.warning("No matches to display")
        return
    
    # Build table data
    table_data = []
    for rank, (cand_id, score, cand_data) in enumerate(matches, 1):
        # Match quality (updated thresholds)
        if score >= 0.9:
            quality = "🔥 Excellent"
        elif score >= 0.7:
            quality = "✨ Very Good"
        elif score >= 0.51:
            quality = "✅ Good"
        else:
            quality = "📊 Match"
        
        # Get some candidate info
        skills_preview = ""
        if 'skills' in cand_data and cand_data['skills']:
            skills = cand_data['skills']
            if isinstance(skills, str) and len(skills) > 0:
                if ',' in skills:
                    skills_list = [s.strip() for s in skills.split(',')[:3]]
                    skills_preview = ', '.join(skills_list) + '...'
                else:
                    skills_preview = skills[:50] + ('...' if len(skills) > 50 else '')
        
        table_data.append({
            'Rank': f'#{rank}',
            'Candidate ID': cand_id,
            'Skills Preview': skills_preview if skills_preview else 'N/A',
            'Score': f'{score:.1%}',
            'Quality': quality
        })
    
    # Display as dataframe
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Add info tip
    st.info("💡 **Tip:** Scores above 0.6 indicate strong alignment between candidate skills and company requirements!")


def display_stats_overview_company(company, matches):
    """
    Display statistics overview for company matching (company view).

    Args:
        company: pandas Series with company data
        matches: list of tuples (candidate_id, score, candidate_data)
    """
    if len(matches) == 0:
        st.warning("No matches to display statistics")
        return

    # Calculate stats (updated thresholds)
    total_matches = len(matches)
    avg_score = sum(score for _, score, _ in matches) / total_matches
    excellent_matches = sum(1 for _, score, _ in matches if score >= 0.9)  # Updated to 90%
    best_score = max(score for _, score, _ in matches)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Candidates",
            total_matches,
            help="Number of candidates above minimum threshold"
        )
    
    with col2:
        st.metric(
            "📈 Average Score",
            f"{avg_score:.1%}",
            help="Mean similarity score across all candidates"
        )
    
    with col3:
        st.metric(
            "🔥 Excellent Matches",
            excellent_matches,
            help="Candidates with score ≥ 90%"
        )
    
    with col4:
        st.metric(
            "🎯 Best Match",
            f"{best_score:.1%}",
            help="Highest similarity score achieved"
        )
