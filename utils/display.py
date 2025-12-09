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
    if 'skills' in candidate and candidate['skills']:
        with st.expander("🛠️ Skills & Expertise", expanded=True):
            skills_text = candidate['skills']
            if isinstance(skills_text, str):
                # Try to split into badges if comma-separated
                if ',' in skills_text:
                    skills_list = [s.strip() for s in skills_text.split(',')[:15]]  # Limit to 15
                    
                    # Display as badges in columns
                    cols = st.columns(3)
                    for idx, skill in enumerate(skills_list):
                        with cols[idx % 3]:
                            st.markdown(f"**`{skill}`**")
                else:
                    st.write(skills_text[:300] + ('...' if len(skills_text) > 300 else ''))
    
    # Education
    if 'education' in candidate and candidate['education']:
        with st.expander("🎓 Education", expanded=False):
            st.write(candidate['education'])
    
    # Work Experience
    if 'experience' in candidate and candidate['experience']:
        with st.expander("💼 Work Experience", expanded=False):
            exp_text = candidate['experience']
            if isinstance(exp_text, str):
                st.write(exp_text[:400] + ('...' if len(exp_text) > 400 else ''))
    
    # Additional info box
    st.info("💡 **Profile enriched** with job posting vocabulary for semantic matching")


def display_company_card(company_data, score, rank):
    """
    Display company match card with score and details.
    
    Args:
        company_data: pandas Series with company data
        score: float similarity score
        rank: int rank position
    """
    # Determine match quality
    if score >= 0.7:
        quality = "🔥 Excellent Match"
        color = "green"
    elif score >= 0.6:
        quality = "✨ Very Good Match"
        color = "blue"
    else:
        quality = "✅ Good Match"
        color = "orange"
    
    # Get company name
    company_name = company_data.get('name', f'Company {rank}')
    company_id = company_data.name if hasattr(company_data, 'name') else rank
    
    # Card expander
    with st.expander(f"#{rank} - {company_name} - {score:.1%}", expanded=(rank <= 3)):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Company:** {company_name}")
            st.markdown(f"**Match Score:** {score:.1%}")
            
            # Industry
            if 'industry' in company_data and company_data['industry']:
                st.markdown(f"**Industry:** {company_data['industry']}")
            
            # Description/About
            if 'description' in company_data and company_data['description']:
                desc = company_data['description']
                if isinstance(desc, str) and len(desc) > 0:
                    st.markdown("**About:**")
                    st.write(desc[:250] + ('...' if len(desc) > 250 else ''))
            
            # Job postings indicator
            if 'has_job_postings' in company_data:
                if company_data['has_job_postings']:
                    st.caption("✅ Direct job posting data")
                else:
                    st.caption("🔄 Collaborative filtering")
        
        with col2:
            # Match quality badge
            if color == "green":
                st.success(quality)
            elif color == "blue":
                st.info(quality)
            else:
                st.warning(quality)
            
            # Company ID
            st.caption(f"ID: {company_id}")


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
        
        # Match quality
        if score >= 0.7:
            quality = "🔥 Excellent"
        elif score >= 0.6:
            quality = "✨ Very Good"
        else:
            quality = "✅ Good"
        
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
    
    # Calculate stats
    total_matches = len(matches)
    avg_score = sum(score for _, score, _ in matches) / total_matches
    excellent_matches = sum(1 for _, score, _ in matches if score >= 0.7)
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
            help="Companies with score ≥ 70%"
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
    # Determine match quality
    if score >= 0.7:
        quality = "🔥 Excellent"
        color = "green"
    elif score >= 0.6:
        quality = "✨ Very Good"
        color = "blue"
    else:
        quality = "✅ Good"
        color = "orange"
    
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
        # Match quality
        if score >= 0.7:
            quality = "🔥 Excellent"
        elif score >= 0.6:
            quality = "✨ Very Good"
        else:
            quality = "✅ Good"
        
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
    
    # Calculate stats
    total_matches = len(matches)
    avg_score = sum(score for _, score, _ in matches) / total_matches
    excellent_matches = sum(1 for _, score, _ in matches if score >= 0.7)
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
            help="Candidates with score ≥ 70%"
        )
    
    with col4:
        st.metric(
            "🎯 Best Match",
            f"{best_score:.1%}",
            help="Highest similarity score achieved"
        )
