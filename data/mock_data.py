"""
Mock data for HRHUB demo.
This file contains hardcoded data for MVP demonstration.

TO SWITCH TO REAL DATA:
Replace imports in app.py:
    from data.mock_data import get_candidate_data, get_company_matches
    ↓
    from data.data_loader import get_candidate_data, get_company_matches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


def get_candidate_data(candidate_id: int = 0) -> Dict[str, Any]:
    """
    Get candidate data by ID.
    
    Args:
        candidate_id: Candidate identifier (0 for demo)
    
    Returns:
        Dictionary with candidate information
    """
    
    # Mock candidate data (based on your actual structure)
    candidate = {
        'id': 0,
        'name': 'Demo Candidate #0',
        
        # Skills & Expertise
        'skills': [
            'Python', 'Machine Learning', 'Data Science', 'SQL', 'TensorFlow',
            'Pandas', 'NumPy', 'Scikit-learn', 'Deep Learning', 'NLP',
            'Computer Vision', 'AWS', 'Docker', 'Git', 'Agile'
        ],
        
        # Education
        'educational_institution_name': ['Technical University of Denmark'],
        'degree_names': ['Master of Science'],
        'passing_years': ['2023'],
        'educational_results': ['3.8'],
        'result_types': ['GPA'],
        'major_field_of_studies': ['Business Data Science'],
        
        # Work Experience
        'professional_company_names': ['TechCorp', 'DataHub', 'AI Solutions'],
        'company_urls': ['techcorp.com', 'datahub.io', 'aisolutions.ai'],
        'start_dates': ['Jan 2021', 'Jun 2019', 'Jan 2018'],
        'end_dates': ['Current', 'Dec 2020', 'May 2019'],
        'positions': ['Data Scientist', 'ML Engineer', 'Data Analyst'],
        'locations': ['Copenhagen, Denmark', 'Aalborg, Denmark', 'Aarhus, Denmark'],
        'responsibilities': """
            • Developed ML models for customer segmentation
            • Built NLP pipeline for sentiment analysis
            • Deployed models to production using AWS
            • Collaborated with cross-functional teams
            • Mentored junior data scientists
        """,
        
        # Additional Info
        'languages': ['English', 'Danish', 'Portuguese'],
        'proficiency_levels': ['Fluent', 'Native', 'Native'],
        'certification_providers': ['AWS', 'Google Cloud', 'Coursera'],
        'certification_skills': ['AWS ML Specialty', 'GCP Data Engineer', 'Deep Learning'],
        
        # Career Goals
        'career_objective': 'Seeking senior data science role focusing on NLP and LLM applications',
        'job_position_name': 'Senior Data Scientist / ML Engineer',
        
        # Match score (for demo purposes)
        'matched_score': 0.85,
        
        # Text representation (what gets embedded)
        'text': """
            Skills: Python, Machine Learning, Data Science, SQL, TensorFlow, Pandas, NumPy, 
            Scikit-learn, Deep Learning, NLP, Computer Vision, AWS, Docker, Git, Agile.
            
            Education: Master of Science in Business Data Science from Technical University of Denmark (2023).
            
            Experience: Data Scientist at TechCorp (Current), ML Engineer at DataHub, Data Analyst at AI Solutions.
            Specialized in ML model development, NLP, and production deployment.
            
            Languages: English (Fluent), Danish (Native), Portuguese (Native).
            
            Certifications: AWS ML Specialty, GCP Data Engineer, Deep Learning.
        """
    }
    
    return candidate


def get_company_matches(candidate_id: int = 0, top_k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    Get top company matches for a candidate.
    
    Args:
        candidate_id: Candidate identifier
        top_k: Number of top matches to return
    
    Returns:
        List of tuples: (company_id, similarity_score, company_data)
    """
    
    # Mock company matches
    companies = [
        {
            'id': 29286,
            'name': 'Anblicks',
            'similarity_score': 0.7028,
            'description': 'Leading data analytics and AI consulting firm specializing in cloud-native solutions',
            'industries_list': 'Information Technology, Data Analytics, Cloud Computing',
            'specialties_list': 'Big Data | Machine Learning | Cloud Architecture | Data Engineering',
            'employee_count': '500-1000',
            'city': 'San Francisco',
            'state': 'CA',
            'country': 'USA',
            'required_skills': 'Python | Machine Learning | AWS | TensorFlow | Data Science | SQL | Spark',
            'posted_job_titles': 'Senior Data Scientist | ML Engineer | Data Architect',
            'experience_levels': 'Mid-Senior level | Senior level',
            'work_types': 'Full-time | Remote',
            'text': 'Technology company seeking ML experts with Python, AWS, and production experience...'
        },
        {
            'id': 15234,
            'name': 'iO Associates - US',
            'similarity_score': 0.7026,
            'description': 'Global talent solutions provider connecting tech professionals with innovative companies',
            'industries_list': 'Staffing and Recruiting, Technology',
            'specialties_list': 'Data Science Recruitment | AI/ML Placement | Tech Consulting',
            'employee_count': '1000-5000',
            'city': 'New York',
            'state': 'NY',
            'country': 'USA',
            'required_skills': 'Python | Data Science | Machine Learning | Deep Learning | NLP',
            'posted_job_titles': 'Data Scientist | AI Engineer | Research Scientist',
            'experience_levels': 'Mid-Senior level',
            'work_types': 'Full-time | Contract',
            'text': 'Recruiting firm specializing in data science and AI talent placement...'
        },
        {
            'id': 8721,
            'name': 'DATAECONOMY',
            'similarity_score': 0.6849,
            'description': 'Data platform company building next-gen analytics solutions',
            'industries_list': 'Computer Software, Big Data',
            'specialties_list': 'Data Analytics | Business Intelligence | ETL | Data Warehousing',
            'employee_count': '200-500',
            'city': 'Boston',
            'state': 'MA',
            'country': 'USA',
            'required_skills': 'SQL | Python | Data Modeling | ETL | Tableau | AWS',
            'posted_job_titles': 'Data Engineer | Analytics Engineer | BI Developer',
            'experience_levels': 'Mid level | Mid-Senior level',
            'work_types': 'Full-time | Hybrid',
            'text': 'Building data infrastructure and analytics platforms...'
        },
        {
            'id': 12983,
            'name': 'Datavail',
            'similarity_score': 0.6827,
            'description': 'Database and data management services company',
            'industries_list': 'Information Technology, Database Management',
            'specialties_list': 'Database Administration | Cloud Migration | Performance Tuning',
            'employee_count': '500-1000',
            'city': 'Denver',
            'state': 'CO',
            'country': 'USA',
            'required_skills': 'SQL | Database Design | Python | Cloud Platforms | Performance Optimization',
            'posted_job_titles': 'Database Engineer | Data Platform Engineer | Cloud DBA',
            'experience_levels': 'Mid-Senior level',
            'work_types': 'Full-time | Remote',
            'text': 'Specialized in database management and cloud data solutions...'
        },
        {
            'id': 45672,
            'name': 'BitPusher',
            'similarity_score': 0.6776,
            'description': 'Software development and IT consulting firm',
            'industries_list': 'Computer Software, IT Services',
            'specialties_list': 'Custom Software Development | Cloud Solutions | DevOps',
            'employee_count': '50-200',
            'city': 'Austin',
            'state': 'TX',
            'country': 'USA',
            'required_skills': 'Python | JavaScript | AWS | Docker | Kubernetes | CI/CD',
            'posted_job_titles': 'Software Engineer | DevOps Engineer | Full Stack Developer',
            'experience_levels': 'Entry level | Mid level',
            'work_types': 'Full-time',
            'text': 'Building custom software solutions for enterprise clients...'
        },
        {
            'id': 33421,
            'name': 'Neural Dynamics',
            'similarity_score': 0.6654,
            'description': 'AI research lab focused on neural networks and deep learning',
            'industries_list': 'Research, Artificial Intelligence',
            'specialties_list': 'Deep Learning | Computer Vision | NLP | Reinforcement Learning',
            'employee_count': '100-200',
            'city': 'Seattle',
            'state': 'WA',
            'country': 'USA',
            'required_skills': 'PyTorch | TensorFlow | Deep Learning | Computer Vision | Research',
            'posted_job_titles': 'Research Scientist | ML Researcher | AI Engineer',
            'experience_levels': 'Senior level | Lead',
            'work_types': 'Full-time | Onsite',
            'text': 'Cutting-edge AI research in neural networks and applications...'
        },
        {
            'id': 28945,
            'name': 'CloudScale Analytics',
            'similarity_score': 0.6543,
            'description': 'Cloud-native data analytics platform',
            'industries_list': 'Cloud Computing, Analytics',
            'specialties_list': 'Cloud Analytics | Real-time Processing | Data Pipelines',
            'employee_count': '200-500',
            'city': 'San Jose',
            'state': 'CA',
            'country': 'USA',
            'required_skills': 'AWS | Python | Spark | Kafka | Data Engineering | Distributed Systems',
            'posted_job_titles': 'Data Engineer | Platform Engineer | Solutions Architect',
            'experience_levels': 'Mid-Senior level',
            'work_types': 'Full-time | Remote',
            'text': 'Building scalable data analytics infrastructure in the cloud...'
        },
        {
            'id': 19283,
            'name': 'DataForge Labs',
            'similarity_score': 0.6421,
            'description': 'ML operations and MLOps platform provider',
            'industries_list': 'Machine Learning, DevOps',
            'specialties_list': 'MLOps | Model Deployment | ML Infrastructure | Monitoring',
            'employee_count': '50-100',
            'city': 'Palo Alto',
            'state': 'CA',
            'country': 'USA',
            'required_skills': 'Python | Docker | Kubernetes | ML Deployment | Monitoring Tools',
            'posted_job_titles': 'MLOps Engineer | Platform Engineer | DevOps Engineer',
            'experience_levels': 'Mid level | Mid-Senior level',
            'work_types': 'Full-time | Hybrid',
            'text': 'Helping companies deploy and manage ML models at scale...'
        },
        {
            'id': 51234,
            'name': 'InsightAI',
            'similarity_score': 0.6312,
            'description': 'Business intelligence and predictive analytics company',
            'industries_list': 'Business Intelligence, Predictive Analytics',
            'specialties_list': 'Forecasting | Predictive Modeling | BI Tools | Dashboards',
            'employee_count': '100-200',
            'city': 'Chicago',
            'state': 'IL',
            'country': 'USA',
            'required_skills': 'Python | R | Tableau | PowerBI | Statistical Modeling | SQL',
            'posted_job_titles': 'Data Analyst | BI Developer | Analytics Engineer',
            'experience_levels': 'Mid level',
            'work_types': 'Full-time | Hybrid',
            'text': 'Providing predictive analytics and BI solutions for enterprises...'
        },
        {
            'id': 67821,
            'name': 'QuantumLeap Technologies',
            'similarity_score': 0.6198,
            'description': 'Quantum computing and advanced algorithms research',
            'industries_list': 'Quantum Computing, Research',
            'specialties_list': 'Quantum Algorithms | High-Performance Computing | Cryptography',
            'employee_count': '50-100',
            'city': 'Cambridge',
            'state': 'MA',
            'country': 'USA',
            'required_skills': 'Python | Quantum Computing | Linear Algebra | Algorithms | Research',
            'posted_job_titles': 'Quantum Research Scientist | Algorithm Engineer | Research Engineer',
            'experience_levels': 'Senior level | PhD level',
            'work_types': 'Full-time | Onsite',
            'text': 'Pioneering quantum computing applications and algorithms...'
        }
    ]
    
    # Return as list of tuples
    matches = [
        (comp['id'], comp['similarity_score'], comp)
        for comp in companies[:top_k]
    ]
    
    return matches


def get_network_graph_data(candidate_id: int = 0, top_k: int = 10) -> Dict[str, Any]:
    """
    Generate network graph data for visualization.
    
    Args:
        candidate_id: Candidate identifier
        top_k: Number of companies to include
    
    Returns:
        Dictionary with nodes and edges for network graph
    """
    
    candidate = get_candidate_data(candidate_id)
    matches = get_company_matches(candidate_id, top_k)
    
    # Create nodes
    nodes = []
    
    # Add candidate node
    nodes.append({
        'id': f'C{candidate_id}',
        'label': f"Candidate #{candidate_id}",
        'title': candidate['name'],
        'color': '#00FF00',  # Green
        'shape': 'dot',
        'size': 25
    })
    
    # Add company nodes
    for comp_id, score, comp_data in matches:
        nodes.append({
            'id': f'J{comp_id}',
            'label': comp_data['name'][:20],  # Truncate long names
            'title': f"{comp_data['name']}\nScore: {score:.4f}",
            'color': '#FF0000',  # Red
            'shape': 'square',
            'size': 15 + (score * 20)  # Size based on score
        })
    
    # Create edges (connections)
    edges = []
    
    for comp_id, score, comp_data in matches:
        edges.append({
            'from': f'C{candidate_id}',
            'to': f'J{comp_id}',
            'value': score,  # Line thickness
            'title': f'Match Score: {score:.4f}',
            'color': {'opacity': score}  # Transparency based on score
        })
    
    return {
        'nodes': nodes,
        'edges': edges
    }


# For testing
if __name__ == "__main__":
    # Test functions
    candidate = get_candidate_data(0)
    print(f"✅ Candidate: {candidate['name']}")
    
    matches = get_company_matches(0, 5)
    print(f"✅ Top 5 matches loaded")
    
    graph_data = get_network_graph_data(0, 5)
    print(f"✅ Graph data: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
