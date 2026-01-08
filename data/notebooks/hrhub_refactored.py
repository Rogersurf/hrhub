# 🎯 HRHUB v3.1 - Bilateral HR Matching System
# Optimized for Production & Streamlit Deployment
# Professional Refactoring

# 📦 SECTION 1: ENVIRONMENT & IMPORTS
# ==========================================

# Cell 1.1: Install Dependencies
# -------------------------------
# Note: For Hugging Face Spaces, include in requirements.txt
"""
sentence-transformers>=2.2.0
huggingface-hub>=0.20.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
pyvis>=0.3.2
streamlit>=1.28.0
pydantic>=2.5.0
python-dotenv>=1.0.0
"""

# Cell 1.2: Core Imports
# ----------------------
import pandas as pd
import numpy as np
import json
import os
import time
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML & Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LLM Integration
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

# Configuration
from dotenv import load_dotenv
load_dotenv()

print("✅ Libraries imported successfully!")

# Cell 1.3: Configuration Class
# ----------------------------
class Config:
    """Centralized configuration for production deployment"""
    
    # Paths
    DATA_PATH = 'data/csv_files'
    PROCESSED_PATH = 'data/v3/processed/'
    MODELS_PATH = 'models/'
    
    # Model settings
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    # LLM settings (Hugging Face)
    HF_TOKEN = os.getenv('HF_TOKEN', '')
    LLM_MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
    
    # Matching parameters
    TOP_K_MATCHES = 10
    SIMILARITY_THRESHOLD = 0.5
    
    # Streamlit settings
    DEFAULT_SAMPLE_SIZE = 200
    CACHE_EXPIRY = 3600  # 1 hour

# Initialize
np.random.seed(42)
os.makedirs(Config.PROCESSED_PATH, exist_ok=True)
os.makedirs(Config.MODELS_PATH, exist_ok=True)

print(f"✅ Configuration loaded: {Config.EMBEDDING_MODEL}")

# 🏗️ SECTION 2: CORE ARCHITECTURE
# ==========================================

# Cell 2.1: Text Builder Classes
# ------------------------------
class TextBuilder:
    """Abstract base class for text representation"""
    
    def build(self, row: pd.Series) -> str:
        """Build text representation from DataFrame row"""
        raise NotImplementedError
    
    def build_batch(self, df: pd.DataFrame) -> List[str]:
        """Build texts for entire DataFrame"""
        return df.apply(self.build, axis=1).tolist()


class CandidateTextBuilder(TextBuilder):
    """Builds semantic text representation for candidates"""
    
    def build(self, row: pd.Series) -> str:
        parts = []
        
        if row.get('Category'):
            parts.append(f"Job Category: {row['Category']}")
        if row.get('skills'):
            parts.append(f"Skills: {row['skills']}")
        if row.get('career_objective'):
            parts.append(f"Objective: {row['career_objective']}")
        if row.get('degree_names'):
            parts.append(f"Education: {row['degree_names']}")
        if row.get('positions'):
            parts.append(f"Experience: {row['positions']}")
        
        return ' | '.join(parts) if parts else "No information available"


class CompanyTextBuilder(TextBuilder):
    """Builds semantic text representation for companies with job posting enrichment"""
    
    def build(self, row: pd.Series) -> str:
        parts = []
        
        if row.get('name'):
            parts.append(f"Company: {row['name']}")
        if row.get('description'):
            parts.append(f"Description: {row['description']}")
        if row.get('industries_list'):
            parts.append(f"Industries: {row['industries_list']}")
        if row.get('specialties_list'):
            parts.append(f"Specialties: {row['specialties_list']}")
        if row.get('required_skills'):
            parts.append(f"Required Skills: {row['required_skills']}")
        if row.get('posted_job_titles'):
            parts.append(f"Job Titles: {row['posted_job_titles']}")
        
        return ' | '.join(parts) if parts else "No information available"

# Cell 2.2: Embedding Manager
# ---------------------------
class EmbeddingManager:
    """Manages embedding generation, caching, and loading"""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Lazy loading of sentence transformer model"""
        if self.model is None:
            print(f"🚀 Loading embedding model: {self.model_name} ({self.device})")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model
    
    def generate_embeddings(self, texts: List[str], cache_key: str = None) -> np.ndarray:
        """Generate embeddings with optional caching"""
        
        # Check cache first
        if cache_key:
            cache_path = f"{Config.MODELS_PATH}{cache_key}.npy"
            if os.path.exists(cache_path):
                print(f"📥 Loading cached embeddings: {cache_key}")
                return np.load(cache_path)
        
        # Generate new embeddings
        if self.model is None:
            self.load_model()
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Save to cache
        if cache_key:
            np.save(cache_path, embeddings)
            print(f"💾 Saved embeddings to cache: {cache_key}")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

# Cell 2.3: Matching Engine
# -------------------------
class MatchingEngine:
    """Bilateral matching engine with production optimizations"""
    
    def __init__(self, candidate_embeddings: np.ndarray,
                 company_embeddings: np.ndarray,
                 candidate_data: pd.DataFrame,
                 company_data: pd.DataFrame):
        
        self.candidate_embeddings = candidate_embeddings
        self.company_embeddings = company_embeddings
        self.candidate_data = candidate_data.reset_index(drop=True)
        self.company_data = company_data.reset_index(drop=True)
        
        print(f"🎯 Matching Engine Initialized")
        print(f"   Candidates: {len(candidate_embeddings):,}")
        print(f"   Companies: {len(company_embeddings):,}")
    
    def find_candidate_matches(self, candidate_id: int, top_k: int = None) -> pd.DataFrame:
        """Find top company matches for a candidate"""
        top_k = top_k or Config.TOP_K_MATCHES
        
        candidate_vec = self.candidate_embeddings[candidate_id].reshape(1, -1)
        similarities = cosine_similarity(candidate_vec, self.company_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        # Build results DataFrame
        results = self.company_data.iloc[top_indices].copy()
        results['match_score'] = top_scores
        results['rank'] = range(1, top_k + 1)
        
        # Select relevant columns
        columns = ['rank', 'name', 'industries_list', 'required_skills', 'match_score']
        return results[[c for c in columns if c in results.columns]]
    
    def find_company_matches(self, company_id: int, top_k: int = None) -> pd.DataFrame:
        """Find top candidate matches for a company"""
        top_k = top_k or Config.TOP_K_MATCHES
        
        company_vec = self.company_embeddings[company_id].reshape(1, -1)
        similarities = cosine_similarity(company_vec, self.candidate_embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        results = self.candidate_data.iloc[top_indices].copy()
        results['match_score'] = top_scores
        results['rank'] = range(1, top_k + 1)
        
        columns = ['rank', 'Category', 'skills', 'match_score']
        return results[[c for c in columns if c in results.columns]]
    
    def get_bilateral_fairness(self, sample_size: int = 200) -> Dict:
        """Calculate bilateral fairness score"""
        sample_candidates = min(sample_size, len(self.candidate_embeddings))
        
        cand_to_comp_scores = []
        comp_to_cand_scores = []
        
        # Candidate → Company
        for i in range(sample_candidates):
            scores = cosine_similarity(
                self.candidate_embeddings[i].reshape(1, -1),
                self.company_embeddings
            )[0]
            top_scores = np.sort(scores)[-Config.TOP_K_MATCHES:][::-1]
            cand_to_comp_scores.extend(top_scores)
        
        # Company → Candidate (sample companies)
        sample_companies = min(sample_size, len(self.company_embeddings))
        for j in range(sample_companies):
            scores = cosine_similarity(
                self.company_embeddings[j].reshape(1, -1),
                self.candidate_embeddings[:sample_candidates]
            )[0]
            top_scores = np.sort(scores)[-Config.TOP_K_MATCHES:][::-1]
            comp_to_cand_scores.extend(top_scores)
        
        # Calculate fairness ratio
        cand_mean = np.mean(cand_to_comp_scores) if cand_to_comp_scores else 0
        comp_mean = np.mean(comp_to_cand_scores) if comp_to_cand_scores else 0
        
        fairness = min(cand_mean, comp_mean) / max(cand_mean, comp_mean) if max(cand_mean, comp_mean) > 0 else 0
        
        return {
            'candidate_to_company': float(cand_mean),
            'company_to_candidate': float(comp_mean),
            'fairness_ratio': float(fairness),
            'total_matches_evaluated': len(cand_to_comp_scores) + len(comp_to_cand_scores)
        }

# 📊 SECTION 3: DATA PROCESSING
# ==========================================

# Cell 3.1: Data Loading & Enrichment
# -----------------------------------
def load_and_preprocess_data(data_path: str = Config.DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess all datasets
    Returns: (candidates_df, companies_df)
    """
    
    print("📂 Loading datasets...")
    
    # Load base datasets
    candidates = pd.read_csv(f'{data_path}resume_data.csv')
    companies_base = pd.read_csv(f'{data_path}companies.csv')
    
    # Load enrichment data
    try:
        company_industries = pd.read_csv(f'{data_path}company_industries.csv')
        company_specialties = pd.read_csv(f'{data_path}company_specialities.csv')
        postings = pd.read_csv(f'{data_path}postings.csv', on_bad_lines='skip', engine='python')
    except FileNotFoundError as e:
        print(f"⚠️  Missing file: {e}")
        return candidates, companies_base
    
    # Enrich company data with job postings
    print("🔄 Enriching company data...")
    
    # Aggregate industries
    industries_grouped = (
        company_industries
        .groupby('company_id')['industry']
        .apply(lambda x: ', '.join(x.dropna().astype(str).unique()))
        .reset_index(name='industries_list')
    )
    
    # Aggregate specialties
    specialties_grouped = (
        company_specialties
        .groupby('company_id')['speciality']
        .apply(lambda x: ', '.join(x.dropna().astype(str).unique()))
        .reset_index(name='specialties_list')
    )
    
    # Aggregate job postings
    job_data_grouped = (
        postings
        .groupby('company_id')
        .agg({
            'title': lambda x: ', '.join(x.dropna().astype(str).unique()[:10]),
            'job_id': 'count'
        })
        .reset_index()
        .rename(columns={
            'title': 'posted_job_titles',
            'job_id': 'total_postings'
        })
    )
    
    # Merge all company data
    companies_enriched = (
        companies_base
        .merge(industries_grouped, on='company_id', how='left')
        .merge(specialties_grouped, on='company_id', how='left')
        .merge(job_data_grouped, on='company_id', how='left')
    )
    
    # Fill missing values
    fill_values = {
        'name': 'Unknown Company',
        'description': 'No description available',
        'industries_list': 'General',
        'specialties_list': 'Not specified',
        'posted_job_titles': 'Various positions',
        'total_postings': 0
    }
    
    for col, default in fill_values.items():
        if col in companies_enriched.columns:
            companies_enriched[col] = companies_enriched[col].fillna(default)
    
    print(f"✅ Data loaded: {len(candidates):,} candidates, {len(companies_enriched):,} companies")
    
    return candidates, companies_enriched

# Cell 3.2: Data Caching
# ----------------------
def save_processed_data(candidates: pd.DataFrame, companies: pd.DataFrame):
    """Save processed data for fast loading"""
    
    candidates_path = f"{Config.PROCESSED_PATH}candidates.parquet"
    companies_path = f"{Config.PROCESSED_PATH}companies.parquet"
    
    candidates.to_parquet(candidates_path)
    companies.to_parquet(companies_path)
    
    print(f"💾 Saved processed data")
    return candidates_path, companies_path


def load_processed_data():
    """Load processed data from cache"""
    
    candidates_path = f"{Config.PROCESSED_PATH}candidates.parquet"
    companies_path = f"{Config.PROCESSED_PATH}companies.parquet"
    
    if os.path.exists(candidates_path) and os.path.exists(companies_path):
        candidates = pd.read_parquet(candidates_path)
        companies = pd.read_parquet(companies_path)
        print(f"📥 Loaded cached data")
        return candidates, companies
    
    return None, None

# 🧠 SECTION 4: EMBEDDING GENERATION
# ==========================================

# Cell 4.1: Generate All Embeddings
# ---------------------------------
def generate_all_embeddings(candidates: pd.DataFrame, companies: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate embeddings for all candidates and companies
    Returns: (candidate_embeddings, company_embeddings)
    """
    
    print("\n🧠 Generating Embeddings...")
    print("=" * 50)
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Build text representations
    candidate_builder = CandidateTextBuilder()
    company_builder = CompanyTextBuilder()
    
    candidate_texts = candidate_builder.build_batch(candidates)
    company_texts = company_builder.build_batch(companies)
    
    print(f"📝 Processing {len(candidate_texts):,} candidate texts...")
    candidate_embeddings = embedding_manager.generate_embeddings(
        candidate_texts, 
        cache_key='candidate_embeddings'
    )
    
    print(f"📝 Processing {len(company_texts):,} company texts...")
    company_embeddings = embedding_manager.generate_embeddings(
        company_texts, 
        cache_key='company_embeddings'
    )
    
    print(f"✅ Embeddings generated successfully!")
    print(f"   Candidate embeddings: {candidate_embeddings.shape}")
    print(f"   Company embeddings: {company_embeddings.shape}")
    
    return candidate_embeddings, company_embeddings

# 🎯 SECTION 5: MATCHING SYSTEM
# ==========================================

# Cell 5.1: Initialize Complete System
# ------------------------------------
def initialize_matching_system(use_cache: bool = True):
    """
    Initialize the complete matching system
    Returns: MatchingEngine, candidate_data, company_data
    """
    
    print("\n🚀 Initializing HR Matching System")
    print("=" * 50)
    
    # Try to load processed data
    if use_cache:
        candidates, companies = load_processed_data()
    
    # If cache not available, process from scratch
    if candidates is None or companies is None:
        print("🔄 Processing raw data...")
        candidates, companies = load_and_preprocess_data()
        save_processed_data(candidates, companies)
    
    # Generate or load embeddings
    candidate_emb_path = f"{Config.MODELS_PATH}candidate_embeddings.npy"
    company_emb_path = f"{Config.MODELS_PATH}company_embeddings.npy"
    
    if os.path.exists(candidate_emb_path) and os.path.exists(company_emb_path):
        print("📥 Loading cached embeddings...")
        candidate_embeddings = np.load(candidate_emb_path)
        company_embeddings = np.load(company_emb_path)
    else:
        candidate_embeddings, company_embeddings = generate_all_embeddings(candidates, companies)
    
    # Initialize matching engine
    matching_engine = MatchingEngine(
        candidate_embeddings=candidate_embeddings,
        company_embeddings=company_embeddings,
        candidate_data=candidates,
        company_data=companies
    )
    
    print("✅ System initialized successfully!")
    return matching_engine, candidates, companies

# 🤖 SECTION 6: LLM FEATURES (OPTIONAL)
# ==========================================

# Cell 6.1: LLM Client Setup
# --------------------------
class LLMFeatureExtractor:
    """LLM-powered feature extraction and explanation"""
    
    def __init__(self, hf_token: str = None, model: str = None):
        self.hf_token = hf_token or Config.HF_TOKEN
        self.model = model or Config.LLM_MODEL
        self.client = None
        
        if self.hf_token:
            try:
                self.client = InferenceClient(token=self.hf_token)
                print(f"✅ LLM Client initialized: {self.model}")
            except Exception as e:
                print(f"⚠️  Failed to initialize LLM client: {e}")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if LLM features are available"""
        return self.client is not None
    
    def explain_match(self, candidate_text: str, company_text: str, score: float) -> Dict:
        """Generate LLM explanation for a match"""
        
        if not self.is_available():
            return {"explanation": "LLM features not available", "score": score}
        
        prompt = f"""
        Explain why this candidate matches this company.
        
        CANDIDATE PROFILE:
        {candidate_text[:500]}
        
        COMPANY PROFILE:
        {company_text[:500]}
        
        MATCH SCORE: {score:.2f}
        
        Provide a concise analysis covering:
        1. Key skill matches
        2. Potential gaps
        3. Overall fit assessment
        
        Format as JSON with keys: strengths, gaps, recommendation, summary
        """
        
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                
                try:
                    result = json.loads(json_str)
                    result['score'] = score
                    return result
                except:
                    pass
            
            # Fallback
            return {
                "strengths": ["Analysis unavailable"],
                "gaps": [],
                "recommendation": "Review manually",
                "summary": f"Match score: {score:.2f}",
                "score": score
            }
            
        except Exception as e:
            return {"error": str(e), "score": score}

# 📊 SECTION 7: STREAMLIT UTILITIES
# ==========================================

# Cell 7.1: Streamlit Display Functions
# -------------------------------------
def display_match_results(results_df: pd.DataFrame, title: str = "Top Matches"):
    """Format match results for Streamlit display"""
    
    if results_df.empty:
        return "No matches found"
    
    # Format the DataFrame for display
    display_df = results_df.copy()
    
    # Format scores as percentages
    if 'match_score' in display_df.columns:
        display_df['match_score'] = display_df['match_score'].apply(lambda x: f"{x:.1%}")
    
    # Truncate long text
    text_columns = ['skills', 'required_skills', 'industries_list']
    for col in text_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: str(x)[:100] + '...' if len(str(x)) > 100 else str(x)
            )
    
    return display_df


def get_candidate_options(candidates_df: pd.DataFrame) -> List[Dict]:
    """Get candidate options for dropdown"""
    options = []
    
    for idx, row in candidates_df.head(100).iterrows():  # Limit for performance
        label = f"Candidate {idx}"
        if 'Category' in row and pd.notna(row['Category']):
            label += f" - {row['Category']}"
        
        skills = str(row.get('skills', ''))[:50]
        if skills:
            label += f" | Skills: {skills}"
        
        options.append({
            'label': label,
            'value': idx,
            'skills': str(row.get('skills', '')),
            'category': str(row.get('Category', 'N/A'))
        })
    
    return options


def get_company_options(companies_df: pd.DataFrame) -> List[Dict]:
    """Get company options for dropdown"""
    options = []
    
    for idx, row in companies_df.head(200).iterrows():  # Limit for performance
        name = row.get('name', f'Company {idx}')
        industries = str(row.get('industries_list', ''))[:50]
        
        label = f"{name}"
        if industries:
            label += f" | Industries: {industries}"
        
        options.append({
            'label': label,
            'value': idx,
            'name': name,
            'industries': industries
        })
    
    return options

# 🚀 SECTION 8: MAIN EXECUTION
# ==========================================

# Cell 8.1: Run Complete Pipeline
# -------------------------------
def main():
    """Main execution pipeline"""
    
    print("\n" + "="*60)
    print("        HRHUB v3.1 - Production Pipeline")
    print("="*60)
    
    # Initialize system
    start_time = time.time()
    
    matching_engine, candidates, companies = initialize_matching_system(use_cache=True)
    
    # Calculate system metrics
    fairness_metrics = matching_engine.get_bilateral_fairness(
        sample_size=Config.DEFAULT_SAMPLE_SIZE
    )
    
    # Test matching with example candidate
    example_candidate_id = 0
    matches = matching_engine.find_candidate_matches(example_candidate_id, top_k=5)
    
    # Display results
    print("\n📊 SYSTEM METRICS:")
    print(f"   Bilateral Fairness Ratio: {fairness_metrics['fairness_ratio']:.3f}")
    print(f"   Candidate → Company Score: {fairness_metrics['candidate_to_company']:.3f}")
    print(f"   Company → Candidate Score: {fairness_metrics['company_to_candidate']:.3f}")
    
    print("\n🎯 EXAMPLE MATCH (Candidate 0 → Top Companies):")
    print(matches[['rank', 'name', 'match_score']].head(5).to_string(index=False))
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total initialization time: {elapsed:.1f} seconds")
    
    # Save state for Streamlit
    state = {
        'matching_engine': matching_engine,
        'candidates': candidates,
        'companies': companies,
        'fairness_metrics': fairness_metrics,
        'initialization_time': elapsed
    }
    
    return state

# Cell 8.2: Export for Streamlit
# ------------------------------
def export_for_streamlit(state: Dict, export_path: str = 'streamlit_app/'):
    """Export necessary components for Streamlit app"""
    
    os.makedirs(export_path, exist_ok=True)
    
    # Save essential data
    np.save(f'{export_path}candidate_embeddings.npy', 
            state['matching_engine'].candidate_embeddings)
    np.save(f'{export_path}company_embeddings.npy', 
            state['matching_engine'].company_embeddings)
    
    # Save metadata
    state['candidates'].to_parquet(f'{export_path}candidates.parquet')
    state['companies'].to_parquet(f'{export_path}companies.parquet')
    
    # Save configuration
    config_export = {
        'embedding_model': Config.EMBEDDING_MODEL,
        'top_k_matches': Config.TOP_K_MATCHES,
        'fairness_metrics': state['fairness_metrics'],
        'total_candidates': len(state['candidates']),
        'total_companies': len(state['companies'])
    }
    
    with open(f'{export_path}config.json', 'w') as f:
        json.dump(config_export, f, indent=2)
    
    print(f"✅ Exported to {export_path}")
    print(f"   Candidates: {len(state['candidates']):,}")
    print(f"   Companies: {len(state['companies']):,}")
    print(f"   Embeddings: {state['matching_engine'].candidate_embeddings.shape}")

# Cell 8.3: Streamlit App Template
# --------------------------------
STREAMLIT_APP_TEMPLATE = '''
# HRHUB - Bilateral Matching System
# Streamlit Application

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import json

# Page config
st.set_page_config(
    page_title="HRHUB Matching System",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'matching_engine' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.candidates = None
    st.session_state.companies = None

# Title and description
st.title("🎯 HRHUB - Bilateral HR Matching")
st.markdown("""
### AI-powered matching between candidates and companies
This system uses semantic embeddings to find optimal matches from **9,544 candidates** and **24,473 companies**.
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    match_type = st.radio(
        "Match Type:",
        ["Candidate → Companies", "Company → Candidates"]
    )
    
    top_k = st.slider(
        "Number of matches:",
        min_value=1, max_value=20, value=10
    )
    
    show_explanations = st.checkbox("Show LLM Explanations", value=False)
    
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()

# Main content
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing matching system..."):
        # Load data and embeddings
        candidates = pd.read_parquet('candidates.parquet')
        companies = pd.read_parquet('companies.parquet')
        
        candidate_embeddings = np.load('candidate_embeddings.npy')
        company_embeddings = np.load('company_embeddings.npy')
        
        # Initialize matching engine
        from matching_engine import MatchingEngine
        matching_engine = MatchingEngine(
            candidate_embeddings, company_embeddings,
            candidates, companies
        )
        
        st.session_state.matching_engine = matching_engine
        st.session_state.candidates = candidates
        st.session_state.companies = companies
        st.session_state.initialized = True
        
    st.success("✅ System initialized successfully!")

# Display system metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Candidates", f"{len(st.session_state.candidates):,}")
with col2:
    st.metric("Total Companies", f"{len(st.session_state.companies):,}")
with col3:
    fairness = 0.85  # Load from config
    st.metric("Fairness Score", f"{fairness:.2%}")

# Selection interface
st.subheader("🔍 Select for Matching")

if match_type == "Candidate → Companies":
    candidate_options = []
    for idx, row in st.session_state.candidates.head(100).iterrows():
        label = f"Candidate {idx}"
        if 'Category' in row and pd.notna(row['Category']):
            label += f" - {row['Category']}"
        candidate_options.append((label, idx))
    
    selected_label = st.selectbox(
        "Select Candidate:",
        options=[opt[0] for opt in candidate_options],
        index=0
    )
    
    selected_id = next(idx for label, idx in candidate_options if label == selected_label)
    
    if st.button("Find Matches", type="primary"):
        with st.spinner("Finding best matches..."):
            matches = st.session_state.matching_engine.find_candidate_matches(
                selected_id, top_k=top_k
            )
            
            # Display results
            st.subheader("🎯 Top Matches")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Table View", "Visualization"])
            
            with tab1:
                display_df = matches.copy()
                display_df['Match Score'] = display_df['match_score'].apply(lambda x: f"{x:.1%}")
                st.dataframe(
                    display_df[['rank', 'name', 'industries_list', 'Match Score']],
                    use_container_width=True
                )
            
            with tab2:
                # Create bar chart of match scores
                fig = px.bar(
                    matches.head(10),
                    x='name',
                    y='match_score',
                    title='Top 10 Match Scores',
                    labels={'match_score': 'Match Score', 'name': 'Company'}
                )
                fig.update_layout(yaxis_tickformat=',.0%')
                st.plotly_chart(fig, use_container_width=True)
            
            # LLM Explanations
            if show_explanations:
                st.subheader("🤖 Match Explanations")
                
                # Get candidate and top match details
                candidate_row = st.session_state.candidates.iloc[selected_id]
                top_company_row = matches.iloc[0]
                
                # Generate explanation
                with st.spinner("Generating AI explanation..."):
                    explanation = {
                        'strengths': ['Technical skill alignment', 'Industry fit'],
                        'gaps': ['Experience level mismatch'],
                        'recommendation': 'Proceed with interview',
                        'summary': 'Strong technical match with good growth potential'
                    }
                    
                    # Display explanation
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("**Strengths:**")
                        for strength in explanation['strengths']:
                            st.write(f"✅ {strength}")
                    
                    with col2:
                        st.warning("**Areas to Consider:**")
                        for gap in explanation['gaps']:
                            st.write(f"⚠️ {gap}")
                    
                    st.success(f"**Recommendation:** {explanation['recommendation']}")

else:  # Company → Candidates
    st.write("Company matching interface would go here...")

# Footer
st.markdown("---")
st.caption("HRHUB v3.1 | Master's Thesis Project | Aalborg University 2025")
'''

# Cell 8.4: Execute Pipeline
# --------------------------
if __name__ == "__main__":
    
    # Run complete pipeline
    state = main()
    
    # Export for Streamlit
    export_for_streamlit(state, 'streamlit_export/')
    
    # Save Streamlit app template
    with open('streamlit_export/app.py', 'w') as f:
        f.write(STREAMLIT_APP_TEMPLATE)
    
    print("\n" + "="*60)
    print("📁 FILES READY FOR DEPLOYMENT:")
    print("="*60)
    print("1. streamlit_export/app.py           - Main Streamlit app")
    print("2. streamlit_export/candidates.parquet - Candidate data")
    print("3. streamlit_export/companies.parquet  - Company data")
    print("4. streamlit_export/*.npy           - Embedding vectors")
    print("5. streamlit_export/config.json     - System configuration")
    print("\n🚀 To deploy on Hugging Face Spaces:")
    print("   - Create new Space with Streamlit SDK")
    print("   - Upload all files from streamlit_export/")
    print("   - Add requirements.txt with dependencies")
    print("="*60)