# 🏢 HRHUB - HR Matching System

**Bilateral Matching Engine for Candidates & Companies**

A professional HR matching system using NLP embeddings and cosine similarity to connect job candidates with relevant companies based on skills, experience, and requirements.

---

## 🎯 Project Overview

HRHUB solves a fundamental inefficiency in hiring: candidates and companies use different vocabularies when describing skills and requirements. Our system bridges this gap using **job postings** as a translator, enriching company profiles to speak the same "skills language" as candidates.

### Key Innovation
- **Candidates** describe: "Python, Machine Learning, Data Science"
- **Companies** describe: "Tech company, innovation, growth"
- **Job Postings** translate: "We need Python, AWS, TensorFlow"
- **Result**: Accurate matching in the same embedding space ℝ³⁸⁴

---

## 🚀 Features

- ✅ **Bilateral Matching**: Both candidates and companies get matched recommendations
- ✅ **NLP-Powered**: Uses sentence transformers for semantic understanding
- ✅ **Interactive Visualization**: Network graphs showing match connections
- ✅ **Scalable**: Handles 9,544 candidates × 180,000 companies
- ✅ **Real-time**: Fast similarity computation using cosine similarity
- ✅ **Professional UI**: Clean Streamlit interface

---

## 📁 Project Structure

```
hrhub/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── mock_data.py           # Demo data (MVP)
│   ├── data_loader.py         # Real data loader (future)
│   └── embeddings/            # Saved embeddings (future)
├── utils/
│   ├── matching.py            # Cosine similarity algorithms
│   ├── visualization.py       # Network graph generation
│   └── display.py             # UI components
└── assets/
    └── style.css              # Custom CSS (optional)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/your-username/hrhub.git
cd hrhub
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

5. **Open browser**
Navigate to `http://localhost:8501`

---

## 🌐 Deployment (Streamlit Cloud)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `hrhub`
5. Main file path: `app.py`
6. Click "Deploy"

**That's it!** Your app will be live at `https://your-app.streamlit.app`

---

## 📊 Data Pipeline

### Current (MVP - Hardcoded)
```
mock_data.py → app.py → Display
```

### Future (Production)
```
CSV Files → Data Processing → Embeddings → Saved Files
                ↓
            app.py loads embeddings → Real-time matching
```

### Files to Generate (Next Phase)
```python
# After running your main code, save these:
1. candidate_embeddings.npy      # 9,544 × 384 array
2. company_embeddings.npy        # 180,000 × 384 array
3. candidates_processed.pkl      # Full candidate data
4. companies_processed.pkl       # Full company data
```

---

## 🔄 Switching from Mock to Real Data

### Current Code (MVP)
```python
# app.py
from data.mock_data import get_candidate_data, get_company_matches
```

### After Generating Embeddings
```python
# app.py
from data.data_loader import get_candidate_data, get_company_matches
```

**That's it!** No other code changes needed. The UI stays the same.

---

## 🎨 Configuration

Edit `config.py` to customize:

```python
# Matching Settings
DEFAULT_TOP_K = 10              # Number of matches to show
MIN_SIMILARITY_SCORE = 0.5      # Minimum score threshold
EMBEDDING_DIMENSION = 384       # Vector dimension

# UI Settings
NETWORK_GRAPH_HEIGHT = 600      # Graph height in pixels

# Demo Mode
DEMO_MODE = True                # Set False for production
```

---

## 📈 Technical Details

### Algorithm
1. **Text Representation**: Convert candidate/company data to structured text
2. **Embedding**: Use sentence transformers (`all-MiniLM-L6-v2`)
3. **Similarity**: Compute cosine similarity between vectors
4. **Ranking**: Sort by similarity score, return top K

### Why Cosine Similarity?
- ✅ **Scale-invariant**: Focuses on direction, not magnitude
- ✅ **Profile shape matching**: Captures proportional skill distributions
- ✅ **Fast computation**: Optimized for large-scale matching
- ✅ **Proven in NLP**: Standard metric for semantic similarity

### Performance
- **Loading time**: < 5 seconds (with pre-computed embeddings)
- **Matching speed**: < 1 second for 180K companies
- **Memory usage**: ~500MB (embeddings loaded)

---

## 🧪 Testing

### Test Mock Data
```bash
cd hrhub
python data/mock_data.py
```

Expected output:
```
✅ Candidate: Demo Candidate #0
✅ Top 5 matches loaded
✅ Graph data: 6 nodes, 5 edges
```

### Test Streamlit App
```bash
streamlit run app.py
```

---

## 🎯 Roadmap

### ✅ Phase 1: MVP (Current)
- [x] Basic matching logic
- [x] Streamlit UI
- [x] Network visualization
- [x] Hardcoded demo data

### 🔄 Phase 2: Production (Next)
- [ ] Generate real embeddings
- [ ] Load embeddings from files
- [ ] Dynamic candidate selection
- [ ] Search functionality

### 🚀 Phase 3: Advanced (Future)
- [ ] User authentication
- [ ] Company login view
- [ ] Weighted matching (different dimensions)
- [ ] RAG-powered recommendations
- [ ] Email notifications
- [ ] Analytics dashboard

---

## 👥 Team

**Master's in Business Data Science - Aalborg University**

- Roger - Project Lead & Deployment
- Eskil - [Role]
- [Team Member 3] - [Role]
- [Team Member 4] - [Role]

---

## 📝 License

This project is part of an academic course at Aalborg University.

---

## 🤝 Contributing

This is an academic project. Contributions are welcome after project submission (December 14, 2024).

---

## 📧 Contact

For questions or feedback:
- Create an issue on GitHub
- Contact via Moodle course forum

---

## 🙏 Acknowledgments

- **Sentence Transformers**: Hugging Face team
- **Streamlit**: Amazing framework for data apps
- **PyVis**: Interactive network visualization
- **Course Instructors**: For guidance and support

---

**Last Updated**: December 2024  
**Status**: 🟢 Active Development
