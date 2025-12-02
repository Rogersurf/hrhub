# 🚀 HRHUB SETUP GUIDE

**Quick Start Guide for Deployment**

---

## 📦 What You Have

A complete, production-ready Streamlit application with:
- ✅ Professional code structure
- ✅ Mock data for MVP demo
- ✅ Interactive UI with network graphs
- ✅ Ready for GitHub + Streamlit Cloud deployment

---

## ⚡ OPTION 1: Quick Local Test (2 minutes)

### For Mac/Linux:
```bash
cd hrhub
./run.sh
```

### For Windows:
```bash
cd hrhub
run.bat
```

**That's it!** Open `http://localhost:8501` in your browser.

---

## 🌐 OPTION 2: Deploy to Internet (10 minutes)

### Step 1: Install Git (if not already)
- **Windows**: Download from https://git-scm.com/
- **Mac**: Install Xcode Command Line Tools
- **Linux**: `sudo apt install git`

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `hrhub`
3. Keep it PUBLIC
4. Don't initialize with README (we have one)
5. Click "Create repository"

### Step 3: Push Your Code

Open terminal/command prompt in the `hrhub` folder:

```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial HRHUB MVP deployment"

# Connect to GitHub (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/hrhub.git

# Push
git branch -M main
git push -u origin main
```

### Step 4: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "Sign in" → Sign in with GitHub
3. Click "New app"
4. Fill in:
   - **Repository**: `YOUR-USERNAME/hrhub`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy!"

**Wait 2-3 minutes** and your app will be live! 🎉

You'll get a URL like: `https://hrhub-YOUR-USERNAME.streamlit.app`

---

## 🎯 Testing Your Deployment

### What You Should See:

1. **Header**: "🏢 HRHUB - HR Matching System"
2. **Demo Mode Banner**: Blue info box saying mock data is active
3. **Statistics**: 4 metric cards showing:
   - Total Matches: 10
   - Average Score: ~65%
   - Excellent Matches: 4
   - Best Match: ~70%

4. **Two Columns**:
   - **Left**: Candidate profile with expandable sections
   - **Right**: Company matches (table or cards)

5. **Network Graph**: Interactive visualization at the bottom

### Interaction Tests:

- ✅ Change slider in sidebar (matches 5-20)
- ✅ Change min score slider
- ✅ Switch view modes (Overview/Cards/Table)
- ✅ Expand candidate sections
- ✅ Hover over network graph nodes
- ✅ Drag nodes in the graph

---

## 🔧 Common Issues & Solutions

### Issue 1: "streamlit: command not found"

**Solution:**
```bash
pip install streamlit
```

### Issue 2: "Module not found"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue 3: Port 8501 already in use

**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### Issue 4: Git push fails (authentication)

**Solution:**
1. Generate GitHub Personal Access Token:
   - Settings → Developer settings → Personal access tokens → Generate new token
   - Select "repo" scope
   - Copy the token
2. When prompted for password, paste the token (not your GitHub password)

### Issue 5: Streamlit Cloud deployment fails

**Solution:**
- Check `requirements.txt` has all dependencies
- Ensure `app.py` is in root directory
- Check logs in Streamlit Cloud dashboard
- Make sure repository is PUBLIC

---

## 📝 Next Steps (After Demo Works)

### Phase 1: Generate Real Embeddings

1. Run your original code with save functionality:
```python
import numpy as np
import pickle

# After generating embeddings...
np.save('candidate_embeddings.npy', candidate_embeddings)
np.save('company_embeddings.npy', company_embeddings)

with open('candidates_processed.pkl', 'wb') as f:
    pickle.dump(candidates, f)
    
with open('companies_processed.pkl', 'wb') as f:
    pickle.dump(companies_full, f)
```

2. Place files in `hrhub/data/` folder

### Phase 2: Create Real Data Loader

Create `data/data_loader.py`:
```python
import numpy as np
import pickle
from utils.matching import find_top_matches

def load_embeddings():
    """Load pre-computed embeddings."""
    candidate_emb = np.load('data/candidate_embeddings.npy')
    company_emb = np.load('data/company_embeddings.npy')
    
    with open('data/candidates_processed.pkl', 'rb') as f:
        candidates = pickle.load(f)
    
    with open('data/companies_processed.pkl', 'rb') as f:
        companies = pickle.load(f)
    
    return candidate_emb, company_emb, candidates, companies

# Add functions matching mock_data.py structure
```

### Phase 3: Swap Data Sources

In `app.py`, change:
```python
# FROM:
from data.mock_data import get_candidate_data, get_company_matches

# TO:
from data.data_loader import get_candidate_data, get_company_matches
```

In `config.py`, change:
```python
DEMO_MODE = False  # Turn off demo banner
```

**That's it!** The UI stays exactly the same.

---

## 🎓 For Your Teachers Demo

### What to Show:

1. **Start the app**: Show the clean UI loading
2. **Explain the candidate**: "This represents a real data scientist profile"
3. **Point out match scores**: "70% means strong alignment"
4. **Show the graph**: "Green = candidate, Red = companies, thickness = match strength"
5. **Demonstrate interaction**: Drag nodes, zoom, hover
6. **Highlight the concept**: "No hardcoded rules - pure semantic similarity"

### Key Points to Emphasize:

- ✅ **Scalable**: Works for 9.5K × 180K matching
- ✅ **Fast**: Real-time similarity computation
- ✅ **Bilateral**: Can work both directions
- ✅ **No manual rules**: NLP understands semantics
- ✅ **Production-ready**: Clean code, modular design

---

## 📊 Project Structure Explained

```
hrhub/
├── app.py              # Main app - teachers see this running
├── config.py           # Easy to tweak settings
├── requirements.txt    # All dependencies listed
│
├── data/
│   └── mock_data.py   # Demo data (swap later)
│
├── utils/
│   ├── matching.py    # Core algorithm - your innovation
│   ├── visualization.py  # Network graphs
│   └── display.py     # UI components
│
└── README.md          # Documentation
```

**Why this structure?**
- **Modular**: Easy to swap mock → real data
- **Professional**: Industry-standard layout
- **Maintainable**: Clear separation of concerns
- **Scalable**: Ready to add features

---

## 🎯 Timeline Suggestion

**Tuesday (Today):**
- ✅ Test locally: `./run.sh`
- ✅ Deploy to GitHub
- ✅ Deploy to Streamlit Cloud
- ✅ Share link with team

**Wednesday:**
- Run your original code
- Generate & save embeddings
- Test loading saved files

**Thursday:**
- Create `data_loader.py`
- Swap to real data
- Test end-to-end
- Fix any bugs

**Friday:**
- ✅ **DEMO READY**
- Polish presentation
- Prepare talking points

**Weekend:**
- Focus 100% on report
- App already deployed!

---

## 🆘 Need Help?

### Quick Checks:

1. **Is Python 3.8+ installed?** `python --version`
2. **Are dependencies installed?** `pip list | grep streamlit`
3. **Is the file structure correct?** `ls -la`
4. **Are you in the right directory?** `pwd`

### Still Stuck?

Check these in order:
1. Error message in terminal
2. Streamlit Cloud logs (if deployed)
3. GitHub Actions (if using)
4. This guide's "Common Issues" section

---

## ✅ Deployment Checklist

Before presenting to teachers:

- [ ] Local test works: `./run.sh`
- [ ] Pushed to GitHub
- [ ] Deployed on Streamlit Cloud
- [ ] Can access via public URL
- [ ] All sections display correctly
- [ ] Graph is interactive
- [ ] No error messages
- [ ] Screenshot/video of working app
- [ ] Link shared with team
- [ ] Backup plan (run locally if cloud fails)

---

## 🎉 You're Done!

You now have:
- ✅ Professional codebase
- ✅ Working demo
- ✅ Online deployment
- ✅ Easy path to production

**The hard part is done. Now focus on your report!** 📝

---

**Good luck with your presentation!** 🚀

*Questions? Check README.md for more details.*
