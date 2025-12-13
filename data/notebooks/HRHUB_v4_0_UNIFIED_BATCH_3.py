"""
HRHUB v4.0 - BATCH 3: Matching Algorithms

This batch implements THREE matching methods for academic comparison:
1. 🔴 TF-IDF + Cosine Similarity (Traditional Baseline)
2. 🟡 Keyword Overlap (Jaccard Similarity)
3. 🟢 SBERT Semantic Embeddings (Our Method)

Educational Focus:
- Comparative analysis of approaches
- Performance trade-offs
- Why semantic understanding wins
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import time

# ============================================================================
# SECTION 3: MATCHING ALGORITHMS (COMPARATIVE ANALYSIS)
# ============================================================================

# ============================================================================
# Cell 3.1: Method 1 - TF-IDF Baseline
# ============================================================================

class TFIDFMatcher:
    """
    Traditional keyword-based matching using TF-IDF.
    
    How It Works:
    ------------
    1. TF (Term Frequency): How often a word appears in a document
    2. IDF (Inverse Document Frequency): How rare the word is across all documents
    3. TF-IDF = TF × IDF (rare words in a doc get high scores)
    4. Cosine similarity between TF-IDF vectors
    
    Mathematical Foundation:
    -----------------------
    TF-IDF(t,d) = (count(t in d) / len(d)) × log(N / df(t))
    
    where:
    - t = term (word)
    - d = document
    - N = total documents
    - df(t) = documents containing t
    
    Strengths:
    ---------
    + Fast: O(n*m) where n=vocab, m=docs
    + Explainable: Can see which keywords matched
    + Memory efficient: Sparse matrices
    + No training required
    
    Weaknesses:
    ----------
    - No semantic understanding: "Python programmer" ≠ "Python developer"
    - Vocabulary mismatch: "ML Engineer" ≠ "Machine Learning Engineer"
    - Bag-of-words: Loses word order and context
    - Poor with synonyms: "car" ≠ "automobile"
    
    Use Cases:
    ---------
    • Document retrieval
    • Spam filtering
    • When interpretability is critical
    • When semantic similarity is not needed
    
    Performance Characteristics:
    --------------------------
    Time Complexity:
    - Training: O(n*m) where n=vocab, m=docs
    - Query: O(k) where k=query length
    
    Space Complexity:
    - O(n*m) worst case, but sparse matrix compression helps
    
    Why This is Our Baseline:
    ------------------------
    TF-IDF is the industry standard for keyword matching.
    If our semantic method can't beat this, it's not worth using!
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TF-IDF matcher.
        
        Args:
            max_features: Maximum vocabulary size (prevents overfitting)
            ngram_range: (min_n, max_n) for n-grams
                        (1,1) = only unigrams
                        (1,2) = unigrams + bigrams (captures "machine learning")
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english',  # Remove "the", "a", "is", etc.
            min_df=2,  # Word must appear in at least 2 docs
            max_df=0.95  # Ignore words in >95% of docs
        )
        self.fitted = False
    
    def fit(self, texts: List[str]):
        """
        Fit vectorizer on corpus.
        
        What This Does:
        --------------
        1. Builds vocabulary from all texts
        2. Computes IDF scores for each word
        3. Creates sparse matrix representation
        
        Performance Note:
        ----------------
        For 9,544 candidates + 24,473 companies:
        - Time: ~5-10 seconds
        - Memory: ~50MB for vocabulary
        
        Args:
            texts: List of text documents
        """
        print(f"   📊 Fitting TF-IDF on {len(texts):,} documents...")
        start = time.time()
        
        self.vectorizer.fit(texts)
        self.fitted = True
        
        elapsed = time.time() - start
        vocab_size = len(self.vectorizer.vocabulary_)
        
        print(f"   ✅ Fitted in {elapsed:.2f}s")
        print(f"   📖 Vocabulary size: {vocab_size:,} terms")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Returns:
            Sparse matrix of shape (n_texts, vocab_size)
        """
        if not self.fitted:
            raise ValueError("Must call fit() first!")
        return self.vectorizer.transform(texts)
    
    def match(self, query_texts: List[str], corpus_texts: List[str], 
             top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        Find top-k matches for each query.
        
        Algorithm:
        ---------
        1. Transform query and corpus to TF-IDF vectors
        2. Compute cosine similarity matrix
        3. For each query, get top-k highest scores
        
        Time Complexity: O(Q*C) where Q=queries, C=corpus
        
        Args:
            query_texts: Texts to match
            corpus_texts: Corpus to search in
            top_k: Number of matches per query
            
        Returns:
            List of [(index, score), ...] for each query
        """
        # Transform to TF-IDF vectors
        query_vectors = self.transform(query_texts)
        corpus_vectors = self.transform(corpus_texts)
        
        # Compute similarity matrix
        similarities = cosine_similarity(query_vectors, corpus_vectors)
        
        # Get top-k for each query
        results = []
        for sim_row in similarities:
            # argsort gives indices sorted by value (ascending)
            # [-top_k:] gets last k elements
            # [::-1] reverses to descending order
            top_indices = np.argsort(sim_row)[-top_k:][::-1]
            top_scores = sim_row[top_indices]
            results.append(list(zip(top_indices, top_scores)))
        
        return results

# ============================================================================
# Cell 3.2: Method 2 - Keyword Overlap (Jaccard)
# ============================================================================

class KeywordOverlapMatcher:
    """
    Simple keyword overlap using Jaccard similarity.
    
    How It Works:
    ------------
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    Example:
    -------
    Text A: "Python Java developer"
    Text B: "Python C++ developer"
    
    Set A: {python, java, developer}
    Set B: {python, c++, developer}
    
    Intersection: {python, developer} → 2 words
    Union: {python, java, c++, developer} → 4 words
    
    Jaccard = 2/4 = 0.5
    
    Strengths:
    ---------
    + Extremely simple
    + Fast O(n) per comparison
    + Interpretable
    + Works well for exact keyword matching
    
    Weaknesses:
    ----------
    - No word importance weighting
    - Treats all words equally ("the" = "Python")
    - No semantic understanding
    - Sensitive to text length
    
    Use Cases:
    ---------
    • Quick similarity checks
    • Exact keyword matching
    • When speed is critical
    • Baseline for comparison
    
    Why This is Our Second Baseline:
    --------------------------------
    Even simpler than TF-IDF. If semantic embeddings can't beat
    simple set intersection, something is wrong!
    """
    
    @staticmethod
    def jaccard_similarity(set_a: set, set_b: set) -> float:
        """
        Compute Jaccard similarity between two sets.
        
        Args:
            set_a, set_b: Sets to compare
            
        Returns:
            float: Jaccard similarity in [0, 1]
        """
        if not set_a or not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def text_to_words(text: str) -> set:
        """Convert text to set of lowercase words."""
        # Simple tokenization (could use nltk for better results)
        words = text.lower().split()
        # Remove common stop words (optional)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return {w for w in words if w not in stop_words and len(w) > 2}
    
    def match(self, query_texts: List[str], corpus_texts: List[str],
             top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        Find top-k matches using Jaccard similarity.
        
        Performance:
        -----------
        Time: O(Q*C*W) where Q=queries, C=corpus, W=avg words
        Space: O(W) for sets
        
        For large datasets, this is slower than TF-IDF despite simpler math!
        Why? TF-IDF uses optimized sparse matrix operations.
        """
        results = []
        
        for query_text in query_texts:
            query_set = self.text_to_words(query_text)
            
            # Compute similarity with all corpus texts
            similarities = []
            for corpus_text in corpus_texts:
                corpus_set = self.text_to_words(corpus_text)
                sim = self.jaccard_similarity(query_set, corpus_set)
                similarities.append(sim)
            
            # Get top-k
            similarities = np.array(similarities)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]
            
            results.append(list(zip(top_indices, top_scores)))
        
        return results

# ============================================================================
# Cell 3.3: Method 3 - SBERT Semantic Embeddings (OUR METHOD)
# ============================================================================

class SBERTMatcher:
    """
    Semantic matching using Sentence-BERT embeddings.
    
    How It Works:
    ------------
    1. Use pre-trained BERT model fine-tuned for semantic similarity
    2. Each text → 384-dimensional vector (embedding)
    3. Similar texts have high cosine similarity in embedding space
    
    Key Innovation:
    --------------
    Unlike bag-of-words methods, SBERT understands:
    - "Python programmer" ≈ "Python developer" (synonyms)
    - "ML Engineer" ≈ "Machine Learning Engineer" (abbreviations)
    - "Data Scientist" ≈ "Data Analyst" (related roles)
    
    Mathematical Foundation:
    -----------------------
    Text → BERT → Pooling → L2 Normalization → 384-D vector
    
    Similarity = cosine(v1, v2) = (v1 · v2) / (||v1|| ||v2||)
    
    Why This Works:
    --------------
    BERT was pre-trained on billions of words, learning:
    - Word relationships (king - man + woman ≈ queen)
    - Context understanding (bank of river vs. bank account)
    - Semantic similarity (car ≈ automobile)
    
    Strengths:
    ---------
    + Semantic understanding: Handles synonyms, paraphrases
    + Context-aware: Word meaning depends on surrounding words
    + Dense representations: Every dimension is meaningful
    + Transfer learning: Pre-trained on massive datasets
    
    Weaknesses:
    ----------
    - Slower: Neural network forward pass
    - Black box: Hard to explain why texts match
    - Memory: 384 dimensions vs. sparse TF-IDF
    - Requires pre-trained model (~80MB)
    
    Performance Characteristics:
    --------------------------
    Time Complexity:
    - Embedding: O(L) per text where L=text length
    - Query: O(1) with pre-computed embeddings
    
    Space Complexity:
    - O(n*384) for embeddings
    - ~3KB per entity (384 floats × 8 bytes)
    
    Optimization Strategies:
    ----------------------
    1. Batch processing: Embed 32-128 texts at once
    2. Caching: Pre-compute and save embeddings
    3. GPU acceleration: 10x faster with CUDA
    4. Quantization: 8-bit embeddings (4x smaller, 1% accuracy loss)
    
    Why This is Our PRIMARY Method:
    ------------------------------
    Semantic understanding is ESSENTIAL for HR matching because:
    - Candidates use different terms than companies
    - Skills have many equivalent expressions
    - Context matters (Python for data vs. Python for web)
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize SBERT matcher.
        
        Model Selection Rationale:
        -------------------------
        all-MiniLM-L6-v2:
        • Balanced: Quality vs. speed
        • Size: 80MB (deployable)
        • Speed: ~10ms per text on CPU
        • Quality: 68.06 on STS benchmark
        
        Alternatives:
        • all-mpnet-base-v2: Better quality (768D), but 420MB
        • paraphrase-MiniLM-L3-v2: Faster but lower quality
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"   🧠 Loading SBERT model: {model_name}...")
        start = time.time()
        
        self.model = SentenceTransformer(model_name)
        
        elapsed = time.time() - start
        print(f"   ✅ Model loaded in {elapsed:.2f}s")
        print(f"   📏 Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed(self, texts: List[str], batch_size: int = 32, 
             show_progress: bool = True) -> np.ndarray:
        """
        Embed texts to semantic vectors.
        
        Optimization Notes:
        ------------------
        1. Batch size 32: Good balance for CPU
           - Too small: Underutilizes model
           - Too large: Memory issues
        
        2. show_progress_bar: Useful for large datasets
        
        3. convert_to_numpy: Returns numpy array (default is list)
        
        Performance:
        -----------
        For 9,544 candidates:
        - Time: ~30 seconds on CPU
        - Time: ~3 seconds on GPU
        - Memory: 9,544 × 384 × 4 bytes = 14MB
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (n_texts, 384)
        """
        print(f"   🔄 Embedding {len(texts):,} texts...")
        start = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        elapsed = time.time() - start
        print(f"   ✅ Embedded in {elapsed:.2f}s ({len(texts)/elapsed:.0f} texts/sec)")
        print(f"   📦 Shape: {embeddings.shape}")
        
        return embeddings
    
    def match(self, query_embeddings: np.ndarray, corpus_embeddings: np.ndarray,
             top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        Find top-k matches using cosine similarity.
        
        Optimization:
        ------------
        Pre-computed embeddings make this EXTREMELY fast:
        - Matrix multiplication: O(Q*C*D) but highly optimized
        - For 10K x 20K comparison: ~100ms
        
        Why So Fast?
        -----------
        1. NumPy/BLAS optimizations
        2. L2-normalized vectors: cosine = dot product
        3. Batch matrix operations
        
        Args:
            query_embeddings: Shape (n_queries, 384)
            corpus_embeddings: Shape (n_corpus, 384)
            top_k: Number of matches per query
            
        Returns:
            List of [(index, score), ...] for each query
        """
        # Compute similarity matrix
        # Since vectors are L2-normalized, cosine = dot product
        similarities = query_embeddings @ corpus_embeddings.T
        
        # Get top-k for each query
        results = []
        for sim_row in similarities:
            top_indices = np.argsort(sim_row)[-top_k:][::-1]
            top_scores = sim_row[top_indices]
            results.append(list(zip(top_indices, top_scores)))
        
        return results

# ============================================================================
# Cell 3.4: Comparative Analysis
# ============================================================================

def compare_methods(query_texts: List[str], corpus_texts: List[str],
                   top_k: int = 10) -> Dict[str, any]:
    """
    Compare all three matching methods.
    
    This function runs all three methods and collects:
    - Match quality (similarity scores)
    - Performance (time taken)
    - Top matches from each method
    
    Educational Purpose:
    -------------------
    Shows empirically that semantic embeddings outperform
    traditional keyword-based methods for HR matching.
    
    Returns:
        Dict with results from each method
    """
    results = {}
    
    print("\n" + "="*80)
    print("🔬 COMPARATIVE ANALYSIS: THREE MATCHING METHODS")
    print("="*80)
    
    # ========================================================================
    # Method 1: TF-IDF
    # ========================================================================
    print("\n🔴 METHOD 1: TF-IDF + Cosine Similarity")
    print("-" * 80)
    start = time.time()
    
    tfidf_matcher = TFIDFMatcher()
    all_texts = query_texts + corpus_texts
    tfidf_matcher.fit(all_texts)
    tfidf_results = tfidf_matcher.match(query_texts, corpus_texts, top_k)
    
    tfidf_time = time.time() - start
    tfidf_avg_score = np.mean([score for matches in tfidf_results 
                               for _, score in matches])
    
    results['tfidf'] = {
        'matches': tfidf_results,
        'time': tfidf_time,
        'avg_score': tfidf_avg_score
    }
    
    print(f"   ⏱️  Time: {tfidf_time:.2f}s")
    print(f"   📊 Avg similarity: {tfidf_avg_score:.4f}")
    
    # ========================================================================
    # Method 2: Keyword Overlap
    # ========================================================================
    print("\n🟡 METHOD 2: Keyword Overlap (Jaccard)")
    print("-" * 80)
    start = time.time()
    
    jaccard_matcher = KeywordOverlapMatcher()
    jaccard_results = jaccard_matcher.match(query_texts, corpus_texts, top_k)
    
    jaccard_time = time.time() - start
    jaccard_avg_score = np.mean([score for matches in jaccard_results 
                                 for _, score in matches])
    
    results['jaccard'] = {
        'matches': jaccard_results,
        'time': jaccard_time,
        'avg_score': jaccard_avg_score
    }
    
    print(f"   ⏱️  Time: {jaccard_time:.2f}s")
    print(f"   📊 Avg similarity: {jaccard_avg_score:.4f}")
    
    # ========================================================================
    # Method 3: SBERT
    # ========================================================================
    print("\n🟢 METHOD 3: SBERT Semantic Embeddings")
    print("-" * 80)
    start = time.time()
    
    sbert_matcher = SBERTMatcher()
    query_embeddings = sbert_matcher.embed(query_texts, show_progress=False)
    corpus_embeddings = sbert_matcher.embed(corpus_texts, show_progress=False)
    sbert_results = sbert_matcher.match(query_embeddings, corpus_embeddings, top_k)
    
    sbert_time = time.time() - start
    sbert_avg_score = np.mean([score for matches in sbert_results 
                               for _, score in matches])
    
    results['sbert'] = {
        'matches': sbert_results,
        'time': sbert_time,
        'avg_score': sbert_avg_score
    }
    
    print(f"   ⏱️  Time: {sbert_time:.2f}s")
    print(f"   📊 Avg similarity: {sbert_avg_score:.4f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("📊 SUMMARY COMPARISON")
    print("="*80)
    print(f"\n{'Method':<25} {'Time (s)':<12} {'Avg Score':<12} {'Speedup':<10}")
    print("-" * 80)
    
    base_time = tfidf_time
    for method_name, method_results in results.items():
        speedup = base_time / method_results['time']
        print(f"{method_name:<25} "
              f"{method_results['time']:>10.2f}s  "
              f"{method_results['avg_score']:>10.4f}  "
              f"{speedup:>8.2f}x")
    
    print("="*80)
    
    return results

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Sample test data
    queries = [
        "Python developer with machine learning experience",
        "Senior data scientist seeking AI role"
    ]
    
    corpus = [
        "Company needs Python programmer for ML projects",
        "Hiring data analyst with Python and SQL",
        "AI engineer position open for senior developer",
        "Backend developer role, Java and Python required"
    ]
    
    # Run comparison
    results = compare_methods(queries, corpus, top_k=3)
    
    print("\n✅ Comparative analysis complete!")

