# Part 3: Dense Retrieval, Embeddings, and Vector Databases

> **Series:** Retrieval-Augmented Generation — From Foundations to Production  
> **Part:** 3 of 6  
> **Level:** Intermediate  
> **Prerequisites:** Parts 1 & 2; comfort with Python lists, functions, and NumPy arrays  
> **Key Papers:** Lewis et al. (2020), Karpukhin et al. (2020), Robertson & Zaragoza (2009)

---

> **Prerequisites Check ✓**  
> Before starting this part, you should be able to:  
> - Describe the three-step RAG loop (Retrieve → Augment → Generate)  
> - Implement a basic keyword-overlap retriever in Python  
> - Construct a system + user message for an LLM API call  
> - Explain why keyword retrieval fails on semantically similar but lexically different queries  
>
> If any of these feel shaky, revisit Parts 1 and 2 before continuing.

---

## Table of Contents

1. [Introduction: Why Keyword Retrieval Is Not Enough](#1-introduction-why-keyword-retrieval-is-not-enough)
2. [Classical Information Retrieval: TF-IDF and BM25](#2-classical-information-retrieval-tf-idf-and-bm25)
3. [The Semantic Gap: Moving from Tokens to Meaning](#3-the-semantic-gap-moving-from-tokens-to-meaning)
4. [Word Embeddings: A Brief History](#4-word-embeddings-a-brief-history)
5. [Sentence and Passage Embeddings](#5-sentence-and-passage-embeddings)
6. [Dense Retrieval: The DPR Framework](#6-dense-retrieval-the-dpr-framework)
7. [Similarity Metrics in Embedding Space](#7-similarity-metrics-in-embedding-space)
8. [Vector Databases: Architecture and Indexing](#8-vector-databases-architecture-and-indexing)
9. [Hybrid Retrieval: Combining Sparse and Dense](#9-hybrid-retrieval-combining-sparse-and-dense)
10. [Implementing Dense Retrieval in Python](#10-implementing-dense-retrieval-in-python)
11. [Chunking: Preparing Documents for Retrieval](#11-chunking-preparing-documents-for-retrieval)
12. [Evaluating Retrieval Quality](#12-evaluating-retrieval-quality)
13. [Key Concepts Summary](#13-key-concepts-summary)
14. [Further Reading and Foundational Papers](#14-further-reading-and-foundational-papers)
15. [Review Questions](#15-review-questions)

---

## 1. Introduction: Why Keyword Retrieval Is Not Enough

In Part 2, we built a functional RAG pipeline with a **keyword-overlap retriever** — a system that scores documents by counting how many words from the query appear in each document. Despite its simplicity, this retriever correctly illustrated the core RAG loop and works adequately for toy corpora where queries and documents share vocabulary.

Production RAG systems, however, encounter a fundamental challenge that keyword retrieval cannot overcome: **the semantic gap**.

Consider a user who asks: *"How do I reduce memory usage in my neural network?"*

A keyword retriever will score highly any document containing the words "reduce," "memory," "usage," "neural," or "network." But the most relevant document in the knowledge base might contain none of those words — instead discussing "techniques for lowering GPU RAM consumption in deep learning models." Every word is different; the meaning is identical.

This mismatch between surface form (words) and underlying meaning (semantics) is not a corner case — it is the normal condition of natural language. Humans naturally use synonyms, paraphrases, domain-specific jargon, and varied grammatical structures to express the same ideas. A retriever that cannot handle this variability will fail systematically on real user queries.

This part addresses the semantic gap head-on. We begin with classical IR methods (TF-IDF and BM25) to establish rigorous baselines, then build up to the modern paradigm: **dense retrieval with neural embeddings**, where text is encoded as high-dimensional vectors that capture semantic meaning — enabling retrieval by conceptual similarity rather than keyword overlap.

---

## 2. Classical Information Retrieval: TF-IDF and BM25

Before neural embeddings existed, information retrieval was dominated by **sparse lexical methods** — algorithms that represent documents and queries as sparse vectors over a vocabulary of all known words. These methods remain important: they are fast, interpretable, require no training data, and form the "sparse" half of modern hybrid retrieval systems.

### 2.1 TF-IDF: Term Frequency — Inverse Document Frequency

TF-IDF (Salton & Buckley, 1988) scores each word in a document according to two complementary signals:

**Term Frequency (TF):** How often does this word appear in this document?

```
TF(t, d) = count(t in d) / total_words(d)
```

A word that appears many times in a document is likely important to that document's topic.

**Inverse Document Frequency (IDF):** How rare is this word across the entire corpus?

```
IDF(t, D) = log( N / df(t) )
```

where `N` is the total number of documents and `df(t)` is the number of documents containing term `t`. Common words like "the," "and," "is" appear in almost every document, so their IDF is near zero — they provide no discriminating signal. Rare, specific terms like "epigenomics" or "transformer" appear in few documents and receive high IDF, indicating they are informative.

**TF-IDF score:**
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

To score a document against a query, TF-IDF computes the **dot product** of the document's TF-IDF vector and the query's TF-IDF vector — equivalent to summing the TF-IDF scores of all query terms that appear in the document.

**Limitations of TF-IDF:**
- Treats each document's length as irrelevant (longer documents accumulate higher raw term counts);
- Does not account for term saturation (the 100th occurrence of a term is treated equally to the 1st);
- No understanding of word meaning — purely statistical.

### 2.2 BM25: Best Match 25

BM25 (Robertson & Zaragoza, 2009) — sometimes called Okapi BM25 — is a sophisticated refinement of TF-IDF that has been the dominant sparse retrieval algorithm for over two decades. It is the default retriever in systems like Elasticsearch, OpenSearch, and Lucene, meaning it underlies virtually every production search system in use today.

BM25 introduces two improvements over raw TF-IDF:

**Term Frequency Saturation (parameter `k1`):**

BM25 applies a saturation function to TF:

```
TF_BM25(t, d) = TF(t, d) × (k1 + 1) / (TF(t, d) + k1)
```

This function is bounded: as TF grows very large, TF_BM25 approaches `(k1 + 1)`. The parameter `k1` (typically 1.2–2.0) controls how quickly saturation occurs. This prevents very long documents from dominating solely due to raw repetition.

**Document Length Normalization (parameter `b`):**

```
TF_BM25_norm(t, d) = TF(t, d) × (k1 + 1) / (TF(t, d) + k1 × (1 - b + b × |d| / avgdl))
```

where `|d|` is the document length and `avgdl` is the average document length across the corpus. The parameter `b` (typically 0.75) controls the degree of length normalization. A longer document is "penalized" — its term frequency scores are reduced relative to what a shorter document with the same raw count would receive.

**The complete BM25 scoring function:**

```
BM25(q, d) = Σ_{t ∈ q} IDF(t) × TF_BM25_norm(t, d)
```

where the sum is over all query terms `t`.

BM25's combination of IDF weighting, TF saturation, and length normalization makes it remarkably robust across a wide variety of retrieval tasks — to the point that it frequently outperforms poorly tuned neural retrievers and serves as a critical baseline in RAG evaluation.

### 2.3 Why BM25 Is Still Relevant in the Age of Neural Retrieval

Despite neural methods outperforming BM25 on most benchmarks (in aggregate), BM25 retains critical advantages:

| Property | BM25 | Dense Retrieval |
|---|---|---|
| Training required | ✗ None | ✓ Large labeled dataset or contrastive pretraining |
| Latency | Very fast (inverted index) | Moderate (ANN search) |
| Interpretability | High (which terms matched) | Low (black box) |
| Exact keyword matching | ✓ Exact | ✗ Approximate |
| Semantic generalization | ✗ None | ✓ Strong |
| Out-of-vocabulary terms | Handles well | May struggle |
| Domain adaptation | Immediate | Requires re-training or fine-tuning |

For enterprise RAG systems where exact keyword matching matters (e.g., finding documents that mention a specific regulation number, product SKU, or proper noun), BM25 remains essential. This motivates the hybrid approach discussed in §9.

---

## 3. The Semantic Gap: Moving from Tokens to Meaning

The core limitation of all sparse retrieval methods is that they operate on **tokens** (words) rather than **meaning**. This creates the semantic gap: two texts can be semantically equivalent but share no vocabulary.

Classic examples of the semantic gap in retrieval:

| User Query | Relevant Document Contains | Keyword Overlap |
|---|---|---|
| "How to fix memory leak?" | "Resolving RAM consumption issues in Python" | ~0% |
| "What causes inflation?" | "Why prices rise: monetary expansion and demand pressure" | ~0% |
| "Neural network regularization" | "Preventing overfitting in deep learning models" | ~30% |
| "COVID vaccine side effects" | "Adverse reactions to coronavirus immunization" | ~0% |

In each case, a human reader immediately recognizes the relevance. A keyword retriever returns a relevance score near zero.

The solution is to represent text not as a bag of tokens but as a point in a continuous, high-dimensional **semantic space** — a **vector space** where similar meanings are geometrically nearby. This is the fundamental insight behind embeddings.

---

## 4. Word Embeddings: A Brief History

The history of word embeddings is one of the most productive research trajectories in NLP, spanning from simple co-occurrence statistics to billion-parameter neural models.

### 4.1 Distributional Semantics: The Foundation

The theoretical foundation of embeddings is the **distributional hypothesis** (Harris, 1954; Firth, 1957):

> *"A word is characterized by the company it keeps."*

Words that appear in similar contexts tend to have similar meanings. "Dog" and "cat" both appear before "barked" / "meowed," near "pet" and "owner," and in sentences about animals. Their meaning is captured by the statistical patterns of their co-occurrence with other words across a large corpus.

### 4.2 Word2Vec: Dense Vectors from Neural Prediction

Word2Vec (Mikolov et al., 2013) operationalized the distributional hypothesis as a neural prediction task. The model learns dense (low-dimensional, non-sparse) vector representations — **word embeddings** — by training on one of two objectives:

**Skip-Gram:** Given a word, predict its surrounding context words.  
**CBOW (Continuous Bag of Words):** Given the surrounding context, predict the central word.

Both objectives force the model to encode semantic relationships in its weight matrix. The resulting embeddings exhibit remarkable geometric properties:

```
king - man + woman ≈ queen
Paris - France + Germany ≈ Berlin
```

These arithmetic relationships emerge not from explicit programming but from the distributional patterns in the training corpus. Word2Vec embeddings are typically 100–300 dimensional.

### 4.3 GloVe and FastText

**GloVe** (Pennington et al., 2014) approached the same problem from the perspective of global co-occurrence statistics rather than local prediction tasks. GloVe's loss function directly factorizes the word co-occurrence matrix, often achieving better performance on word analogy tasks.

**FastText** (Bojanowski et al., 2017) extended Word2Vec by representing words as bags of character n-grams. This enables FastText to generate embeddings for **out-of-vocabulary words** — a critical capability for handling domain-specific terminology, typos, and morphologically rich languages.

### 4.4 Contextual Embeddings: ELMo and BERT

A fundamental limitation of Word2Vec, GloVe, and FastText is that each word receives a **single, fixed embedding** regardless of context. The word "bank" has the same vector whether it appears as "river bank" or "financial bank."

**ELMo** (Peters et al., 2018 — Embeddings from Language Models) introduced **contextualized embeddings**: word representations that change based on the surrounding sentence. ELMo used bidirectional LSTMs to produce context-sensitive representations.

**BERT** (Devlin et al., 2019 — Bidirectional Encoder Representations from Transformers) dramatically advanced contextual embeddings by applying the Transformer architecture to bidirectional masked language modeling. BERT-based embeddings are now the foundation of state-of-the-art dense retrieval.

---

## 5. Sentence and Passage Embeddings

For retrieval, we need embeddings not of individual words but of **entire sentences or passages** — the units at which information is retrieved and inserted into prompts.

### 5.1 The Challenge of Passage Embeddings

Producing a high-quality embedding for a passage of 100–500 words is significantly harder than producing a word embedding. The embedding must:

- Capture the **main topic** of the passage (for broad relevance matching);
- Encode **specific details** (for precise fact retrieval);
- Be **comparable** to short query embeddings (queries are typically 5–25 words).

The query-document embedding asymmetry — that queries are short while documents are long — is a fundamental challenge in dense retrieval.

### 5.2 Mean Pooling: Averaging Word Embeddings

The simplest approach to sentence/passage embeddings is **mean pooling**: averaging the word/token embeddings produced by a pre-trained language model across all tokens in the passage.

```python
import numpy as np

def mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Compute mean-pooled passage embedding from token embeddings.

    Args:
        token_embeddings: Array of shape (seq_len, hidden_dim)
        attention_mask:   Binary mask of shape (seq_len,); 1 for real tokens, 0 for padding

    Returns:
        Passage embedding of shape (hidden_dim,)
    """
    # Expand mask to match embedding dimensions
    mask_expanded = attention_mask[:, np.newaxis]         # (seq_len, 1)
    # Zero out padding token embeddings
    masked_embeddings = token_embeddings * mask_expanded  # (seq_len, hidden_dim)
    # Average over non-padding tokens
    sum_embeddings = masked_embeddings.sum(axis=0)        # (hidden_dim,)
    sum_mask = mask_expanded.sum(axis=0)                  # (1,)
    return sum_embeddings / sum_mask                      # (hidden_dim,)
```

Mean pooling is simple and works reasonably well, but it does not inherently produce embeddings optimized for retrieval — the underlying model was not trained to make passage embeddings comparable to query embeddings.

### 5.3 CLS Token Embedding

BERT and similar models produce a special `[CLS]` (classification) token at the beginning of every sequence, designed to summarize the sequence's representation. For short texts, using the `[CLS]` token embedding directly is a common alternative to mean pooling.

In practice, neither mean pooling nor CLS embeddings from a vanilla BERT model produce retrieval-optimal representations — both require fine-tuning on retrieval-specific objectives, which leads to the DPR framework.

### 5.4 Sentence-Transformers: Embeddings Optimized for Similarity

**Sentence-BERT** (Reimers & Gurevych, 2019) introduced a Siamese network architecture that fine-tunes BERT to produce semantically meaningful sentence embeddings using **Natural Language Inference (NLI)** and **semantic textual similarity (STS)** data. The resulting models, available through the `sentence-transformers` library, produce embeddings where geometric proximity in the vector space directly corresponds to semantic similarity.

Key models from the Sentence-Transformers family relevant to RAG:

| Model | Dimensions | Strengths |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast, small, good general-purpose |
| `all-mpnet-base-v2` | 768 | Higher quality, slower |
| `e5-large-v2` | 1024 | State-of-the-art retrieval performance |
| `bge-large-en-v1.5` | 1024 | Strong on MTEB benchmark; good for RAG |
| `text-embedding-ada-002` | 1536 | OpenAI's embedding model (API) |
| `text-embedding-3-large` | 3072 | OpenAI's latest; high performance |

---

## 6. Dense Retrieval: The DPR Framework

**Dense Passage Retrieval** (DPR) (Karpukhin et al., 2020) is the framework that established dense retrieval as the standard for knowledge-intensive NLP tasks. It is the direct ancestor of every modern neural retriever used in RAG systems.

### 6.1 The Bi-Encoder Architecture

DPR uses a **bi-encoder** architecture: two independent BERT encoders that separately encode the query and the document (passage):

```
Query  → Query Encoder  (BERT_Q) → q ∈ ℝ^d
Passage → Passage Encoder (BERT_P) → p ∈ ℝ^d
```

The relevance score between query `q` and passage `p` is their **dot product**:

```
score(q, p) = q · p = Σ_i q_i × p_i
```

The bi-encoder's key advantage for retrieval: **passages can be encoded offline** (before any query arrives) and stored as static vectors. At query time, only the query encoder runs; retrieval is then a nearest-neighbor search over the pre-computed passage vectors. This enables extremely fast inference at scale.

### 6.2 Contrastive Training with In-Batch Negatives

DPR is trained using **contrastive learning**: the model is trained to maximize the dot product between a query and its relevant passage (positive) while minimizing the dot product between the query and irrelevant passages (negatives).

The training objective is a **negative log-likelihood** over a softmax distribution:

```
L = -log [ exp(sim(q, p+)) / (exp(sim(q, p+)) + Σ_j exp(sim(q, p_j^-))) ]
```

where `p+` is the positive (relevant) passage and `{p_j^-}` are negative (irrelevant) passages.

A crucial training technique is **in-batch negatives**: within each training batch, the positive passages of other questions serve as negatives for the current question. This is computationally efficient (no extra negative mining needed) and has been shown to produce highly effective models.

### 6.3 Asymmetric Bi-Encoder: Query vs. Document Towers

In practice, many modern retrieval systems use **asymmetric encoders**:
- A small, fast encoder for the query (must run at inference time);
- A large, high-capacity encoder for the documents (runs offline during indexing).

This asymmetry allows quality and speed to be independently optimized. Models like E5 and BGE support this with "query:" and "passage:" prefixes that activate different internal behavior:

```python
query_embedding   = model.encode("query: How does RAG reduce hallucinations?")
passage_embedding = model.encode("passage: RAG grounds LLM responses by...")
```

The prefix-based asymmetry is a form of instruction-following embedded in the embedding model — the model behaves differently depending on the prefix.

---

## 7. Similarity Metrics in Embedding Space

Once queries and documents are encoded as vectors, retrieval becomes a geometric problem: find the document vectors that are closest to the query vector. The choice of similarity metric matters.

### 7.1 Cosine Similarity

**Cosine similarity** measures the angle between two vectors, ignoring their magnitudes:

```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||)
```

Range: [-1, 1]. Value of 1 means the vectors point in the same direction (semantically identical); 0 means orthogonal (unrelated); -1 means opposite.

Cosine similarity is particularly appropriate when vector magnitude is not meaningful — for example, when embeddings are produced by models that may output vectors of varying magnitude depending on the input length.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 7.2 Dot Product

The **dot product** (inner product) is equivalent to cosine similarity when vectors are normalized to unit length:

```
dot_product(a, b) = a · b = Σ_i a_i × b_i
```

DPR and most modern retrieval models use dot product as the scoring function. When using normalized embeddings (via L2 normalization), dot product and cosine similarity are mathematically equivalent.

```python
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product between two vectors."""
    return np.dot(a, b)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length (L2 norm = 1)."""
    return v / np.linalg.norm(v)
```

### 7.3 Euclidean (L2) Distance

**L2 distance** measures the straight-line distance between two points in vector space:

```
L2(a, b) = sqrt(Σ_i (a_i - b_i)^2)
```

For retrieval, *lower* distance means *higher* relevance (unlike similarity metrics where higher is better). L2 distance is less commonly used than cosine similarity or dot product in NLP retrieval, but it is the native metric for some vector database implementations.

### 7.4 Choosing the Right Metric

The correct metric depends on how the embedding model was trained:

| Model trained with | Use this metric |
|---|---|
| Cosine similarity objective | Cosine similarity |
| Dot product / inner product objective | Dot product (or normalize first) |
| Euclidean loss | L2 distance |
| L2-normalized output | Either cosine or dot product (equivalent) |

Always check the embedding model's documentation. Using the wrong metric can dramatically degrade retrieval quality — even for an otherwise excellent model.

---

## 8. Vector Databases: Architecture and Indexing

A **vector database** is a data management system specialized for storing, indexing, and querying high-dimensional embedding vectors. It is the infrastructure layer that makes dense retrieval scalable from thousands to hundreds of millions of documents.

### 8.1 The Nearest-Neighbor Search Problem

Retrieval over a vector database reduces to **k-nearest-neighbor (k-NN) search**: given a query vector `q`, find the `k` database vectors most similar to `q`.

**Exact k-NN search** requires computing the similarity between `q` and every vector in the database — an `O(N × d)` operation where `N` is the number of vectors and `d` is the embedding dimension. For a corpus of 1 million 768-dimensional vectors, this requires roughly 768 million floating-point multiplications per query — feasible on GPU but too slow for real-time use on CPU.

**Approximate Nearest Neighbor (ANN) search** trades a small amount of recall for orders-of-magnitude speedup. The key insight: for retrieval, finding the *exact* top-k is rarely necessary — finding the *approximately* top-k with high probability is sufficient and far faster.

### 8.2 ANN Indexing Algorithms

Several ANN indexing strategies are widely used:

**HNSW (Hierarchical Navigable Small World)**

HNSW (Malkov & Yashunin, 2020) builds a hierarchical graph structure where each node (vector) is connected to its approximate nearest neighbors at multiple levels of granularity. Search proceeds by navigating this graph greedily from a coarse level to a fine level, converging rapidly on the nearest neighbors.

- **Strengths:** Extremely fast query time; high recall (typically >95%); no training required.
- **Weakness:** High memory usage; slow index construction.
- **Used in:** Weaviate, Qdrant, ChromaDB, FAISS.

**IVF (Inverted File Index)**

IVF partitions the vector space into `N` clusters (using k-means). At query time, only the vectors in the `M` most similar clusters are searched (where `M << N`), dramatically reducing the search space.

- **Strengths:** Low memory overhead; scales to very large corpora.
- **Weakness:** Cluster boundary artifacts; sensitive to the number of clusters.
- **Used in:** FAISS (IVF-Flat, IVF-PQ), Milvus.

**Product Quantization (PQ)**

PQ compresses high-dimensional vectors by decomposing them into subspaces and quantizing each subspace independently. This reduces storage by 4–64× with modest recall loss.

- Often combined with IVF (IVF-PQ) for extremely large corpora (billions of vectors).

### 8.3 Vector Database Comparison

| System | Open-Source | Hosting | Native Index | Hybrid Search |
|---|---|---|---|---|
| **Weaviate** | ✓ | Cloud / Self-hosted | HNSW | ✓ Built-in BM25 |
| **Pinecone** | ✗ | Cloud only | Proprietary | ✓ |
| **Qdrant** | ✓ | Cloud / Self-hosted | HNSW | ✓ |
| **ChromaDB** | ✓ | Local / Cloud | HNSW (via FAISS) | Limited |
| **Milvus** | ✓ | Cloud / Self-hosted | HNSW, IVF, PQ | ✓ |
| **FAISS** | ✓ | Library only | HNSW, IVF, PQ | ✗ (library) |
| **pgvector** | ✓ | PostgreSQL extension | HNSW, IVF | Manual |

**For RAG prototyping:** ChromaDB or FAISS (via LangChain or direct use) — minimal setup, in-memory operation, Python-native.

**For production RAG:** Weaviate, Qdrant, or Pinecone — persistent storage, filtering, hybrid search, production-grade reliability.

### 8.4 Metadata Filtering

Beyond pure vector similarity, production RAG systems need to filter by **metadata** — structured attributes attached to each document:

```python
# Example: retrieve only documents from a specific department and after a certain date
results = collection.query(
    query_embeddings=[query_vector],
    n_results=10,
    where={
        "department": "engineering",
        "date": {"$gte": "2024-01-01"}
    }
)
```

Metadata filtering enables RAG systems to respect access controls, scope retrieval to specific time windows, or restrict search to particular document categories — all without changing the embedding model.

---

## 9. Hybrid Retrieval: Combining Sparse and Dense

Neither BM25 nor dense retrieval is universally superior. Their failure modes are complementary:

| Failure Mode | BM25 | Dense Retrieval |
|---|---|---|
| Synonym / paraphrase | ✗ Fails | ✓ Handles |
| Exact keyword match | ✓ Strong | ✗ Can miss |
| Out-of-domain terminology | ✓ Lexical match | ✗ May struggle |
| Rare proper nouns | ✓ Exact match | ✗ May miss |
| Multi-hop semantics | ✗ Fails | ✓ Better |
| Short, ambiguous queries | ✗ Unreliable | ✓ More robust |

**Hybrid retrieval** combines both signals, consistently outperforming either alone.

### 9.1 Reciprocal Rank Fusion (RRF)

The simplest and most robust hybrid combination strategy is **Reciprocal Rank Fusion** (Cormack et al., 2009):

```python
def reciprocal_rank_fusion(
    sparse_results: list,   # List of (doc_id, score) from BM25, ranked
    dense_results:  list,   # List of (doc_id, score) from dense retrieval, ranked
    k: int = 60             # Smoothing constant (60 is standard)
) -> list:
    """
    Combine sparse and dense retrieval results using Reciprocal Rank Fusion.
    Each document's score is: Σ 1/(k + rank_in_list)
    Documents appearing in both lists get contributions from both.
    """
    rrf_scores = {}

    for rank, (doc_id, _) in enumerate(sparse_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    # Sort by combined RRF score (descending)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

RRF is particularly attractive because it does not require score normalization — BM25 scores and cosine similarities are on different scales, but rank positions are directly comparable.

### 9.2 Score Normalization and Weighted Combination

An alternative to RRF is to **normalize** both score distributions to [0, 1] and compute a weighted sum:

```
hybrid_score(d) = α × dense_score_norm(d) + (1 - α) × sparse_score_norm(d)
```

The weight `α` is a hyperparameter (typically 0.5–0.7 favoring dense) tuned on a validation set. This approach is more interpretable than RRF but requires careful normalization.

### 9.3 When to Use Hybrid Retrieval

Hybrid retrieval is the recommended default for production RAG systems. Pure dense retrieval is appropriate only when:
- The corpus is entirely in a specialized domain where the embedding model was fine-tuned;
- Exact keyword matching is never required;
- The query distribution closely matches the dense model's training distribution.

In most real-world enterprise applications, hybrid retrieval provides the best balance of semantic understanding (dense) and lexical precision (sparse).

---

## 10. Implementing Dense Retrieval in Python

We now build a working dense retriever using `sentence-transformers` for embeddings and `faiss` for vector indexing.

### 10.1 Setup and Dependencies

```python
# Install required packages
# pip install sentence-transformers faiss-cpu numpy

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
```

### 10.2 Encoding Documents

```python
# Load a pre-trained embedding model
# 'all-MiniLM-L6-v2' is fast and effective for English retrieval
model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384  # Output dimension for this model

# Knowledge base: a list of document strings
knowledge_base = [
    "RAG stands for Retrieval-Augmented Generation. It improves LLM accuracy by "
    "providing relevant documents at inference time.",

    "Hallucinations occur when an LLM generates plausible-sounding but factually "
    "incorrect text. RAG reduces hallucinations by grounding responses in retrieved documents.",

    "The retriever component searches the knowledge base and returns the most relevant "
    "documents for a given query. It assigns a relevance score to each document.",

    "Chunking splits long documents into smaller segments so they fit within the LLM's "
    "context window and can be retrieved individually.",

    "Vector databases store document embeddings — dense numerical representations of text — "
    "and support fast approximate nearest-neighbor search over millions of documents.",

    "BM25 is a classical sparse retrieval algorithm. It improves on TF-IDF by adding "
    "term frequency saturation and document length normalization.",

    "Hybrid retrieval combines dense (semantic) and sparse (keyword) retrieval. "
    "Reciprocal Rank Fusion (RRF) is a simple, effective combination strategy.",

    "Sentence-Transformers is a Python library for computing dense sentence embeddings "
    "using pre-trained BERT-based models fine-tuned for semantic similarity tasks.",
]

# Encode all documents (offline — done once during indexing)
print("Encoding knowledge base...")
doc_embeddings = model.encode(
    knowledge_base,
    normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
    show_progress_bar=True
)

print(f"Encoded {len(doc_embeddings)} documents, each of dimension {doc_embeddings.shape[1]}")
# Output: Encoded 8 documents, each of dimension 384
```

### 10.3 Building the FAISS Index

```python
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index for fast inner-product (cosine) search.

    Args:
        embeddings: Array of shape (N, d) — N documents, d dimensions.

    Returns:
        A FAISS index containing all document embeddings.
    """
    d = embeddings.shape[1]

    # IndexFlatIP: exact inner product search (equivalent to cosine when normalized)
    # For large corpora, replace with IndexHNSWFlat or IndexIVFFlat for speed
    index = faiss.IndexFlatIP(d)

    # FAISS requires float32
    index.add(embeddings.astype(np.float32))

    print(f"FAISS index built: {index.ntotal} vectors, dimension {d}")
    return index


# Build the index
faiss_index = build_faiss_index(doc_embeddings)
```

### 10.4 The Dense Retriever Function

```python
def dense_retrieve(
    query: str,
    index: faiss.Index,
    corpus: List[str],
    model: SentenceTransformer,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Dense retrieval using semantic embeddings and FAISS.

    Args:
        query:  The user's question.
        index:  The FAISS index containing document embeddings.
        corpus: The original document texts (parallel to the index).
        model:  The sentence embedding model.
        top_k:  Number of results to return.

    Returns:
        List of (document_text, similarity_score) tuples, ranked by relevance.
    """
    # Encode the query with the same model (online — runs at query time)
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    # Search the FAISS index
    # Returns: distances (similarity scores), indices (positions in corpus)
    scores, indices = index.search(query_embedding, top_k)

    # Assemble results
    results = [
        (corpus[idx], float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx < len(corpus)  # Safety check
    ]

    return results
```

### 10.5 Running the Dense Retriever

```python
# Test queries — including ones with vocabulary mismatch
test_queries = [
    "How does RAG reduce memory issues in neural networks?",      # Vocabulary mismatch test
    "Why does RAG prevent the model from making things up?",      # Paraphrase of hallucination
    "What technique splits large texts for storage?",             # Paraphrase of chunking
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 60)
    results = dense_retrieve(query, faiss_index, knowledge_base, model, top_k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"  [{i}] Score: {score:.3f}")
        print(f"       {doc[:80]}...")
```

**Example output:**
```
Query: 'Why does RAG prevent the model from making things up?'
------------------------------------------------------------
  [1] Score: 0.721
       Hallucinations occur when an LLM generates plausible-sounding but factually...
  [2] Score: 0.634
       RAG stands for Retrieval-Augmented Generation. It improves LLM accuracy by...
  [3] Score: 0.502
       The retriever component searches the knowledge base and returns the most...
```

The dense retriever correctly identifies the hallucination document as most relevant despite the query containing none of the words "hallucination," "factually," "incorrect," or "plausible" — exactly the vocabulary mismatch scenario where keyword retrieval would fail.

### 10.6 Completing the Dense RAG Pipeline

```python
from utils import generate_with_multiple_input

def dense_rag_pipeline(
    user_question: str,
    corpus: List[str],
    index: faiss.Index,
    embedding_model: SentenceTransformer,
    top_k: int = 3,
    max_tokens: int = 300
) -> str:
    """
    Complete RAG pipeline with dense retrieval.
    Retrieve (semantic) → Augment → Generate.
    """
    # Step 1: Dense retrieval
    results = dense_retrieve(user_question, index, corpus, embedding_model, top_k)
    retrieved_docs = [doc for doc, score in results]
    retrieved_scores = [score for doc, score in results]

    # Step 2: Build augmented prompt
    context_lines = [
        f"[Document {i+1} | similarity: {score:.3f}]\n{doc}"
        for i, (doc, score) in enumerate(zip(retrieved_docs, retrieved_scores))
    ]
    context_block = "\n\n".join(context_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise research assistant. "
                "Answer questions using ONLY the provided documents. "
                "Cite document numbers (e.g., '[Document 1]') when referencing specific information. "
                "If the answer is not in the documents, say so explicitly."
            )
        },
        {
            "role": "user",
            "content": f"Documents:\n{context_block}\n\nQuestion: {user_question}\n\nAnswer:"
        }
    ]

    # Step 3: Generate
    output = generate_with_multiple_input(messages=messages, max_tokens=max_tokens, temperature=0.0)
    return output['content']
```

---

## 11. Chunking: Preparing Documents for Retrieval

Before documents can be indexed, they must be **chunked** — divided into retrievable segments. This is one of the most consequential design decisions in RAG engineering, and it is not fully solved even in production systems.

### 11.1 Why Chunking Is Necessary

Three constraints motivate chunking:

1. **Embedding quality degrades with length.** Sentence-Transformer models are trained on sentences and short paragraphs. Encoding a 50-page PDF as a single vector loses almost all specific information in a single averaged representation.

2. **The LLM context window is finite.** Inserting 10 full documents into a prompt may exhaust the context budget. Inserting 10 targeted 300-word chunks is much more efficient and often produces better responses.

3. **Retrieval precision improves with granularity.** A smaller chunk that precisely matches the query is more useful than a large document that happens to contain the answer buried among unrelated content.

### 11.2 Fixed-Size Chunking

The simplest strategy: divide the text into chunks of exactly `chunk_size` tokens/words, with an optional `overlap` between consecutive chunks to prevent context from being cut at chunk boundaries.

```python
def fixed_size_chunk(
    text: str,
    chunk_size: int = 256,    # words per chunk
    chunk_overlap: int = 50   # words to overlap between consecutive chunks
) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text:          The document text to chunk.
        chunk_size:    Target number of words per chunk.
        chunk_overlap: Number of words to repeat between adjacent chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks = []
    stride = chunk_size - chunk_overlap

    for start in range(0, len(words), stride):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break

    return chunks


# Example
long_document = """Retrieval-Augmented Generation (RAG) is a technique... [full text]"""
chunks = fixed_size_chunk(long_document, chunk_size=100, chunk_overlap=20)
print(f"Document split into {len(chunks)} chunks")
```

**The overlap parameter** is critical: without overlap, a sentence split across two chunks will be half-truncated in both — potentially destroying the meaning. With overlap, the sentence appears fully in at least one chunk.

### 11.3 Semantic Chunking

Fixed-size chunking is topic-agnostic — it may split a paragraph mid-thought. **Semantic chunking** attempts to split at natural topic boundaries by measuring semantic similarity between consecutive sentences:

```python
def semantic_chunk(
    sentences: List[str],
    model: SentenceTransformer,
    similarity_threshold: float = 0.5
) -> List[str]:
    """
    Chunk a document by grouping semantically similar consecutive sentences.
    A new chunk is started when similarity to the previous sentence drops below threshold.
    """
    if not sentences:
        return []

    embeddings = model.encode(sentences, normalize_embeddings=True)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i], embeddings[i-1])  # cosine (normalized)
        if sim >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

Semantic chunking tends to produce more coherent chunks at the cost of variable chunk sizes — some chunks may be very short or very long, requiring additional size-based post-processing.

### 11.4 Hierarchical Chunking: Parent-Child Documents

A sophisticated strategy used in production systems is **hierarchical chunking** (also called parent-child retrieval):

1. Split the document into **small, precise child chunks** (e.g., 100 words) for retrieval;
2. Also store **larger parent chunks** (e.g., full paragraphs or sections) that contain each child;
3. At retrieval time, retrieve the small child chunk (for precision), but insert its parent chunk into the prompt (for context).

This gives the retriever precision without sacrificing the broader context the LLM needs to generate a good answer.

### 11.5 Chunking Hyperparameters: Practical Guidance

| Parameter | Typical Range | Effect of Increasing |
|---|---|---|
| `chunk_size` | 100–512 tokens | More context per chunk; lower retrieval precision |
| `chunk_overlap` | 10–20% of chunk_size | Fewer context cuts at boundaries; more storage |
| `top_k` (retrieved chunks) | 3–10 | More coverage; larger prompt; higher cost |
| `similarity_threshold` (semantic) | 0.4–0.7 | Higher → more fine-grained splits |

There is no universally optimal setting. These are hyperparameters that must be tuned empirically against your specific corpus and query distribution (covered in Part 5: Evaluation).

---

## 12. Evaluating Retrieval Quality

A RAG system can only be as good as its retriever. Before evaluating the end-to-end system, it is essential to evaluate the retriever in isolation.

### 12.1 Retrieval Evaluation Metrics

**Recall@k:** Of all documents that are relevant to a query, what fraction does the retriever find in its top-k results?

```
Recall@k = |relevant ∩ top_k_retrieved| / |relevant|
```

A Recall@k of 1.0 means the retriever found all relevant documents in its top-k. This metric matters most for RAG: if the relevant document is not retrieved, the LLM has no chance of answering correctly.

**Precision@k:** Of the k retrieved documents, what fraction is relevant?

```
Precision@k = |relevant ∩ top_k_retrieved| / k
```

High precision means fewer irrelevant documents enter the prompt, reducing noise and cost.

**MRR (Mean Reciprocal Rank):** For each query, the reciprocal rank is `1/rank_of_first_relevant_result`. MRR averages this across all queries:

```
MRR = (1/|Q|) × Σ_{q ∈ Q} (1 / rank_q)
```

MRR rewards retrievers that place the most relevant document first.

**NDCG@k (Normalized Discounted Cumulative Gain):** A graded metric that gives higher credit for relevant documents appearing earlier in the ranked list. NDCG@k is the standard metric on major retrieval benchmarks (BEIR, MTEB).

### 12.2 Standard Benchmarks

| Benchmark | Description | Key Metric |
|---|---|---|
| **BEIR** | 18 diverse retrieval tasks; zero-shot evaluation | NDCG@10 |
| **MTEB** | 58 tasks across 8 categories of embedding evaluation | Average NDCG |
| **MS-MARCO** | 1M queries from Bing search logs; passage retrieval | MRR@10, Recall@1000 |
| **Natural Questions** | Google search queries answered from Wikipedia | Recall@20, Recall@100 |

### 12.3 A Simple Evaluation Framework

```python
def evaluate_retriever(
    queries: List[str],
    ground_truth: List[List[str]],  # For each query, list of relevant doc texts
    corpus: List[str],
    retrieve_fn,                    # Any retriever function: query → List[(doc, score)]
    k: int = 5
) -> dict:
    """
    Evaluate a retriever by Recall@k, Precision@k, and MRR.

    Args:
        queries:       List of test queries.
        ground_truth:  For each query, the list of relevant documents.
        corpus:        The full knowledge base.
        retrieve_fn:   A function that takes a query and returns List[(doc, score)].
        k:             Evaluation depth.

    Returns:
        Dict with mean Recall@k, Precision@k, and MRR.
    """
    recalls, precisions, rrs = [], [], []

    for query, relevant_docs in zip(queries, ground_truth):
        results = retrieve_fn(query)[:k]
        retrieved_texts = [doc for doc, score in results]

        # Convert to sets for set operations
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_texts)

        # Recall@k
        hits = len(relevant_set & retrieved_set)
        recall = hits / len(relevant_set) if relevant_set else 0
        recalls.append(recall)

        # Precision@k
        precision = hits / k
        precisions.append(precision)

        # MRR: find rank of first relevant document
        rr = 0
        for rank, doc in enumerate(retrieved_texts, start=1):
            if doc in relevant_set:
                rr = 1 / rank
                break
        rrs.append(rr)

    return {
        f"Recall@{k}":    np.mean(recalls),
        f"Precision@{k}": np.mean(precisions),
        "MRR":            np.mean(rrs),
    }
```

---

## 13. Key Concepts Summary

| Concept | Definition |
|---|---|
| **Semantic Gap** | The mismatch between surface vocabulary (words) and underlying meaning; the core limitation of keyword retrieval. |
| **TF-IDF** | A sparse retrieval scoring function weighting terms by frequency in the document (TF) and rarity across the corpus (IDF). |
| **BM25** | An improved sparse retrieval function adding TF saturation and document length normalization; the dominant baseline. |
| **Embedding** | A dense numerical vector representation of text that encodes semantic meaning; geometrically similar vectors represent similar meanings. |
| **Word2Vec** | A neural model that learns word embeddings via next-word prediction; captures semantic analogies as vector arithmetic. |
| **Sentence-Transformers** | BERT-based models fine-tuned on semantic similarity tasks; produce embeddings optimized for passage retrieval. |
| **DPR (Dense Passage Retrieval)** | The bi-encoder framework for dense retrieval; encodes queries and passages separately, then retrieves by maximum inner product. |
| **Bi-Encoder** | A retrieval architecture with two independent encoders for queries and documents; enables offline document indexing. |
| **Cosine Similarity** | A similarity metric between vectors measuring the angle between them; scale-invariant; range [-1, 1]. |
| **FAISS** | Meta's open-source library for efficient approximate nearest-neighbor search; the standard building block for dense retrieval. |
| **HNSW** | A graph-based ANN index algorithm; very fast query time; default index in Weaviate, Qdrant, ChromaDB. |
| **Hybrid Retrieval** | Combining dense (semantic) and sparse (keyword) retrieval results; consistently outperforms either alone. |
| **RRF (Reciprocal Rank Fusion)** | A rank-based combination strategy for hybrid retrieval; robust, scale-invariant, no hyperparameter tuning needed. |
| **Chunking** | Dividing long documents into smaller retrievable segments; a critical preprocessing step whose parameters significantly affect RAG quality. |
| **Chunk Overlap** | Repeated words between adjacent chunks to prevent context from being lost at chunk boundaries. |
| **Recall@k** | The fraction of all relevant documents found in the top-k retrieval results; the primary metric for RAG retrieval evaluation. |
| **NDCG@k** | Normalized Discounted Cumulative Gain; a graded retrieval metric that accounts for position — higher-ranked results are worth more. |

---

## 14. Further Reading and Foundational Papers

### Classical Information Retrieval

- **Robertson, S. E., & Zaragoza, H. (2009).** *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval. [`Direct PDF`](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf)  
  *(The definitive reference on BM25 — required reading for understanding the sparse retrieval baseline.)*

- **Manning, C. D., Raghavan, P., & Schütze, H. (2008).** *Introduction to Information Retrieval.* Cambridge University Press. [`Free online`](https://nlp.stanford.edu/IR-book/)  
  *(Chapters 1–6 cover TF-IDF, indexing, and evaluation metrics; Chapter 19 introduces web-scale retrieval.)*

### Word and Sentence Embeddings

- **Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).** *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013. [`arXiv:1310.4546`](https://arxiv.org/abs/1310.4546)  
  *(Introduced Word2Vec Skip-Gram with negative sampling — the foundational word embedding model.)*

- **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. [`arXiv:1810.04805`](https://arxiv.org/abs/1810.04805)  
  *(BERT introduced contextual embeddings that underlie all modern dense retrievers.)*

- **Reimers, N., & Gurevych, I. (2019).** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. [`arXiv:1908.10084`](https://arxiv.org/abs/1908.10084)  
  *(Introduced Sentence-Transformers — the framework and model family used directly in RAG retrieval.)*

### Dense Retrieval

- **Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020).** *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. [`arXiv:2004.04906`](https://arxiv.org/abs/2004.04906)  
  *(Introduced DPR — the bi-encoder architecture that defines modern dense retrieval. A must-read.)*

- **Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., ... & Wei, F. (2022).** *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).* [`arXiv:2212.03533`](https://arxiv.org/abs/2212.03533)  
  *(State-of-the-art text embedding model; top-performing on MTEB benchmark; widely used in production RAG.)*

### ANN and Vector Databases

- **Malkov, Y. A., & Yashunin, D. A. (2020).** *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI. [`arXiv:1603.09320`](https://arxiv.org/abs/1603.09320)  
  *(The HNSW algorithm — the index structure powering Weaviate, Qdrant, and ChromaDB.)*

- **Johnson, J., Douze, M., & Jégou, H. (2019).** *Billion-scale Similarity Search with GPUs.* IEEE Big Data. [`arXiv:1702.08734`](https://arxiv.org/abs/1702.08734)  
  *(The FAISS library paper — essential reference for understanding ANN indexing at scale.)*

### Hybrid Retrieval

- **Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).** *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009.  
  *(Introduced RRF — simple, robust, and still the standard for combining retrieval signals.)*

- **Ma, X., Wang, L., Yang, N., Wei, F., & Lin, J. (2022).** *Fine-Tuning LLaMA for Multi-Stage Text Retrieval.* [`arXiv:2310.08319`](https://arxiv.org/abs/2310.08319)  
  *(Modern approach to combining sparse and dense retrieval in a unified pipeline.)*

### Evaluation Benchmarks

- **Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).** *BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of Information Retrieval Models.* NeurIPS 2021. [`arXiv:2104.08663`](https://arxiv.org/abs/2104.08663)  
  *(The BEIR benchmark — the standard for evaluating retrieval models across diverse domains.)*

---

## 15. Review Questions

> **Difficulty guide:** ★ Foundational · ★★ Intermediate · ★★★ Advanced

### Conceptual Understanding

1. ★ Explain the semantic gap in your own words. Give an original example (not from the text) of a user query and a relevant document that would cause a keyword retriever to return a near-zero relevance score despite high semantic relevance.

2. ★ BM25 improves over raw TF by adding a saturation function. What specific failure mode of raw TF does this saturation function address? Would you expect BM25 to perform better or worse than TF-IDF on a corpus of short product reviews (avg. 50 words)? Justify your answer.

3. ★★ The distributional hypothesis states that "a word is characterized by the company it keeps." How does this philosophical principle translate into the concrete mathematical training objective of Word2Vec (Skip-Gram)?

4. ★★ Explain why a bi-encoder architecture (as in DPR) is preferred over a cross-encoder for retrieval at scale, even though cross-encoders typically produce higher-quality similarity scores. Under what conditions might a cross-encoder be used in a RAG system?

5. ★★★ The "lost in the middle" problem (Liu et al., 2023) suggests LLMs attend poorly to information in the middle of long contexts. How does this interact with the chunking decision? If you must retrieve 10 chunks to achieve adequate recall, but the LLM only reliably uses the first and last few, what retrieval and augmentation strategies would you combine to mitigate this?

### Engineering Practice

6. ★ You have built a dense retriever using cosine similarity. A colleague suggests you should switch to L2 distance. Under what condition would cosine similarity and L2 distance produce identical rankings? Write a one-line Python function that converts a cosine-similarity retriever into an L2-equivalent one.

7. ★★ Your dense retriever is deployed with `chunk_size=512` tokens and `top_k=5`. You notice the LLM's responses are often generic and miss specific details. Propose two independent changes to the retrieval configuration and explain the expected effect of each on response quality.

8. ★★ Implement a BM25 scorer from scratch in Python. Your implementation should:
   - Accept a corpus of documents and a query as input;
   - Compute IDF scores for all query terms;
   - Apply TF saturation (use `k1=1.5`);
   - Apply length normalization (use `b=0.75`);
   - Return the top-k documents by BM25 score.

9. ★★ You are building a RAG system for a legal firm. The attorneys frequently search for specific statute numbers (e.g., "§ 214(b)(2)(A)") that appear verbatim in documents. Explain why dense retrieval alone would struggle with this requirement and design a hybrid retrieval configuration that handles it.

10. ★★★ A product manager asks you to compare three retrieval configurations on the company's internal knowledge base:
    - Configuration A: BM25 only
    - Configuration B: Dense (E5-large) only  
    - Configuration C: Hybrid (BM25 + Dense, RRF combination)
    
    Design a rigorous evaluation protocol (evaluation set construction, metrics, statistical significance testing) to determine which configuration is best for your specific use case. What are the risks of evaluating on a non-representative query set?

### Design and Analysis

11. ★★ A competitor claims their RAG system achieves "95% Recall@10." Without further context, is this a strong or weak claim? What additional information would you need to evaluate it properly? Consider: What is the corpus size? What query distribution was used? How were "relevant documents" defined?

12. ★★★ Semantic chunking (§11.3) splits documents at points of low inter-sentence similarity. Describe a document type where this approach would systematically produce poor chunks, and propose an alternative chunking strategy tailored to that document type. Hint: consider structured documents like financial reports, legal contracts, or scientific papers.

---

*End of Part 3 — Dense Retrieval, Embeddings, and Vector Databases*

---

> **Up Next:** [Part 4 — Advanced Prompt Engineering and LLM Output Control](./04_Advanced_Prompt_Engineering.md)  
> In Part 4, we move beyond basic augmented prompts to explore system message design, output formatting, chain-of-thought reasoning over retrieved context, citation generation, and structured output extraction.

---

*This document is part of the **Retrieval-Augmented Generation: From Foundations to Production** learning series.*  
*Content adapted from DeepLearning.AI RAG course materials and enriched with foundational academic references in information retrieval, representation learning, and vector database systems.*
