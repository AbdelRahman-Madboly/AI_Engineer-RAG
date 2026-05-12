# RAG Engineering Course — Notebook Reference

A complete guide to every lab notebook in the course: what it covers, what you build,
how it connects to the reference documents, and what those documents cover that the labs do not.

---

## How to use this document

Each lab section lists what the lab teaches and builds, then ends with a **"Covered by the reference doc but not this lab"** note — topics from the module documents that you need to study separately. Those gaps are not missing content; they are the conceptual and theoretical layer that the labs assume you have read.

The short version: **do the labs to build things, read the docs to understand why**.

---

## Module 01 — RAG Overview

**Labs path:** `projects/01_RAG_Overview/labs/`
**Reference document:** `01_RAG_Overview.docx`

The reference document covers the full theoretical foundation: the knowledge problem, how LLMs work mechanically, all three LLM limitations, every RAG application domain, the comparison against fine-tuning and long-context approaches, agentic RAG, and 22 interview questions with full model answers. The labs cover the hands-on core: Python patterns, LLM calls, and a working end-to-end RAG pipeline.

---

### lab01 — Python Essentials
**File:** `lab01_python_essentials.ipynb`

Every RAG pipeline is built from a handful of Python patterns that repeat everywhere.
This lab teaches exactly those patterns — nothing more, nothing less.

**What you learn:**
- Lists: indexing, slicing, `append` vs `extend`, list comprehensions
- Dicts: `.get()` with defaults, `.items()`, lists of dicts (the shape of every retrieved document)
- f-strings: building prompt templates from a query and a context block
- Functions with type hints: `List[Dict]`, `str`, `float` — how `utils.py` is written

**What you build:**
- `format_documents()` — converts a list of retrieved doc dicts into a formatted context block
- `build_rag_prompt()` — assembles a complete augmented prompt from a query and documents

**Exercises:** build a conversation history list, filter documents by score with a list comprehension, write a retrieval summary function.

**Covered by the reference doc but not this lab:**
The doc's Section 3.4 shows the same three functions — `retrieve()`, `build_augmented_prompt()`, `generate()` — as pseudocode explaining the full RAG architecture. Read that section alongside this lab to understand where the code you write here fits in the bigger picture.

---

### lab02 — Environment Setup and utils.py
**File:** `lab02_environment_and_utils.ipynb`

Before writing any RAG code, you need a working LLM connection. This lab explains
exactly how `utils.py` works and confirms your backend is responding.

**What you learn:**
- How the `.env` file controls which backend is active (Ollama / Gemini / Together)
- What `generate_with_single_input` and `generate_with_multiple_input` do internally
- Every parameter — `role`, `max_tokens`, `temperature` — and what it controls
- How to switch backends without changing any notebook code

**What you build:**
- A live connection test that gives you a clear error message if something is wrong
- A reusable `chat()` function you will use in labs 04 and 05

**Covered by the reference doc but not this lab:**
The doc's Section 4.4 explains temperature mechanically: it controls the shape of the probability distribution at each token sampling step, not just a "creativity dial". Section 4.3 explains why the same prompt produces different outputs on different runs (stochastic sampling). Read these to understand what `temperature=0.0` actually does inside the model.

---

### lab03 — LLM Single Calls
**File:** `lab03_llm_single_calls.ipynb`

A focused lab on `generate_with_single_input`. You run small experiments and observe
exactly how each parameter changes the output.

**What you learn:**
- The return value structure: always `{'role': 'assistant', 'content': '...'}`
- `role='user'` vs `role='system'` — what each one does
- `max_tokens`: how cutting off the response early affects the output
- `temperature=0.0` vs `temperature=1.0`: deterministic vs creative outputs
- Calling the LLM in a loop to batch-process a list of inputs

**What you build:**
- A `classify_topic()` function that labels each document with one category
- `answer_with_context()` — takes a query and a context string, returns a grounded answer

**Key insight:** `temperature=0.0` is the right setting for RAG. You want consistent, grounded answers — not creative variation.

**Covered by the reference doc but not this lab:**
The doc's Section 4.1 explains tokenization: the model does not process words, it processes subword tokens produced by Byte Pair Encoding. This matters because `max_tokens` limits token count, not word count, and long words count as multiple tokens. Section 4.2 explains the generation loop — why each token choice constrains every subsequent one (autoregressive behavior), and why early errors propagate and amplify.

---

### lab04 — Conversations and System Prompts
**File:** `lab04_llm_conversations.ipynb`

Entirely focused on `generate_with_multiple_input`: multi-turn conversations, system prompts,
and growing a conversation history dynamically.

**What you learn:**
- The messages list format: `[{'role': ..., 'content': ...}, ...]`
- System messages: how they shape the model's behaviour for the whole conversation
- How the model uses all prior turns as context when answering the latest question
- Building conversation history in-place as the conversation grows

**What you build:**
- A `chat()` function that appends each user message and LLM response to history
- A simulated RAG chatbot that remembers what was retrieved across multiple turns
- A product support chatbot with a system prompt that restricts the topic

**Key insight:** The model has no memory between calls. You give it the full history every time. This is why the messages list exists.

**Covered by the reference doc but not this lab:**
The doc's Section 5.2 covers the context window constraint precisely: the O(n²) computational cost of self-attention means processing a prompt twice as long takes roughly four times the computation. This is why you cannot simply pass unlimited history. The "Lost in the Middle" finding (Liu et al. 2023, referenced in Section 7.2) shows that models attend well to information at the start and end of long contexts but underuse what is in the middle — which is why conversation history management and selective retrieval both matter.

---

### lab05 — Augmented Prompts: The RAG Core Loop
**File:** `lab05_augmented_prompts.ipynb`

The central lab of Module 01. You implement the complete RAG pipeline from scratch
using a hardcoded knowledge base — no vector database yet.

**What you learn:**
- The three-step loop: Retrieve → Augment → Generate
- Why retrieved documents go into the prompt instead of the model's weights
- The difference between a grounded answer and a hallucinated one
- What happens when the answer is not in the knowledge base

**What you build:**
- `simple_retriever()` — keyword-overlap scoring over a list of documents
- `build_augmented_prompt()` — formats retrieved docs and query into one prompt
- `rag()` — a single function that runs the full pipeline end-to-end

**Key demonstration:** the same question answered without RAG (bare LLM, may hallucinate) and with RAG (grounded by retrieved documents). The difference is visible and concrete.

**Covered by the reference doc but not this lab:**
The doc makes a distinction this lab does not: **intrinsic hallucination** (the model contradicts the provided context) vs **extrinsic hallucination** (the model adds information not present in the context). The lab demonstrates extrinsic hallucination. The doc's Section 5.1 explains why these are different failure modes requiring different fixes.

The doc also names and explains **in-context learning** (Section 2.3) as the specific LLM capability that makes RAG possible without retraining: any information placed in the augmented prompt is available to the model regardless of whether it appeared in training data. The lab uses this capability but does not name it.

The doc's Section 3.3 enumerates the **five structural advantages** of RAG: access to unavailable information, hallucination reduction, easy knowledge updates without retraining, source citation, and separation of concerns. The lab demonstrates the first two clearly but does not enumerate all five.

---

### lab06 — Manual Retrieval: What Breaks and Why
**File:** `lab06_manual_retrieval.ipynb`

The bridge between Module 01 and Module 02. This lab deliberately breaks the naive
retriever from lab05 and shows exactly what fails — so you understand why the Module 02
algorithms were invented.

**What you learn:**
- When keyword overlap works (exact shared words)
- Vocabulary mismatch: the query says "automobile", the document says "car" — no match
- The stop-word problem: "the", "is", "in" appear everywhere and inflate scores incorrectly
- How bad retrieval poisons the LLM's answer even when the LLM itself is good

**Covered by the reference doc but not this lab:**
The failures this lab demonstrates are exactly what the Module 02 reference document opens with (Section 1) — vocabulary mismatch framed as the central challenge of all information retrieval. The lab shows the failures empirically; the Module 02 doc explains the theoretical grounding for why each failure happens and names the algorithms that fix each one. Read Module 02 Section 1 immediately after this lab.

---

## Module 01 — Topics in the reference document not covered by any lab

These are important for understanding and interviews, but are study topics rather than coding topics.

**RAG applications (Section 6):** code generation, enterprise chatbots, healthcare and legal systems, AI-assisted web search, personalized assistants. The doc explains why each domain needs RAG and what makes it a good fit.

**RAG vs fine-tuning vs long-context (Section 7):** when to use each approach, the full comparison table, why fine-tuning is the wrong tool for the knowledge currency problem, how RAG and fine-tuning are complementary rather than competing.

**Agentic RAG (Section 8):** dynamic retrieval decisions, multi-step retrieval for complex queries, Self-RAG (Asai et al. 2023), and the direction the field is moving. None of the labs touch this.

**Foundational papers (Section 11):** Lewis et al. 2020 (named RAG), Vaswani et al. 2017 (Attention Is All You Need), Liu et al. 2023 (Lost in the Middle), Ji et al. 2023 (hallucination survey), Asai et al. 2023 (Self-RAG), Gao et al. 2024 (RAG survey), Robertson & Zaragoza 2009 (BM25), Karpukhin et al. 2020 (Dense Passage Retrieval).

**The 22 interview Q&A (Section 10):** hallucination mechanics, autoregressive generation, two-phase model, grounding, context window constraints, knowledge cutoff, intrinsic vs extrinsic hallucination, RAG architecture end to end, chunking trade-offs, precision-recall tradeoff, RAG vs fine-tuning, and applied design scenarios including a hospital drug interaction system and a customer service chatbot.

---

## Module 02 — Information Retrieval and Search Foundations

**Labs path:** `projects/02_Information_Retrieval_and_Search_Foundations/labs/`
**Reference document:** `02_Information_Retrieval.docx`

The reference document covers the full hybrid retrieval pipeline architecture, the theoretical grounding for every algorithm, worked numerical examples for every metric, the design rationale for every pipeline decision (why parallel search, why filter after search, why RRF uses ranks not scores), and 22 interview questions. The labs implement all the algorithms from scratch and evaluate them on real data.

---

### lab01 — Vector Embeddings
**File:** `lab01_vector_embeddings.ipynb`

Embeddings are what make semantic search work. This lab makes them completely concrete:
you compute them, compare them, visualize them, and discover their limits.

**What you learn:**
- Cosine similarity: measures the angle between vectors (direction = meaning)
- Euclidean distance: measures the straight-line gap (less useful for text)
- Why cosine similarity is preferred for semantic search
- The `BAAI/bge-base-en-v1.5` model: 768-dimensional embeddings
- The 512-token input limit and why it forces chunking

**What you build:**
- `cosine_similarity()` and `euclidean_distance()` from scratch using NumPy
- `retrieve_by_similarity()` — embed a query and rank documents by cosine score
- A PCA visualization projecting 768 dimensions down to 2 for plotting

**Key demonstration:** "car" and "automobile" end up close together in vector space; "car" and "love" are far apart. The model learned meaning, not just spelling.

**Covered by the reference doc but not this lab:**
The doc's Section 5.1 explains the **curse of dimensionality** — why Euclidean distance becomes unreliable in high-dimensional space (distances concentrate into a narrow range, making it hard to distinguish similar from dissimilar texts). The lab shows that cosine works better; the doc explains the mathematical reason why.

The doc also covers the **dot product vs cosine** distinction: they are equivalent only when vectors are L2-normalized, and using the wrong metric for a given model (e.g. applying cosine to a model trained with dot product on unnormalized vectors) produces subtly incorrect rankings that are hard to diagnose.

The doc's Section 5.2 covers **contrastive training** in depth — how the embedding model learns from positive and negative pairs, and why the resulting geometric space is not manually designed but emerges from statistical patterns in the training data. Crucially, it explains **model incompatibility**: vectors from two different embedding models live in incompatible coordinate systems, so comparing them is meaningless. If you update the embedding model, every document must be re-embedded from scratch.

---

### lab02 — Keyword Search: TF-IDF and BM25
**File:** `lab02_keyword_search.ipynb`

BM25 is the standard keyword search algorithm — inside Elasticsearch, Weaviate, and most
production search systems. This lab builds it from scratch so you understand every formula.

**What you learn:**
- Bag-of-words: word order is ignored, only counts matter
- TF-IDF: Term Frequency × Inverse Document Frequency — rare words score higher
- BM25's two improvements over TF-IDF: term saturation (`k1`) and length normalization (`b`)
- The `rank-bm25` library for production use

**What you build:**
- `compute_tf()` and `compute_idf()` from scratch
- `tfidf_score()` — ranks a document for a query
- A full `BM25` class from scratch with the complete scoring formula
- A saturation experiment: TF-IDF grows linearly, BM25 plateaus

**Covered by the reference doc but not this lab:**
The doc's Section 4.1 includes a **worked IDF example** with a 1,000-document corpus: the word "in" (appearing in 990 docs) gets IDF = 0.01; "GPU" (appearing in 12 docs) gets IDF = 4.42. This makes the intuition for rare-word weighting concrete with real numbers.

The doc also explains the **inverted index** structure in detail: why it is called "inverted", how sparse vectors are stored, and why this structure makes keyword search fast enough to scan millions of documents — the retriever reads only the index entries for the query words, never touching documents that share no words with the query. The lab implements BM25 scoring but not the index data structure.

---

### lab03 — Semantic Search with Embeddings
**File:** `lab03_semantic_search.ipynb`

A full semantic search system: embed documents once at index time, then embed each query at
search time and rank by cosine similarity.

**What you learn:**
- Index time vs query time: when each embedding is computed and why it matters
- Why you never re-embed documents at search time
- Where semantic search wins over BM25 (synonyms, paraphrases, conceptual queries)
- Where BM25 wins over semantic search (exact technical terms, product codes, IDs)

**What you build:**
- An indexed embedding matrix saved to disk with `joblib`
- `semantic_search()` — a complete retriever using pre-computed embeddings
- A side-by-side comparison of BM25 vs semantic search on the same queries
- `semantic_rag()` — the full RAG pipeline using semantic retrieval

**Key insight:** "Tell me about the waterway that connects the Red Sea and the Mediterranean" retrieves the Suez Canal document with no shared words. BM25 cannot do this. Semantic search can.

**Covered by the reference doc but not this lab:**
The doc's Section 5.2 covers **model incompatibility** as a production risk: if you update the embedding model, every document embedding in the knowledge base must be regenerated from scratch. Documents embedded with the old model and queries embedded with the new model live in incompatible coordinate systems and produce meaningless similarity scores. The lab does not address this operational constraint.

The doc also references **approximate nearest neighbor (ANN) search** (HNSW, FAISS) as the production mechanism for large-scale semantic search. The lab uses exact search over a small corpus, which does not scale beyond tens of thousands of documents.

---

### lab04 — Hybrid Search and Reciprocal Rank Fusion
**File:** `lab04_hybrid_search.ipynb`

Hybrid search is the production standard. This lab combines BM25 and semantic search using
Reciprocal Rank Fusion — the algorithm used by Weaviate, Pinecone, and most enterprise search systems.

**What you learn:**
- Why you cannot average BM25 and cosine scores directly (incompatible scales)
- Reciprocal Rank Fusion (RRF): uses rank position, not raw score
- The RRF formula: `1 / (K + rank)`
- The `K` hyperparameter: how it controls top-rank dominance vs spread
- The `beta` parameter: how to weight keyword vs semantic contribution

**What you build:**
- `reciprocal_rank_fusion()` — merges any number of ranked lists
- `hybrid_search()` — the complete BM25 + semantic + RRF pipeline
- A plot showing how K changes the shape of rank contributions

**Key insight:** A document ranked 1st by BM25 and 5th by semantic search scores higher in RRF than one ranked 2nd by both. Position is what matters, not the raw score.

**Covered by the reference doc but not this lab:**
The doc's Section 2.2 explains a subtlety the lab does not address: **why both searches must run in parallel over the full knowledge base, not sequentially**. If semantic search ran only over the documents that keyword search already returned, it would only re-rank those documents — never finding documents that keyword search missed entirely. That defeats the purpose of hybrid retrieval.

The doc's Section 6.3 includes a **worked RRF table** with five documents, showing exactly how the final ranking is computed rank-by-rank with real numbers — useful to study alongside the lab's implementation.

The doc's Section 6.4 gives **domain-specific guidance on beta**: legal and medical coding applications need higher keyword weight (50/50 or more) because of exact citation matching requirements; general conversational chatbots benefit from higher semantic weight (70/30 or more). The lab exposes the beta parameter but does not explain when to move it.

---

### lab05 — Retrieval Evaluation
**File:** `lab05_retrieval_evaluation.ipynb`

You cannot improve what you cannot measure. This lab gives you four metrics to quantitatively
evaluate any retriever, applied to the real 20 Newsgroups dataset.

**What you learn:**
- Precision@K: what fraction of the top-K results are relevant (trustworthiness)
- Recall@K: what fraction of all relevant documents did we find (comprehensiveness)
- MAP@K: does the retriever rank relevant documents near the top
- MRR: how soon does the first relevant document appear
- The precision-recall tradeoff: as K increases, recall rises and precision falls

**What you build:**
- All four metrics from scratch as Python functions
- An evaluation harness that runs 10 test queries against the full 20 Newsgroups corpus
- A precision-recall tradeoff plot comparing K=5, K=20, K=50

**Covered by the reference doc but not this lab:**
The doc's Section 7.1 covers the **three ingredients every metric requires** — the prompt, the ranked retrieved list, and the ground truth set — and explains why the ground truth is the most labor-intensive and most commonly shortchanged part. Automated heuristics for building ground truth (e.g. using documents that were clicked after a search) produce metrics that do not predict real-world performance.

The doc's Section 7.5 explains **how to use the four metrics together as a feedback loop**: when Recall@K is low, fix search coverage; when MAP@K is low relative to recall, fix the RRF weighting; when MRR is low, add a reranker. The lab teaches you to compute the metrics; the doc teaches you to act on them.

The doc's Section 7.3 includes a **worked MAP@K example** with five relevant documents at specific rank positions, computing precision at each rank step by step — useful to study alongside the lab's implementation.

---

## Module 02 — Topics in the reference document not covered by any lab

**Metadata filtering (Section 3) — significant gap.** No lab covers this topic at all. The doc covers it in full: what metadata is, how filtering works (SQL-like boolean WHERE conditions), the two most important production use cases (access control by tier, and regional/audience targeting), and the hard limitations (cannot rank, only include/exclude; inflexible date cutoffs; depends entirely on tagging quality). Metadata filtering is the third pillar of the hybrid pipeline alongside keyword and semantic search, and it is the only mechanism that enforces hard access control guarantees that no similarity score can override.

**The full hybrid pipeline architecture (Section 2):** The doc covers the five-stage pipeline — parallel search, metadata filter, rank fusion, top-K selection, output — with the design rationale for each stage ordering. The labs implement individual components but not the integrated five-stage pipeline with the rationale for why each stage sits where it does.

**Why parallel search, not sequential (Section 2.2):** The doc explicitly explains why running semantic search only over keyword search results destroys the value of semantic search. The lab's `hybrid_search()` does run parallel search correctly but does not explain why sequential would be wrong.

**Why filter after search, not before (Section 2.3):** The design rationale — clean separation of content relevance from access control, preserving ranking order into the fusion step — is not covered in any lab.

**The librarian analogy (Section 1.2):** A mental model of an expert research librarian simultaneously applying subject index lookup, semantic familiarity with the collection, and access rules — then synthesizing a prioritized reading list. Useful for interviews and for explaining the system to non-technical stakeholders.

**Foundational papers (Section 10):** Robertson & Zaragoza 2009 (BM25 — the source of k1 and b), Karpukhin et al. 2020 (Dense Passage Retrieval / DPR — established the dense retrieval paradigm), Lewis et al. 2020 (RAG), Gao et al. 2024 (RAG survey), Cormack et al. 2009 (RRF — the theoretical justification for K=60), Johnson et al. 2021 (FAISS — ANN search at scale), Reimers & Gurevych 2019 (Sentence-BERT — established the sentence embedding training approach).

**The 22 interview Q&A (Section 9):** vocabulary mismatch, inverted indexes, embedding geometry, contrastive training, cosine vs Euclidean, precision vs recall in RAG context, metadata filtering limits, full hybrid pipeline, RRF mechanics, tuning with evaluation metrics, top-K decision for a chatbot, building domain-specific retrievers (legal, medical), and interpreting unusual metric combinations (e.g. high MAP with low MRR).

---

## Lab dependency map

```
Module 01
lab01 Python Essentials
    └── lab02 Environment & utils.py
            └── lab03 LLM Single Calls
                    └── lab04 Conversations & System Prompts
                            └── lab05 Augmented Prompts  ← the full RAG pipeline
                                    └── lab06 Manual Retrieval → motivates Module 02

Module 02
lab01 Vector Embeddings ──────────────────────────────────────────┐
lab02 Keyword Search (TF-IDF, BM25) ─────────────────────────────┤
    └── lab03 Semantic Search                                      │
            └── lab04 Hybrid Search + RRF ← combines 02 and 03   │
                    └── lab05 Retrieval Evaluation ───────────────┘
```

---

## Coverage summary

| Topic | Labs | Reference doc |
|---|---|---|
| Python patterns for RAG | ✅ M01 lab01 | ✅ Section 3.4 pseudocode |
| LLM calls — single and multi-turn | ✅ M01 labs 02–04 | ✅ Sections 4, 5 |
| Temperature (observable) | ✅ observed | ✅ explained mechanically Section 4.4 |
| Tokenization / BPE | ❌ | ✅ Section 4.1 |
| Autoregressive generation in depth | ❌ | ✅ Sections 4.2–4.3 |
| Full RAG pipeline end to end | ✅ M01 lab05 | ✅ Sections 2–3 |
| Grounding vs hallucination (demo) | ✅ demonstrated | ✅ explained mechanically |
| Intrinsic vs extrinsic hallucination | ❌ | ✅ Section 5.1 |
| In-context learning (named) | ❌ | ✅ Section 2.3 |
| Five structural advantages of RAG | ❌ | ✅ Section 3.3 |
| RAG vs fine-tuning vs long-context | ❌ | ✅ Section 7 |
| RAG applications by domain | ❌ | ✅ Section 6 |
| Agentic RAG / Self-RAG | ❌ | ✅ Section 8 |
| Naive retrieval failure modes | ✅ M01 lab06 | ✅ Module 02 Section 1 |
| Cosine vs Euclidean (implemented) | ✅ M02 lab01 | ✅ curse of dimensionality explained |
| Dot product vs cosine — when they differ | ❌ | ✅ Section 5.1 |
| Contrastive training | ❌ | ✅ Section 5.2 |
| Model incompatibility risk | ❌ | ✅ Section 5.2 |
| TF-IDF from scratch | ✅ M02 lab02 | ✅ worked IDF example Section 4.1 |
| BM25 from scratch | ✅ M02 lab02 | ✅ full formula Section 4.2 |
| Inverted index structure | ❌ | ✅ Section 4.1 |
| Semantic search with pre-computed embeddings | ✅ M02 lab03 | ✅ Section 5.1 |
| ANN search at scale (HNSW, FAISS) | ❌ | ✅ referenced Section 5.1 |
| Metadata filtering | ❌ | ✅ Section 3 (full coverage) |
| Full hybrid pipeline architecture | ✅ implemented | ✅ design rationale Section 2 |
| Why parallel search (not sequential) | ❌ explained | ✅ Section 2.2 |
| Why filter after search (not before) | ❌ | ✅ Section 2.3 |
| RRF from scratch | ✅ M02 lab04 | ✅ worked table Section 6.3 |
| Beta weighting by domain | ✅ parameter | ✅ domain guidance Section 6.4 |
| Precision@K, Recall@K | ✅ M02 lab05 | ✅ Section 7.2 |
| MAP@K | ✅ M02 lab05 | ✅ worked example Section 7.3 |
| MRR | ✅ M02 lab05 | ✅ Section 7.4 |
| Using metrics as a feedback loop | ❌ | ✅ Section 7.5 |
| Ground truth construction | ❌ | ✅ Section 7.1 |
| 44 interview Q&A (22 per module) | ❌ | ✅ both modules Sections 9/10 |
| Foundational papers | ❌ | ✅ both modules Sections 10/11 |
