# Hybrid Search — Combining Retrieval Signals with Reciprocal Rank Fusion

> **Series:** AI_Engineer-RAG Knowledge Base  
> **Module:** 2 — Information Retrieval and Search Foundations  
> **Folder:** `knowledge_base/03_retrieval/`  
> **File:** `05_hybrid_search.md`  
> **Academic depth:** [`03_Dense_Retrieval_and_Vector_Databases.md`](../../resources/03_Dense_Retrieval_and_Vector_Databases.md) §9

---

## The Core Idea

Hybrid search runs keyword search and semantic search in parallel, applies metadata filters to both result sets, then merges the two ranked lists into a single final ranking using Reciprocal Rank Fusion. Because the two search techniques fail in complementary ways, the combined pipeline reliably outperforms either technique used alone.

## The Problem It Solves

No single retrieval technique dominates across all query types. Keyword search misses documents that use different vocabulary for the same concept. Semantic search can mis-rank documents containing specific technical terms when those terms appear in semantically related but actually irrelevant documents. For any real knowledge base that receives diverse user queries, a retriever built on only one technique will have systematic blind spots. Hybrid search eliminates those blind spots by combining signals — each technique contributes evidence, and the combination is more robust than either alone.

## The Hybrid Pipeline

```
User Prompt
    │
    ├─────────────────────┬──────────────────────┐
    ▼                     ▼                      │
Keyword Search       Semantic Search             │
(BM25 over          (Embedding model +           │
inverted index)      vector similarity)          │
    │                     │                      │
    ▼                     ▼                      │
Ranked List A         Ranked List B              │
(~50 docs)            (~50 docs)                 │
    │                     │                      │
    ▼                     ▼                      │
Metadata Filter      Metadata Filter      ← user attributes
    │                     │                      │
    ▼                     ▼                      │
Filtered List A       Filtered List B            │
(~35 docs)            (~30 docs)                 │
    │                     │
    └──────────┬───────────┘
               ▼
    Reciprocal Rank Fusion (RRF)
    weighted by beta (e.g., 70% semantic / 30% keyword)
               │
               ▼
         Final Ranking
               │
               ▼
         Top-K Documents  →  Augmented Prompt
```

## Why Hybrid Outperforms Single-Technique Retrieval

Keyword and semantic search have complementary failure modes, not overlapping ones.

Keyword search fails when the query and relevant documents share meaning but not vocabulary — synonyms, paraphrases, and domain-shift queries. Semantic search fails when precise terminology matters — rare proper nouns, regulation codes, product SKUs, and technical strings where approximate meaning is insufficient. A document about "Section 4.2.1 of GDPR" may embed near other legal-privacy documents, but only keyword search guarantees the document containing the exact string "Section 4.2.1" ranks highest.

Because the failures are non-overlapping, a document that one technique misses is often found by the other. The fusion step synthesizes these independent signals into a ranking that captures both lexical precision and semantic coverage.

## Reciprocal Rank Fusion (RRF)

The central challenge of combining two ranked lists is that the scores from keyword search (BM25 scores) and semantic search (cosine similarities) are on entirely different numerical scales. A BM25 score of 14.3 and a cosine similarity of 0.72 cannot be directly averaged. Normalizing both to [0, 1] and summing them is possible but introduces sensitivity to the score distribution — a score of 0.72 means something different when the range is 0.60–0.73 versus 0.10–0.90.

**Reciprocal Rank Fusion** (Cormack et al., 2009) sidesteps this problem entirely. It ignores scores and operates only on rank positions. A document's contribution to the final score depends solely on where it appeared in each ranked list, not on the score that determined its position.

The RRF score for a document d, given two ranked lists, is:

```
RRF_score(d) = 1/(K + rank_keyword(d)) + 1/(K + rank_semantic(d))
```

Documents that appear in only one list still receive a score from the list where they appear; the missing term simply contributes zero. K is a smoothing hyperparameter.

### K: Controlling Top-Rank Dominance

When K = 0, the top-ranked document in either list receives a score of 1/(0 + 1) = 1.0, and the tenth-ranked document receives 1/(0 + 10) = 0.1. This is a 10× gap — a document ranked first in one list almost automatically dominates the final ranking, regardless of how it performed in the other list.

Setting K = 60 (the standard default, from Cormack et al.) dramatically reduces this gap: the top-ranked document scores 1/61 ≈ 0.0164 and the tenth-ranked document scores 1/70 ≈ 0.0143. The ratio is now 1.14× rather than 10×. This means documents must perform consistently across both lists to reach the top of the final ranking — a single strong performance in one list no longer guarantees a top-combined position.

Smaller K values weight the agreement heavily. Larger K values produce a flatter ranking where rank position matters less and consistency across lists matters more.

### Worked Example

Suppose keyword search returns: [Doc A (rank 1), Doc B (rank 2), Doc C (rank 3)] and semantic search returns: [Doc C (rank 1), Doc A (rank 2), Doc D (rank 3)].

Using K = 60:

| Document | Keyword rank | Semantic rank | RRF score |
|---|---|---|---|
| Doc A | 1 | 2 | 1/61 + 1/62 = 0.0164 + 0.0161 = **0.0325** |
| Doc C | 3 | 1 | 1/63 + 1/61 = 0.0159 + 0.0164 = **0.0323** |
| Doc B | 2 | — (not in list) | 1/62 + 0 = **0.0161** |
| Doc D | — (not in list) | 3 | 0 + 1/63 = **0.0159** |

Final ranking: Doc A → Doc C → Doc B → Doc D. Doc A ranks first because it performs consistently well on both techniques. Doc C, which ranked first semantically, is edged out by Doc A's strong combined performance. Doc B, which only appeared in the keyword list, ranks above Doc D by a hair.

Notice that the original BM25 score for Doc A and the cosine similarity for Doc C were never used — only their rank positions determined the final outcome.

## Beta: Weighting Keyword Versus Semantic

A second hyperparameter, **beta**, adjusts how much the keyword and semantic rankings each contribute to the final score. If beta = 0.7, the semantic ranking contributes 70% of the weight and the keyword ranking contributes 30%:

```
RRF_score(d) = beta × [1/(K + rank_semantic(d))] + (1-beta) × [1/(K + rank_keyword(d))]
```

A starting point of 70% semantic / 30% keyword is a reasonable default for most natural language query workloads, because most users write queries in conversational language that benefits more from semantic matching. However, this default should be tuned empirically.

**Weight keyword search more heavily when:** the queries contain technical identifiers, product names, regulation numbers, or exact strings; the knowledge base is a technical reference or catalog; users are expected to know precise terminology.

**Weight semantic search more heavily when:** users submit natural language questions or conversational queries; the knowledge base contains general-purpose documents; paraphrase and synonym matching are common query patterns.

## Top-K: Deciding How Many Documents to Return

The final parameter in the hybrid pipeline is **top-K** — the number of documents the retriever returns to the augmentation layer. After RRF produces the final ranking, the retriever simply slices off the top K documents.

K is a tradeoff between context quality and context cost. Smaller K means fewer, higher-precision documents in the prompt — more focused answers, lower token costs, lower latency. Larger K increases the chance that a relevant document is included but increases prompt length, cost, and the risk that the LLM's attention is diluted across irrelevant material. Typical values in production RAG systems range from 3 to 10. K should be tuned in conjunction with retrieval evaluation metrics — increasing K improves recall but may reduce precision.

---

## Key Terms

| Term | Definition |
|---|---|
| Hybrid search | A retrieval strategy combining keyword search, semantic search, and metadata filtering, with results merged via Reciprocal Rank Fusion. |
| Reciprocal Rank Fusion (RRF) | A rank combination algorithm that scores documents by the reciprocal of their position in each ranked list, enabling fusion of heterogeneous scoring systems. |
| K (RRF hyperparameter) | Smoothing constant in the RRF formula that controls how dominant the highest-ranked documents are; standard value is 60. |
| Beta | Hyperparameter weighting the relative contribution of semantic versus keyword rankings in the hybrid RRF score. |
| Top-K | The number of documents the retriever returns to the augmented prompt after the full hybrid pipeline completes. |
| Complementary failure modes | The property of keyword and semantic search whereby each technique's blind spots are covered by the other's strengths, motivating their combination. |
| Score normalization | The alternative to RRF for combining ranked lists, involving scaling scores to a common range before summing; more sensitive to score distribution than RRF. |

---

## What to Carry Forward

- Hybrid search runs keyword and semantic search in parallel, applies metadata filters to both result lists, and merges them with Reciprocal Rank Fusion before returning the top-K documents.
- RRF is effective because it ignores raw scores and combines only rank positions, making it robust to the incompatible scales of BM25 and cosine similarity scores.
- The K hyperparameter controls how dominant top-ranked documents are; the standard value of 60 ensures that consistent performance across both lists beats a single strong ranking in one list.
- Beta lets engineers weight keyword versus semantic rankings — 70% semantic / 30% keyword is a sensible default, but should be tuned for the specific knowledge base and query distribution.
- Top-K is a precision-recall tradeoff: more documents increases recall but adds noise to the LLM's context window and increases cost.
- Hybrid retrieval is the recommended default for production RAG; pure single-technique retrieval is rarely optimal across a real-world query distribution.

---

## Navigation

- **Previous:** [`04_semantic_search.md`](04_semantic_search.md)
- **Next:** [`06_retrieval_evaluation.md`](06_retrieval_evaluation.md)
- **Related:** [`01_retriever_architecture.md`](01_retriever_architecture.md) — the full pipeline diagram; [`03_keyword_search.md`](03_keyword_search.md) and [`04_semantic_search.md`](04_semantic_search.md) — the two techniques being combined
- **Academic depth:** [`03_Dense_Retrieval_and_Vector_Databases.md`](../../resources/03_Dense_Retrieval_and_Vector_Databases.md) §9 (Hybrid Retrieval: Combining Sparse and Dense)