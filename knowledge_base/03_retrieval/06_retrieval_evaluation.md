# Retrieval Evaluation — Measuring What the Retriever Actually Finds

> **Series:** AI_Engineer-RAG Knowledge Base  
> **Module:** 2 — Information Retrieval and Search Foundations  
> **Folder:** `knowledge_base/03_retrieval/`  
> **File:** `06_retrieval_evaluation.md`  
> **Academic depth:** [`03_Dense_Retrieval_and_Vector_Databases.md`](../../resources/03_Dense_Retrieval_and_Vector_Databases.md) §12 · [`05_Evaluation_and_Monitoring.md`](../../resources/05_Evaluation_and_Monitoring.md) §3

---

## The Core Idea

Retrieval evaluation measures whether the retriever is finding the right documents by comparing its ranked output to a ground truth set of known-relevant documents. Four metrics — Precision@K, Recall@K, MAP@K, and MRR — each capture a different dimension of retrieval quality, and they are most useful when read together.

## The Problem It Solves

A retriever that has never been formally evaluated is an unknown system. It may perform well on the queries its developers tested informally and fail systematically on the distribution of queries real users submit. Without measurement, there is no principled basis for deciding whether a change — adjusting the beta weighting between keyword and semantic search, increasing K, or switching embedding models — has improved or degraded the system. Retrieval evaluation provides that measurement baseline and the feedback signal needed to guide systematic improvement.

## The Three Ingredients of Any Retrieval Metric

Every retrieval metric is computed from the same three inputs.

**The prompt.** A retriever's performance is always relative to a specific query. A retriever can perform well on informational natural language questions and poorly on short technical queries. Evaluation must sample prompts that represent the actual distribution of user queries the system will receive.

**The ranked retrieved list.** This is the ordered output the retriever produces for the prompt — the sequence of documents ranked by the retriever's combined scoring. Order matters: most metrics are sensitive to whether relevant documents appear near the top of the list or buried in the lower ranks.

**The ground truth set.** This is the set of documents in the knowledge base that are actually relevant to the prompt, determined by human annotation. It is the "correct answer" the retriever is being measured against. Ground truth construction is time-consuming but irreplaceable — automated labeling heuristics can produce a starting point, but human judgment is the gold standard for what actually constitutes a relevant document for a given query.

## Precision@K and Recall@K

**Precision@K** answers: of the K documents the retriever returned, what fraction were actually relevant?

```
Precision@K = (number of relevant documents in top K) / K
```

Precision measures trustworthiness. A high Precision@K means the retriever is not cluttering the prompt with irrelevant material. Irrelevant documents in the prompt are not merely wasted context — they can actively mislead the LLM, causing it to generate answers that are grounded in the retrieved text but not in the facts relevant to the query.

**Recall@K** answers: of all the relevant documents in the knowledge base, what fraction did the retriever find in its top K?

```
Recall@K = (number of relevant documents in top K) / (total relevant documents in knowledge base)
```

Recall measures comprehensiveness. A Recall@K of 1.0 means the retriever found every relevant document in its top K — nothing was missed. In a RAG system, recall is the more foundational metric: if a relevant document is not retrieved, the LLM has no path to using it. A missed document cannot be recovered downstream.

**The precision-recall tradeoff.** Increasing K mechanically improves recall — by returning more documents, the retriever has more chances to include every relevant one. But precision falls as K increases, because the additional documents tend to be less relevant. This tradeoff is not a failure of any specific technique; it is an inherent property of ranked retrieval. The question is where to set K given the tradeoff, which depends on whether context pollution (low precision) or missed coverage (low recall) is the more expensive failure for your application.

### Worked Example

Knowledge base: 100 documents. Relevant documents for query Q: {Doc 2, Doc 5, Doc 7, Doc 11, Doc 14} — five relevant documents total.

The retriever returns these 10 documents in ranked order:

```
Rank  Document   Relevant?
1     Doc 2      ✓
2     Doc 9      ✗
3     Doc 41     ✗
4     Doc 5      ✓
5     Doc 7      ✓
6     Doc 33     ✗
7     Doc 18     ✗
8     Doc 11     ✓
9     Doc 22     ✗
10    Doc 14     ✓
```

**Precision@5** = 3 relevant in top 5 / 5 = **0.60**  
**Precision@10** = 5 relevant in top 10 / 10 = **0.50**  
**Recall@5** = 3 relevant found / 5 total relevant = **0.60**  
**Recall@10** = 5 relevant found / 5 total relevant = **1.00**

At K=5, the retriever finds 60% of relevant documents with 60% precision. At K=10, it finds all relevant documents (perfect recall) but precision drops to 50%, meaning half the documents in the prompt are noise.

## Mean Average Precision (MAP@K)

Precision@K and Recall@K measure whether the right documents were retrieved, but not whether they were ranked well. A retriever that places all five relevant documents at the bottom of a 10-document list scores identically on Precision@10 to one that places all five at the top — but for the LLM, the position of documents in the prompt matters. Documents at the top of a ranked list tend to receive more attention.

**Mean Average Precision (MAP@K)** rewards retrievers that place relevant documents high in the ranking, not just include them somewhere in the top K.

To compute it, first calculate **Average Precision at K (AP@K)** for a single query. For each relevant document in the top K, compute the precision at the rank where it appears, then average those precision values across all relevant documents found in the top K:

```
AP@K = (1 / |relevant in top K|) × Σ Precision@rank_i
         for each relevant document i in the top K
```

MAP@K is then the mean AP@K across all queries in the evaluation set.

### Worked Example (continued)

Using the same ranked list, computing AP@10:

| Rank | Doc   | Relevant? | Precision@rank (if relevant) |
|------|-------|-----------|------------------------------|
| 1    | Doc 2 | ✓         | 1/1 = 1.000                  |
| 2    | Doc 9 | ✗         | —                            |
| 3    | Doc 41| ✗         | —                            |
| 4    | Doc 5 | ✓         | 2/4 = 0.500                  |
| 5    | Doc 7 | ✓         | 3/5 = 0.600                  |
| 6    | Doc 33| ✗         | —                            |
| 7    | Doc 18| ✗         | —                            |
| 8    | Doc 11| ✓         | 4/8 = 0.500                  |
| 9    | Doc 22| ✗         | —                            |
| 10   | Doc 14| ✓         | 5/10 = 0.500                 |

AP@10 = (1.000 + 0.500 + 0.600 + 0.500 + 0.500) / 5 = **0.620**

A retriever that ranked all five relevant documents in positions 1–5 would produce an AP@10 of (1.0 + 1.0 + 1.0 + 1.0 + 1.0)/5 = 1.0. The difference between 0.620 and 1.0 captures the cost of the irrelevant documents that appeared at ranks 2 and 3, pushing relevant documents further down.

MAP@K averages AP@K across all test queries. High MAP@K indicates not just that relevant documents are found, but that they are placed where they are most useful.

## Mean Reciprocal Rank (MRR)

MAP@K cares about the positions of all relevant documents. Sometimes a narrower question is more important: where does the first relevant document appear? If a user asks a factual question and the answer is contained in a single document, what matters most is that this document appears near the top of the ranked list — specifically, that it appears before the LLM's attention trails off.

**Reciprocal Rank (RR)** for a single query is simply 1 divided by the rank position of the first relevant document:

```
RR = 1 / rank_of_first_relevant_document
```

If the first relevant document is at rank 1: RR = 1.0. At rank 2: RR = 0.5. At rank 5: RR = 0.2. If no relevant document appears in the returned list: RR = 0.

**Mean Reciprocal Rank (MRR)** averages RR across all test queries.

### Worked Example (continued)

In the ranked list above, the first relevant document (Doc 2) appears at rank 1. So RR = 1/1 = **1.0** for this query.

Suppose four test queries produce first-relevant ranks of 1, 3, 6, and 2:

| Query | First relevant rank | Reciprocal Rank |
|-------|---------------------|-----------------|
| Q1    | 1                   | 1.000           |
| Q2    | 3                   | 0.333           |
| Q3    | 6                   | 0.167           |
| Q4    | 2                   | 0.500           |

**MRR** = (1.000 + 0.333 + 0.167 + 0.500) / 4 = **0.500**

An MRR of 0.5 tells us that the first relevant document typically appears around rank 2 on average. A system targeting high MRR specifically optimizes for getting one relevant document to the very top of the list.

## Using the Metrics Together

The four metrics are complementary, not substitutable.

**Recall@K is foundational.** It captures the retriever's core responsibility: finding relevant documents. If recall is low, the LLM cannot answer correctly regardless of how well the rest of the system works. Improving recall typically means increasing K, improving the embedding model, or rebalancing the hybrid search weights.

**Precision@K and MAP@K add ranking quality.** A retriever can have high recall but low precision (it finds everything but returns a lot of noise). MAP@K goes further by rewarding not just whether relevant documents were included but whether they appeared early. Improving MAP typically means improving the RRF ranking rather than the search coverage.

**MRR adds top-of-list sensitivity.** For single-answer factual queries, placing one document at rank 1 can matter more than comprehensive coverage. MRR measures precisely this property.

A complete evaluation process looks like: measure Recall@K first to confirm coverage, then examine MAP@K to assess ranking quality, then use MRR if the application is particularly sensitive to having one relevant document at the very top.

## The Ground Truth Problem

All four metrics require a ground truth set of relevant documents per query. Constructing this dataset is manual and time-consuming. For each test query, a human annotator must review the knowledge base and label which documents are actually relevant. Typically, this involves: defining a relevance judgment rubric, selecting a representative sample of queries from the expected user distribution, and having domain experts annotate relevance for each query-document pair.

The ground truth problem is not a reason to skip evaluation. It is the irreducible cost of rigorous system development. A retrieval evaluation dataset of even a few hundred well-labeled query-document pairs enables principled iteration: change the system, measure the metrics, confirm or reject the change. Without it, improvement is guesswork.

---

## Key Terms

| Term | Definition |
|---|---|
| Ground truth | The set of documents known to be relevant to a query, established by human annotation; the reference used to compute retrieval metrics. |
| Precision@K | The fraction of the top-K retrieved documents that are relevant; measures how much noise enters the prompt. |
| Recall@K | The fraction of all relevant documents in the knowledge base that appear in the top-K retrieved results; measures retrieval completeness. |
| Precision-recall tradeoff | The inverse relationship between precision and recall as K increases: more documents improves recall but typically reduces precision. |
| Average Precision at K (AP@K) | For a single query, the mean of precision scores computed at each rank position where a relevant document appears, among the top K. |
| Mean Average Precision (MAP@K) | The mean AP@K across all test queries; rewards retrievers that rank relevant documents highly, not merely include them. |
| Reciprocal Rank (RR) | For a single query, 1 divided by the rank position of the first relevant document retrieved. |
| Mean Reciprocal Rank (MRR) | The mean reciprocal rank across all test queries; measures how quickly the first relevant document appears in ranked results. |
| Evaluation depth (K) | The cutoff in the ranked list at which metrics are computed; Precision@5 and Recall@10 are examples at different depths. |

---

## What to Carry Forward

- Every retrieval metric requires three inputs: the prompt, the ranked retrieved list, and a human-annotated ground truth set.
- Recall@K is the most fundamental metric — a document that is not retrieved cannot help the LLM, and low recall is the most severe retrieval failure.
- Precision@K measures how much irrelevant material enters the prompt; increasing K improves recall but typically degrades precision.
- MAP@K extends precision by penalizing retrievers that find relevant documents but rank them poorly — it rewards both finding and placing.
- MRR measures specifically how quickly the first relevant document appears, making it the right metric when a single correct document at the top of the list is the primary goal.
- Metrics are the feedback loop for system improvement: change a parameter (beta weighting, K, embedding model), re-evaluate, and use the metric change to accept or reject the modification.
- Building the ground truth annotation dataset is expensive but non-negotiable; approximating it with unlabeled data produces unreliable metrics.

---

## Navigation

- **Previous:** [`05_hybrid_search.md`](05_hybrid_search.md)
- **Next:** [`MODULE_02_QUIZ.md`](MODULE_02_QUIZ.md)
- **Related:** [`01_retriever_architecture.md`](01_retriever_architecture.md) — the pipeline these metrics evaluate; [`05_hybrid_search.md`](05_hybrid_search.md) — how metrics guide beta and K tuning
- **Academic depth:** [`03_Dense_Retrieval_and_Vector_Databases.md`](../../resources/03_Dense_Retrieval_and_Vector_Databases.md) §12 · [`05_Evaluation_and_Monitoring.md`](../../resources/05_Evaluation_and_Monitoring.md) §3