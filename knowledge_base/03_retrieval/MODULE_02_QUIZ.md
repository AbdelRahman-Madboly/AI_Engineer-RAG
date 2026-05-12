# MODULE 02 QUIZ — Information Retrieval and Search Foundations

> **Series:** AI_Engineer-RAG Knowledge Base  
> **Module:** 2 — Information Retrieval and Search Foundations  
> **File:** `MODULE_02_QUIZ.md`

---

## Module Coverage

This quiz tests understanding of all six files in `knowledge_base/03_retrieval/`:

- [`01_retriever_architecture.md`](01_retriever_architecture.md) — The hybrid pipeline
- [`02_metadata_filtering.md`](02_metadata_filtering.md) — Rigid criterion-based exclusion
- [`03_keyword_search.md`](03_keyword_search.md) — Bag of words, TF-IDF, BM25
- [`04_semantic_search.md`](04_semantic_search.md) — Embeddings, vector space, contrastive training
- [`05_hybrid_search.md`](05_hybrid_search.md) — Reciprocal Rank Fusion and the combined pipeline
- [`06_retrieval_evaluation.md`](06_retrieval_evaluation.md) — Precision, Recall, MAP, MRR

---

## Part 1 — Conceptual Questions (7)

**C1.** Explain what an inverted index is and why its structure is described as "inverted." What problem does this data structure solve, and what is it inverted relative to?

**C2.** A colleague claims that IDF alone — without TF — is sufficient for keyword scoring. Construct an argument for why TF is also necessary, and describe a concrete example where IDF-only scoring would produce a worse ranking than TF-IDF.

**C3.** Explain what it means for an embedding model to use "contrastive training." What are positive pairs and negative pairs, and what does the model do with them during training? Why does this process cause semantically similar texts to end up geometrically close?

**C4.** Semantic search finds nearby vectors. But what does "nearby" actually mean — and why does cosine similarity dominate over Euclidean distance as the measure used in practice for text embeddings?

**C5.** Describe what the K hyperparameter in Reciprocal Rank Fusion controls. What happens to the final ranking when K = 0? What changes when K = 60? Give an intuition for when you would prefer a smaller versus a larger K.

**C6.** Explain the difference between precision and recall as retrieval metrics. Why is recall generally considered the more foundational metric in a RAG context, even though precision also matters?

**C7.** Why must all documents and queries in a retrieval system be embedded with the same embedding model? What would happen if you embedded documents with one model and queries with another, and why?

---

## Part 2 — Architecture and Design Questions (4)

**A1.** You are building a RAG system for a large legal firm. The knowledge base contains thousands of case documents. Some documents are restricted to partners only; others are available to all staff. Users will ask natural language questions, but queries sometimes include specific case numbers, statute citations, and exact legal phrases. Design a retrieval architecture for this system. Specify which components you would include, in what order, and what parameters you would pay particular attention to tuning.

**A2.** Your retriever pipeline currently runs keyword search and semantic search sequentially: it runs keyword search first, takes the top 20 results, and then runs semantic search only on those 20. A colleague suggests running both searches independently in parallel over the full knowledge base. Explain the flaw in the sequential approach and why the parallel design is preferable.

**A3.** A production RAG system has the following evaluation results after testing on 50 labeled queries: Recall@10 = 0.65, Precision@10 = 0.45, MAP@10 = 0.38, MRR = 0.72. Interpret these numbers. Which metric suggests the biggest problem? What does the relatively high MRR combined with low MAP tell you about where in the ranked list the retriever's failures are occurring?

**A4.** You are tasked with re-indexing a knowledge base of 2 million documents after switching from one embedding model to a newer, higher-quality model. A colleague suggests you only re-embed documents added or modified in the last six months to save time. Explain precisely why this approach would break the retrieval system, even for queries that only return old documents.

---

## Part 3 — Comparative Questions (3)

**Comp1.** Compare TF-IDF and BM25. Both are keyword scoring algorithms — what does BM25 add, and why do those additions matter in practice? Under what conditions might TF-IDF and BM25 produce noticeably different rankings for the same query?

**Comp2.** Keyword search and semantic search are often described as having "complementary failure modes." Explain what this phrase means concretely. Give one example query and one example knowledge base document for which keyword search would succeed but semantic search might fail, and a second example where the reverse is true.

**Comp3.** Compare Precision@K and MAP@K as evaluation metrics. They are both precision-based, so what does MAP add that Precision@K misses? Describe a specific scenario — a concrete ranked list — where Precision@10 is the same for two retrievers but MAP@10 reveals that one is substantially better.

---

## Part 4 — Application Questions (2)

**App1.** You are building a retriever for a global e-commerce support knowledge base. The knowledge base contains product manuals (requiring exact model number matching), shipping FAQs (general natural language questions), and return policy documents (specific policy language varies by region). Users contact support with a wide mix of query types: "how do I set up model XR-7000?", "when will my order arrive?", and "what is the return window in France?".

Describe how you would configure the hybrid retriever for this use case. Specifically: would you adjust the beta weighting for keyword versus semantic search? Would you use metadata filtering? What evaluation strategy would you use to tune the system?

**App2.** After deploying a hybrid retriever, you notice that your MAP@10 is 0.71 but your Recall@10 is only 0.58 — well below the 0.85 target. Your team proposes two fixes: (a) increase K from 10 to 20, or (b) increase the beta weight toward semantic search from 0.70 to 0.85. Walk through what each change would likely do to Recall@10, Precision@10, and MAP@10, and recommend which change to try first and why.

---

## Answer Key

### C1. The Inverted Index

A standard document index starts from a document and enumerates the words it contains — the natural direction for reading. An inverted index reverses this: it starts from each word and lists the documents containing it. The data structure is a term-document matrix where rows are vocabulary words and columns are documents, with cells holding word counts or weighted scores.

The problem it solves is search speed. At query time, finding every document containing the word "pizza" requires reading only a single row of the index, not scanning every document in the knowledge base. Without the inversion, retrieval would require loading and scanning each document individually — an O(N) scan where N is the number of documents. With the inverted index, keyword lookup is essentially constant-time per query term.

---

### C2. Why TF Is Necessary Alongside IDF

IDF alone weights each query term by its rarity across the corpus but ignores how prominently that term appears in any specific document. Consider two documents, both about pizza: Doc A mentions "pizza" once in a 500-word document about Italian cuisine broadly; Doc B mentions "pizza" 20 times in a 300-word recipe specifically for pizza dough. Both would receive the same IDF contribution for "pizza." But Doc B is clearly more about pizza. TF captures this — it records that "pizza" constitutes a much higher proportion of Doc B's vocabulary, correctly scoring it higher. IDF alone would treat Doc A and Doc B identically on the "pizza" dimension and fail to distinguish between a document that incidentally mentions a term and one whose central topic is that term.

---

### C3. Contrastive Training

An embedding model is trained with pairs of texts labeled as either similar (positive pairs) or dissimilar (negative pairs). During training, the model is evaluated on how well its current parameter settings place positive pairs close together and negative pairs far apart in vector space. Based on how it performs, its parameters are updated to improve — pulling positive pairs toward each other and pushing negative pairs apart. This evaluation-update cycle is called contrastive because the model learns by contrast between similar and dissimilar examples.

After many thousands of update steps across millions of pairs, the parameter updates accumulate into a consistent mapping: texts that appear repeatedly as positive examples in training data end up near each other geometrically, and texts that appear as negatives for the same anchors end up far apart. Semantic proximity in the resulting vector space is a consequence of statistical co-occurrence in training pairs — the model has learned that certain kinds of texts belong together because they were labeled as belonging together.

---

### C4. Why Cosine Similarity Over Euclidean Distance

"Nearby" in semantic search means geometrically close in vector space. Two metrics for measuring closeness: Euclidean distance (straight-line separation) and cosine similarity (angle between vectors, ignoring magnitude).

Cosine similarity dominates for text because of two properties. First, texts of different lengths produce embeddings of different magnitudes — a long document's embedding often has a larger norm than a short sentence's. Euclidean distance conflates magnitude with direction, so a semantically identical short and long text would register as far apart. Cosine similarity normalizes away magnitude, measuring only directional alignment — the direction encodes meaning, not the length of the vector. Second, in high-dimensional space, the "curse of dimensionality" causes Euclidean distances to concentrate: most pairs of points end up roughly equidistant, making fine-grained distance discrimination difficult. Cosine similarity remains more discriminative in high dimensions because angles are geometrically meaningful even when absolute distances converge.

---

### C5. The K Hyperparameter in RRF

K is a smoothing constant in the denominator of the RRF formula: `score = 1/(K + rank)`. It controls how dominant the top-ranked document in any single list is relative to lower-ranked documents.

At K = 0: the top-ranked document scores 1/1 = 1.0 and the tenth-ranked scores 1/10 = 0.1 — a 10× gap. A single strong ranking in one list almost guarantees a high combined score. At K = 60: the top-ranked document scores 1/61 ≈ 0.0164 and the tenth-ranked scores 1/70 ≈ 0.0143 — a 1.14× gap. Top rank still matters, but not by much; consistent performance across both lists now drives the final ranking.

Prefer smaller K when one search technique is expected to reliably identify the best document (e.g., when keyword search and semantic search are both high quality and you trust their top results). Prefer larger K to balance influence and prevent a single strong keyword match from overriding semantic evidence — the standard value of 60 reflects this intent.

---

### C6. Recall vs. Precision: Why Recall Is Foundational

Precision measures what fraction of retrieved documents are relevant — it penalizes noise in the prompt. Recall measures what fraction of all relevant documents were retrieved — it penalizes missed coverage.

In RAG, recall is foundational because of the asymmetry of failure. A missed relevant document (recall failure) means the LLM cannot answer correctly; the information is simply absent. No amount of good prompting, reasoning, or generation quality can recover from a retrieval miss. An irrelevant document (precision failure) adds noise but does not make correct answering impossible — the LLM may still find the relevant document among the retrieved set and generate a correct answer despite the noise. Recall failures are thus a harder constraint than precision failures, making recall the metric to protect first when making system tradeoffs.

---

### C7. Same Embedding Model Requirement

Each embedding model defines its own vector space — its own coordinate system for encoding semantic meaning. The coordinates of a given text depend on the model's architecture, training data, initialization, and all parameter updates during training. Two different models will produce vectors for the same text that are different numbers, assigned to locations in fundamentally different spaces.

Comparing a vector from Model A to a vector from Model B is like comparing GPS coordinates in one geographic datum system to coordinates in a different datum: the numbers may look similar, but they refer to different locations. The similarity scores produced by such cross-model comparisons are meaningless — a high cosine similarity between Model A's "pizza" vector and Model B's "trombone" vector would simply reflect accidental alignment, not semantic relationship. A retriever built this way would produce random, uninterpretable results.

---

### A1. Legal Firm RAG Architecture

The architecture requires all three retrieval components. Metadata filtering should enforce the partner/all-staff access control rule — this is a hard yes/no constraint that no similarity-based technique can model. Each document should be tagged with its access tier, and a filter derived from the requesting user's role should be applied after both searches.

Both keyword and semantic search are necessary. Case numbers and statute citations ("Section 402(a) of the Securities Act") must match exactly — a semantic approximation is legally meaningless. BM25 with tuned k1 provides this. Natural language questions about case outcomes, legal strategies, and precedent analysis benefit from semantic matching. The hybrid RRF combination covers both.

Beta weighting should start at roughly 50% keyword / 50% semantic — higher keyword weight than the typical default, given the frequency of exact-match queries in legal research. Tuning should be driven by a labeled evaluation set sampled from both exact-match queries (case numbers, statute references) and conceptual queries (case outcome analysis, precedent research).

---

### A2. Sequential vs. Parallel Search

The sequential approach runs semantic search only on the 20 documents that survived keyword search. This defeats the purpose of semantic search. The entire value of semantic search lies in its ability to find documents that keyword search misses — documents with similar meaning but different vocabulary. If a document is not in the keyword search top 20, it is excluded from the semantic search entirely, even if it is the most semantically relevant document in the knowledge base. The sequential design is equivalent to running only keyword search with a 20-document cap, then re-ranking those 20 — semantic search adds almost no additional coverage.

The parallel design runs both searches over the full knowledge base independently. Documents not found by keyword search can still appear in the semantic search results, and vice versa. RRF then combines the two independent ranked lists. This is the correct hybrid design — each technique contributes independent evidence, and their complementary coverage is preserved.

---

### A3. Interpreting the Metrics

Recall@10 = 0.65 means the retriever is missing 35% of relevant documents in its top 10 — a significant coverage failure. Precision@10 = 0.45 means 55% of returned documents are irrelevant — significant noise, but secondary. MAP@10 = 0.38 is low, indicating that even the relevant documents that are found tend to appear late in the ranked list. MRR = 0.72 is relatively strong, suggesting the first relevant document typically appears near rank 1–2.

The combination of high MRR and low MAP tells a specific story: the retriever is reliably placing at least one relevant document near the top of the list, but the remaining relevant documents are poorly positioned — they appear late, after many irrelevant documents. The failures are concentrated in the mid-to-lower portions of the ranked list, not at the very top. The priority is improving coverage (Recall@10) and mid-list ranking (MAP@10) without degrading the top-rank placement that MRR reflects.

---

### A4. Partial Re-indexing Breaks Retrieval

Embedding models define vector spaces. All documents in a knowledge base must be embedded in the same vector space for cosine similarity comparisons to be meaningful. If half the documents are embedded with Model A (the old model) and half with Model B (the new model), the two halves exist in incompatible coordinate systems. A query vector from Model B will correctly identify nearby documents in Model B's space (recently embedded documents) but will produce random, meaningless distances when compared to Model A's vectors (old documents). The old documents will appear to be randomly near or far from queries with no relationship to actual semantic content. This breaks retrieval for old documents even when only new queries are submitted — the distance calculations are simply nonsense across model boundaries. Complete re-indexing is required whenever the embedding model changes.

---

### Comp1. TF-IDF vs. BM25

Both are keyword scoring algorithms that combine term frequency with inverse document frequency. BM25 adds two improvements: term frequency saturation (each additional occurrence of a term contributes diminishing marginal score, controlled by k1) and gentler document length normalization (long documents are penalized less aggressively than in TF-IDF, controlled by b).

The additions matter most in corpora with uneven document lengths and repetitive term usage. In a knowledge base mixing short summaries with lengthy reports, TF-IDF over-penalizes the longer documents and fails to distinguish between a document that is topically focused on a term (high density, appropriate to reward) versus one that just uses the term throughout because it is long. BM25's saturation and softer normalization handle both cases better. For short, uniform documents, TF-IDF and BM25 produce similar rankings. For diverse corpora, BM25 consistently outperforms TF-IDF, which is why it has displaced TF-IDF in all modern search systems.

---

### Comp2. Complementary Failure Modes

Keyword search fails when the query and relevant document share meaning but not vocabulary. Semantic search fails when specific exact terminology is required.

Example where keyword succeeds, semantic might fail: query = "ISO 27001 clause 6.1.2 risk treatment". The relevant document contains this exact string. Semantic search might rank documents about "information security risk management frameworks" nearby — similar meaning, but none of them contain the exact clause reference. Keyword search finds the exact string immediately.

Example where semantic succeeds, keyword fails: query = "my computer runs slowly". The relevant document says "optimizing CPU performance and memory allocation in Windows." Zero words overlap with the query. Keyword search scores this document near zero. Semantic search correctly identifies that "runs slowly" and "CPU performance" are related concepts and ranks it highly.

---

### Comp3. Precision@K vs. MAP@K

Precision@K measures whether the right documents appear anywhere in the top K. MAP@K measures whether they appear early in the top K. Consider two retrievers, both evaluated on a 10-document list where 5 documents are relevant:

- Retriever X returns relevant documents at ranks 1, 2, 3, 4, 5. Precision@10 = 5/10 = 0.50. AP@10 = (1.0 + 1.0 + 1.0 + 1.0 + 1.0)/5 = 1.0.
- Retriever Y returns relevant documents at ranks 6, 7, 8, 9, 10. Precision@10 = 5/10 = 0.50. AP@10 = (0.167 + 0.286 + 0.375 + 0.444 + 0.500)/5 = 0.354.

Both retrievers have identical Precision@10. MAP@10 reveals that Retriever X is dramatically better — it places all relevant documents at the top of the list where they will be most useful to the LLM, while Retriever Y buries them at the bottom. Precision@K cannot see this distinction; MAP@K is designed to expose it.

---

### App1. E-Commerce Support Retriever

The three query types map naturally to retrieval components. Product manual queries containing exact model numbers (XR-7000) require strong keyword search — exact product identifiers cannot be semantically approximated. FAQ queries ("when will my order arrive?") are natural language and benefit most from semantic search. Return policy queries with regional variation require metadata filtering on region.

Recommended configuration: run parallel keyword and semantic search over the full knowledge base. Apply a metadata filter for region on the return policy documents. Use RRF to combine results. Beta weighting should be closer to 50%/50% keyword/semantic (weighted toward keyword more than the default 70/30) given the importance of exact model number matching. Tuning requires a labeled evaluation set sampling all three query types proportionally. Measure MAP@K per query type separately to confirm that the weighting serves the product manual queries without degrading the FAQ and policy queries.

---

### App2. Fixing Low Recall

**Option (a): Increase K from 10 to 20.** This mechanically increases Recall@K because the retriever now has 20 chances to include each relevant document. Recall@20 will almost certainly be higher than Recall@10. However, Precision@20 will fall (more documents, same number of relevant ones), and MAP@20 will only improve if the retriever happens to rank the previously-missed relevant documents above the new irrelevant additions — not guaranteed.

**Option (b): Increase beta toward semantic search.** This changes which documents rank highest, potentially surfacing relevant documents that keyword search was depressing. It could improve Recall@10, Precision@10, and MAP@10 simultaneously if the issue is that relevant documents exist in the top 20 but are being ranked outside the top 10 due to keyword-dominated weighting.

**Recommendation: try (b) first.** Increasing K is a blunt instrument that trades precision for recall without improving the retriever's underlying ranking quality. Adjusting beta targets a specific hypothesis — that semantic-matched documents are being systematically underweighted — and is measurable via the evaluation metrics without changing the system's capacity. If beta adjustment alone does not reach the 0.85 recall target, then increasing K is a justified secondary lever.