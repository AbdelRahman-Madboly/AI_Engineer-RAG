# Keyword Search — From Bag of Words to BM25

> **Series:** AI_Engineer-RAG Knowledge Base  
> **Module:** 2 — Information Retrieval and Search Foundations  
> **Folder:** `knowledge_base/03_retrieval/`  
> **File:** `03_keyword_search.md`  
> **Academic depth:** [`03_Dense_Retrieval_and_Vector_Databases.md`](../../resources/03_Dense_Retrieval_and_Vector_Databases.md) §2

---

## The Core Idea

Keyword search retrieves documents by measuring the overlap between the words in a prompt and the words in each document. Two refinements — weighting rare words more heavily (IDF) and applying diminishing returns to repeated terms (BM25 saturation) — turn this simple intuition into a scoring system that has powered search engines and information retrieval for decades.

## The Problem It Solves

The most direct evidence that a document is relevant to a query is that it contains the query's words. For technical documents, product catalogs, legal texts, and regulated domains, this lexical match is not merely convenient — it is necessary. A user querying for "RFC 7231" or "compound sodium bicarbonate" needs documents that contain those exact strings. No amount of semantic similarity can substitute for the actual term appearing in the document. Keyword search is the technique optimized for this requirement.

## From Words to Vectors: The Bag of Words Representation

Keyword search begins by discarding word order entirely. Both the prompt and each document are treated as a **bag of words** — a multiset that records which words appear and how often, with no attention to sequence. The phrase "the pizza was on the pizza stone" becomes: {the: 2, pizza: 2, was: 1, on: 1, stone: 1}.

These word counts are stored in a **sparse vector** — a vector with one dimension for every word in the system's vocabulary, which may be tens of thousands of words. In practice, most dimensions hold zero, since any given document uses only a small fraction of the full vocabulary. The sparsity is what gives these vectors their name and their computational efficiency: sparse data structures only store the non-zero entries.

Collecting all document sparse vectors into a single grid produces the **inverted index** (also called a term-document matrix). Rows correspond to words; columns correspond to documents; cells store word counts. The index is called "inverted" because the natural direction of lookup is reversed — instead of "given this document, what words does it contain?", the index enables "given this word, which documents contain it?" This inversion is what makes keyword retrieval fast: locating every document containing the word "pizza" requires only reading the pizza row, not scanning all document columns.

```
Vocabulary     Doc 1    Doc 2    Doc 3    Doc 4
─────────────  ───────  ───────  ───────  ───────
pizza             2        0        1        0
oven              1        0        0        1
stone             1        0        0        0
recipe            0        3        1        0
flour             0        2        0        0
trombone          0        0        0        2
...             ...      ...      ...      ...
```

Building this index is a one-time offline cost. At query time, only the rows corresponding to query terms need to be read.

## Scoring: From Raw Counts to TF-IDF

The simplest scoring rule is to award a document one point for each query word it contains. This works, but it has two important flaws.

First, it does not account for how many times a document contains each word — a document mentioning "pizza" twenty times should presumably score higher on a pizza query than one that mentions it once. Replacing the binary check with raw term counts fixes this. However, raw counts then introduce a second flaw: longer documents will naturally contain more occurrences of any word simply by virtue of their length. A document with ten times as many words as another will tend to score higher on almost any query, regardless of topical relevance.

Dividing a document's raw count by its total word length — producing a **normalized term frequency (TF)** — levels this playing field. A document that devotes 5% of its words to "pizza" now scores higher than one that devotes 1%, regardless of their absolute lengths.

Term frequency alone, however, still treats all words equally. The word "the" might appear in 99% of documents; "epigenomics" in 0.1%. A query term that appears in almost every document provides almost no discriminating signal — its presence tells you nothing about whether a document is relevant. A term that appears in very few documents is highly discriminating; its presence is strong evidence of relevance.

**Inverse Document Frequency (IDF)** captures this. For a term t in a corpus of N documents, where df(t) documents contain t:

```
IDF(t) = log( N / df(t) )
```

A term appearing in every document receives IDF ≈ 0. A term appearing in only 1 of 1,000 documents receives IDF = log(1000) ≈ 6.9. The log scale prevents rare terms from becoming astronomically dominant.

Multiplying TF by IDF produces the **TF-IDF score** for a term in a document:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

The final relevance score for a document against a query is the sum of TF-IDF values across all query terms that appear in the document. Documents that frequently use rare query terms score highest.

### Worked Example

Suppose a knowledge base of 1,000 documents. A user submits the query: "making pizza without a pizza oven."

| Query term | df (docs containing it) | IDF = log(1000/df) | Doc A TF (5 uses / 200 words) | Doc A TF-IDF |
|---|---|---|---|---|
| making     | 800 | 0.22 | 5/200 = 0.025 | 0.006 |
| pizza      | 50  | 3.00 | 10/200 = 0.05 | 0.150 |
| without    | 700 | 0.36 | 2/200 = 0.01  | 0.004 |
| a          | 999 | 0.00 | —             | 0.000 |
| oven       | 30  | 3.51 | 4/200 = 0.02  | 0.070 |

**Doc A total score: 0.006 + 0.150 + 0.004 + 0.000 + 0.070 = 0.230**

The common words "making," "without," and "a" contribute almost nothing. "Pizza" and "oven" — rare and specific — drive the score. A document about baking trombones that happens to use "making" frequently would score nearly zero, correctly.

## BM25: Two Improvements That Matter in Practice

TF-IDF remains conceptually foundational, but the algorithm used in virtually all modern keyword retrieval systems — including Elasticsearch, OpenSearch, and Lucene — is **BM25** (Best Matching 25, Robertson & Zaragoza, 2009). BM25 makes two targeted improvements.

**Improvement 1: Term Frequency Saturation.** In TF-IDF, doubling the number of times a term appears doubles its contribution to the score. BM25 applies a saturation function: each additional occurrence of a term contributes less than the previous one, with returns diminishing as counts grow large. Intuitively, a document mentioning "pizza" 30 times is not 30× more relevant than one mentioning it once — at some point, additional occurrences stop adding meaningful evidence. The hyperparameter **k1** (typically 1.2–2.0) controls how quickly saturation sets in. Higher k1 means slower saturation; lower k1 means each additional term occurrence matters less.

**Improvement 2: Document Length Normalization with Diminishing Penalties.** TF-IDF normalizes by document length, but does so in a way that can penalize long documents too aggressively. BM25 applies length normalization more gently: longer documents are still penalized relative to the corpus average, but the penalty diminishes rather than scaling linearly. The hyperparameter **b** (typically 0.75) controls the strength of length normalization. Setting b = 0 disables length normalization entirely; b = 1 applies full normalization.

The complete BM25 score for a query term t in document d:

```
BM25(t, d) = IDF(t) × [TF(t,d) × (k1 + 1)] / [TF(t,d) + k1 × (1 - b + b × |d|/avgdl)]
```

where |d| is the document length and avgdl is the average document length across the corpus. The full document score sums this across all query terms.

The k1 and b hyperparameters allow BM25 to be tuned to the specific statistical properties of a knowledge base — a key practical advantage over TF-IDF.

## Strengths and the Vocabulary Mismatch Problem

Keyword search has three compounding strengths. It is fast — the inverted index means retrieval touches only the rows for query terms, not every document. It is interpretable — a document's score can be explained in terms of exactly which words matched and how. It is reliable for exact terminology — when the query and the relevant documents share specific vocabulary, keyword search will find them.

Its fundamental weakness is **vocabulary mismatch**: keyword search has no mechanism for recognizing that two different strings carry the same meaning. "Memory consumption" and "RAM usage" are synonyms; a keyword search for one will not find documents that use only the other. Acronyms, abbreviations, paraphrases, and cross-lingual queries all represent the same problem. For natural language queries where users phrase their intent without knowing the exact vocabulary of the documents, keyword search fails silently — it returns low scores without indicating that relevant documents exist using different words. This limitation motivates semantic search as a necessary complement.

---

## Key Terms

| Term | Definition |
|---|---|
| Bag of words | A text representation that records word counts while discarding word order. |
| Sparse vector | A vector with one dimension per vocabulary word, mostly zeros, storing word frequency counts for a text. |
| Inverted index | A data structure mapping each vocabulary word to the list of documents containing it; enables fast keyword lookup across a corpus. |
| Term Frequency (TF) | A word's frequency within a single document, normalized by document length. |
| Inverse Document Frequency (IDF) | A log-scaled measure of how rare a word is across the full corpus; rare words receive higher IDF. |
| TF-IDF | A scoring function combining term frequency and inverse document frequency to weight relevant, rare terms highly. |
| BM25 | Best Matching 25; a keyword scoring algorithm that adds TF saturation and diminishing length normalization penalties to TF-IDF. |
| Term frequency saturation | The property of BM25 whereby additional occurrences of a term contribute diminishing marginal increases to a document's relevance score. |
| Document length normalization | Adjusting relevance scores to account for document length, so longer documents are not systematically favored for raw term counts. |
| k1 (BM25) | BM25 hyperparameter controlling how quickly term frequency saturates; typical range 1.2–2.0. |
| b (BM25) | BM25 hyperparameter controlling the strength of document length normalization; typical value 0.75. |
| Vocabulary mismatch | The failure mode of keyword search in which a relevant document uses different words than the query, producing a near-zero score despite semantic relevance. |

---

## What to Carry Forward

- Keyword search treats texts as bags of words, ignoring order, and stores word counts in sparse vectors organized into an inverted index that makes retrieval fast.
- TF-IDF scores documents by multiplying within-document term frequency by corpus-wide inverse document frequency, ensuring that rare, specific terms drive scores more than common ones.
- BM25 improves TF-IDF with two targeted fixes — term frequency saturation (diminishing returns per repeated term) and gentler document length normalization — and two tunable hyperparameters k1 and b.
- Keyword search excels at exact terminology, technical terms, proper nouns, and product names; it is fast and interpretable.
- Its critical limitation is vocabulary mismatch: it cannot connect a query to a document that expresses the same idea in different words, which is why semantic search is a necessary complement.

---

## Navigation

- **Previous:** [`02_metadata_filtering.md`](02_metadata_filtering.md)
- **Next:** [`04_semantic_search.md`](04_semantic_search.md)
- **Related:** [`05_hybrid_search.md`](05_hybrid_search.md) — how keyword search integrates with semantic search via RRF
- **Academic depth:** [`03_Dense_Retrieval_and_Vector_Databases.md`](../../resources/03_Dense_Retrieval_and_Vector_Databases.md) §2 (Classical Information Retrieval: TF-IDF and BM25)