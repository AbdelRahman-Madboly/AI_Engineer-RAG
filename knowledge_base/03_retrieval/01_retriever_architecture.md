# Retriever Architecture — The Hybrid Pipeline That Powers RAG Search

> **Series:** AI_Engineer-RAG Knowledge Base  
> **Module:** 2 — Information Retrieval and Search Foundations  
> **Folder:** `knowledge_base/03_retrieval/`  
> **File:** `01_retriever_architecture.md`  
> **Academic depth:** [`03_Dense_Retrieval_and_Vector_Databases.md`](../../resources/03_Dense_Retrieval_and_Vector_Databases.md) §1, §9

---

## The Core Idea

A retriever receives a prompt and must return, as fast as possible, the documents from a knowledge base most likely to help an LLM answer it. No single search technique is adequate for all inputs, so modern retrievers run keyword search and semantic search in parallel, apply metadata filtering to both result sets, and merge the two ranked lists into a final ordering using Reciprocal Rank Fusion.

## The Problem It Solves

Users do not submit structured queries — they write naturally, using synonyms, implied context, and domain jargon. The documents they need may not share a single word with their prompt, or they may share many words but be completely off-topic. Meanwhile, some documents must be excluded for access-control or audience-relevance reasons regardless of their topical similarity.

No single technique handles all three of these problems simultaneously. Keyword search handles exact vocabulary matches but is blind to meaning. Semantic search handles meaning but struggles with precise terminology. Metadata filtering enforces hard exclusions neither search technique can provide. A production retriever must orchestrate all three.

## The Retriever's Place in the RAG Loop

Before examining the internal pipeline, it helps to place the retriever in context. A RAG system follows three steps — Retrieve, Augment, Generate. The retriever owns the first step entirely. It receives the raw user prompt, consults the knowledge base, and hands a ranked set of documents to the augmentation layer, which inserts them into the LLM's context window. The LLM never sees the full knowledge base — only what the retriever selects. This makes retrieval quality the primary determinant of answer quality. A retriever that misses a relevant document gives the LLM no way to recover.

Think of the retriever as a research librarian working in a large archive. When a patron asks a question, the librarian does not read every document in the building — that would take too long. Instead, they use several complementary search strategies simultaneously: scanning an index of known keywords, drawing on familiarity with related concepts, and applying access rules about which parts of the archive the patron is allowed to see. The librarian then synthesizes these results into a short, relevant reading list. The retriever does exactly this, but at machine speed and with explicit algorithms for each strategy.

## The Three Techniques and Their Roles

Each technique in the hybrid pipeline is assigned to the problem it handles best.

**Keyword search** scans documents for the exact words present in the prompt. It is fast, interpretable, and indispensable when users employ technical terminology, product names, regulation numbers, or proper nouns that must appear verbatim in the retrieved documents. A query for "ISO 27001 section 6.1.2" will only be useful if the retrieved documents actually contain that string — semantic approximation is not acceptable.

**Semantic search** represents both the prompt and each document as a point in high-dimensional vector space, where geometrically nearby points share similar meaning. It captures synonyms, paraphrases, and conceptually related content that keyword search cannot reach. A prompt asking about "memory consumption in neural networks" can retrieve a document discussing "GPU RAM in deep learning models" even though no words overlap.

**Metadata filtering** applies boolean, SQL-like criteria to document-level attributes — author, date, section, access level, region — and removes documents that fail the criteria. It does not rank; it strictly excludes. This provides a hard boundary that neither search technique can enforce. A semantic similarity score cannot reliably model "this document is restricted to paid subscribers." A boolean filter can.

## The Full Pipeline

```
User Prompt
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                     RETRIEVER                       │
│                                                     │
│   ┌──────────────────┐   ┌────────────────────┐    │
│   │  Keyword Search  │   │  Semantic Search   │    │
│   │  (BM25 / TF-IDF) │   │  (Embedding Model) │    │
│   └────────┬─────────┘   └──────────┬─────────┘    │
│            │                        │               │
│            ▼                        ▼               │
│   [Ranked List A, ~50 docs]  [Ranked List B, ~50]   │
│            │                        │               │
│            ▼                        ▼               │
│   ┌──────────────────────────────────────────┐     │
│   │         Metadata Filter                  │     │
│   │  (access level, region, date, section)   │     │
│   └──────────────┬───────────────────────────┘     │
│                  │                                  │
│       [Filtered List A]  [Filtered List B]          │
│                  │                                  │
│                  ▼                                  │
│   ┌──────────────────────────────────────────┐     │
│   │      Reciprocal Rank Fusion (RRF)         │     │
│   │  Combines ranks from both filtered lists  │     │
│   └──────────────┬───────────────────────────┘     │
│                  │                                  │
│            [Final Ranking]                          │
│                  │                                  │
│                  ▼                                  │
│           Top-K Documents                           │
└─────────────────────────────────────────────────────┘
     │
     ▼
Augmented Prompt  →  LLM
```

## Why Parallel Search, Not Sequential

The two searches run in parallel rather than in sequence, because running semantic search only on documents that survived keyword search would discard documents that are semantically relevant but lexically different — precisely the class semantic search is designed to find. Running both independently and merging the results preserves the complementary coverage of each technique. Documents that rank highly on both lists receive compounded scores in the fusion step; documents that only one technique finds still have a path to the top of the final ranking.

## Why Fusion After Filtering, Not Before

Metadata filtering is applied after each search but before the lists are merged. This ordering matters. Filtering after search ensures that the ranking quality produced by each technique is intact when the lists enter Reciprocal Rank Fusion — filtering does not affect how documents were scored. Filtering before search would require the filter to interact with the search algorithm's data structures, which varies by implementation. Filtering before fusion but after search is the clean, modular design that modern retriever implementations use.

## The Design Insight

The architecture is not arbitrary — each component is placed where it has decisive advantage. Keyword search excels at lexical precision; semantic search at conceptual generalization; metadata filtering at rigid rule enforcement; Reciprocal Rank Fusion at merging heterogeneous ranked lists without requiring score normalization. The complete pipeline is greater than the sum of its parts because each technique's weaknesses are covered by the others.

---

## Key Terms

| Term | Definition |
|---|---|
| Retriever | The component of a RAG system that searches a knowledge base and returns ranked documents for a given prompt. |
| Knowledge base | The collection of documents available to the retriever; can range from a few hundred to millions of text chunks. |
| Keyword search | A search technique that scores documents by the frequency and rarity of exact word matches with the prompt. |
| Semantic search | A search technique that scores documents by the geometric proximity of their embeddings to the prompt's embedding in vector space. |
| Metadata filtering | Rigid boolean exclusion of documents based on document-level attributes, applied after search and before rank fusion. |
| Reciprocal Rank Fusion (RRF) | An algorithm that merges multiple ranked lists by assigning each document a score based on its rank position in each list. |
| Top-K | The number of documents the retriever ultimately returns to the augmentation layer. |
| Hybrid search | A retrieval strategy that combines keyword search, semantic search, and metadata filtering into a single pipeline. |
| Sparse vector | A vector with one dimension per vocabulary word, mostly zeros, used to represent documents in keyword search. |
| Dense vector (embedding) | A compact, high-dimensional vector output by an embedding model that encodes the semantic meaning of a text. |

---

## What to Carry Forward

- The retriever is the single most consequential component in a RAG system — if a relevant document is not retrieved, the LLM cannot recover.
- Modern retrievers are hybrid: keyword search, semantic search, and metadata filtering each handle a distinct failure mode that the others cannot address.
- The pipeline runs keyword and semantic search in parallel, applies metadata filters to both result lists, then fuses them with Reciprocal Rank Fusion before returning the top-K documents.
- Each technique is assigned to what it does best: keyword for lexical precision, semantic for conceptual generalization, metadata for rigid rule enforcement, RRF for rank merging.
- Designing a high-performing retriever is fundamentally a problem of understanding these complementary strengths and tuning the balance between them for the specific knowledge base and query distribution.

---

## Navigation

- **Previous:** [`knowledge_base/01_rag_fundamentals/03_rag_architecture.md`](../01_rag_fundamentals/03_rag_architecture.md)
- **Next:** [`02_metadata_filtering.md`](02_metadata_filtering.md)
- **Related:** [`../01_rag_fundamentals/01_what_is_rag.md`](../01_rag_fundamentals/01_what_is_rag.md) — the broader RAG loop this retriever sits within
- **Academic depth:** [`03_Dense_Retrieval_and_Vector_Databases.md`](../../resources/03_Dense_Retrieval_and_Vector_Databases.md) §1 (Why Keyword Retrieval Is Not Enough), §9 (Hybrid Retrieval)