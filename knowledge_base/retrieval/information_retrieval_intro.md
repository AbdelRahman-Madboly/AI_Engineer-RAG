# Introduction to Information Retrieval

> **Topic Area:** Retrieval  
> **Covers:** What a retriever does, how it finds relevant documents, ranking, indexing, and the trade-offs involved

---

## The Retriever's Job in One Sentence

Given a user's question, find the documents in the knowledge base that are most likely to help the LLM answer it correctly.

That sentence is simple. The implementation is not — and the quality of the retriever has an enormous impact on the quality of the final RAG system.

---

## The Library Analogy

Imagine you walk into a large library and ask:

*"How can I make New York-style pizza at home?"*

The librarian (the retriever) does several things:
1. Understands what you're really asking about — cooking, specifically Italian-American cuisine, specifically pizza technique
2. Knows how the library is organized (the index)
3. Walks to the right shelves — cooking, Italian food, maybe New York food culture
4. Pulls books that seem most relevant
5. Hands them to you

The retriever in a RAG system does all of these things — just at computer speed, over potentially millions of documents.

**Components of the analogy:**
| Library | RAG Retriever |
|---------|--------------|
| Collection of books | Knowledge base of documents |
| Library catalog / Dewey Decimal | Document index |
| Librarian's understanding of your question | Query processing |
| Selecting the relevant shelf | Similarity search |
| Handing you the books | Returning top-N documents |

---

## The Index — How Documents Are Organized for Fast Search

A retriever cannot scan every document in the knowledge base for every query — that would be impossibly slow at scale.

Instead, the knowledge base is **indexed** ahead of time. The index is a data structure that allows the retriever to quickly find relevant documents without reading every one.

Think of the index like the index at the back of a textbook. Instead of reading the whole book to find where "neural network" is mentioned, you look it up in the index and go directly to the right pages.

In a vector database (the most common production retriever), the index stores mathematical representations of each document's meaning — called **embeddings** — in a way that makes it very fast to find the documents with the most similar meaning to a query.

---

## Similarity Scoring — How Relevance Is Measured

The core problem of retrieval is: **given a query, which documents are most relevant?**

The retriever answers this by calculating a **similarity score** between the query and each document in the knowledge base. The documents with the highest scores are returned.

There are different ways to calculate similarity:

### Keyword / Lexical Search (BM25)
Counts how many words from the query appear in the document, weighted by how rare or common those words are. Fast and simple. Works well when exact words matter.

**Limitation:** If the query says "car" and the document says "automobile," keyword search may miss the match entirely.

### Semantic / Vector Search
Converts both the query and documents into dense numerical vectors (embeddings) that capture *meaning*, not just words. Finds documents that are conceptually similar even when different words are used.

**Limitation:** Can sometimes miss exact matches that keyword search would catch easily.

### Hybrid Search
Combines both approaches — uses keyword search for exact matches and semantic search for conceptual matches. Often gives the best results in practice.

You will learn to implement all three of these in Module 2.

---

## The Precision-Recall Trade-off

The retriever faces a fundamental tension that is important to understand:

**Retrieve too few documents:**
- You return only the most confidently relevant documents
- Risk: you miss some relevant documents that were ranked slightly lower
- The LLM gets less information

**Retrieve too many documents:**
- You capture more potentially relevant documents
- Risk: you include a lot of irrelevant documents too
- The LLM's context window fills up with noise
- Prompts become longer, slower, and more expensive

In information retrieval, this is called the **precision-recall trade-off**:
- **Precision** — of the documents returned, what fraction are actually relevant?
- **Recall** — of all the relevant documents that exist, what fraction did you return?

There is no perfect setting. Tuning the retriever means finding the right balance for your specific use case. This is why monitoring and evaluation (covered in Module 5) are essential parts of building a production RAG system.

---

## What Good Retrieval Looks Like

A well-designed retriever:

✅ Returns documents that are genuinely relevant to the query  
✅ Ranks more relevant documents higher than less relevant ones  
✅ Does not return irrelevant documents that would confuse the LLM  
✅ Returns an appropriate number of documents — enough for coverage, not so many that context is wasted  
✅ Works quickly — retrieval latency adds to the user's wait time  

A poorly designed retriever:

❌ Returns documents that look superficially similar but answer a different question  
❌ Misses highly relevant documents because they used different words  
❌ Returns too many documents, overloading the context window  
❌ Returns too few, leaving the LLM without what it needs  

---

## Familiar Systems That Are Essentially Retrievers

The retriever is not a new idea. You interact with retrieval systems all the time:

- **Web search engine** — you provide a query; it returns relevant web pages
- **SQL database query** — you provide criteria; it returns matching rows
- **Email search** — you search for a word; it returns matching emails
- **Document search in Word/Google Docs** — find all files mentioning a phrase

RAG systems use these same principles, adapted for use with language models and often implemented using vector databases for better semantic understanding.

The field of **information retrieval** predates LLMs by decades. The techniques developed in that field — how to index documents, how to rank results, how to handle relevance — are the foundations on which modern RAG retrievers are built.

---

## Why Vector Databases?

Traditional relational databases (SQL) are excellent at finding exact matches. They are not designed for *semantic* similarity search — finding documents that *mean* the same thing as a query even when they use different words.

Vector databases are specialized for this. They store documents as vectors (embeddings) and use algorithms optimized for finding the nearest vectors in high-dimensional space — which corresponds to finding semantically similar documents.

At scale, this makes vector databases the standard choice for the retrieval component of a production RAG system. You will build with Weaviate, one of the leading vector databases, in Module 3.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Retriever's job | Find the most relevant documents for a given query |
| Index | Pre-built data structure that makes search fast at scale |
| Similarity scoring | Quantifies relevance between query and each document |
| Keyword search | Matches exact words; fast; misses synonyms |
| Semantic search | Matches meaning; finds synonyms; slower |
| Hybrid search | Combines both; best of both worlds |
| Precision-recall trade-off | Returning fewer documents = higher precision; more = higher recall |
| Vector database | Specialized database for semantic similarity search |

---

## Related Topics

- [RAG Architecture](../architecture/rag_architecture.md) — Where the retriever fits in the full system
- [What Is RAG](../rag/what_is_rag.md) — Why retrieval is needed in the first place
- [How LLMs Work](../llms/how_llms_work.md) — What the LLM does with the retrieved documents
