# RAG Architecture — How the Components Work Together

> **Topic Area:** Architecture  
> **Covers:** The full RAG pipeline, how each component connects, advantages over plain LLM usage, and a first look at code structure

---

## From a User's Perspective: Nothing Changes

One of the elegant things about a RAG system is that the user experience looks identical to using a plain LLM. You type a question. You get an answer.

What changes is everything that happens in between.

---

## The RAG Pipeline Step by Step

```
User types a prompt
        │
        ▼
┌───────────────┐
│   RETRIEVER   │ ← searches the knowledge base
│               │   returns top N relevant documents
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────┐
│         PROMPT AUGMENTATION       │
│                                   │
│  Original prompt                  │
│  + Retrieved document 1           │
│  + Retrieved document 2           │
│  + Retrieved document N           │
└───────────────┬───────────────────┘
                │
                ▼
        ┌───────────────┐
        │      LLM      │ ← generates response
        └───────┬───────┘
                │
                ▼
        Response to user
```

### Step 1 — The prompt arrives

The user submits a question or request. The RAG system receives it.

### Step 2 — The retriever searches the knowledge base

The retriever processes the prompt to understand its meaning, then searches the knowledge base — a database of trusted documents. It ranks documents by relevance and returns the top matches.

### Step 3 — The prompt is augmented

The system builds a new, expanded prompt that includes:
- The user's original question
- The retrieved documents (or relevant excerpts from them)
- Instructions telling the LLM how to use the provided context

A simple augmented prompt looks like this:

```
Answer the following question using the provided context.
If the answer is not in the context, say so.

Context:
[retrieved document 1 text]
[retrieved document 2 text]

Question: Why are hotel prices in Vancouver so high this weekend?
```

### Step 4 — The LLM generates a grounded response

The LLM receives the augmented prompt. It now has both:
- Its own general knowledge from training
- The specific, relevant information from the knowledge base

It generates a response that incorporates both. The retrieved context grounds the response — the LLM is less likely to hallucinate because it has accurate information right in front of it.

---

## The Three Core Components in Detail

### The Knowledge Base

The knowledge base is simply a collection of documents. In a production system, this might be:
- Thousands of product documentation pages
- A company's entire internal wiki
- All the articles ever published by a news organization
- A patient's full medical history

What makes it a *knowledge* base rather than just a database is that it is curated, trusted, and relevant to the domain the RAG system is serving.

In most production RAG systems, the knowledge base is stored in a **vector database** — a specialized database optimized for finding documents that are semantically similar to a query. You will learn about vector databases in depth in Module 3.

### The Retriever

The retriever is the bridge between the user's question and the knowledge base. Its job is deceptively simple to describe but technically complex to do well:

*Given a query, find the most relevant documents in the knowledge base.*

To do this, the retriever:
1. Processes the query to understand its meaning
2. Searches the index of the knowledge base
3. Scores and ranks documents by relevance
4. Returns the top-ranked documents

There are multiple approaches to building a retriever — keyword search, semantic search, hybrid approaches — which you will explore in Module 2.

The retriever faces a fundamental tension: return too few documents and you might miss relevant information; return too many and you waste the LLM's context window and increase cost and latency.

### The Large Language Model

The LLM in a RAG system has a focused role: **read the augmented prompt and generate a high-quality response.**

It is not expected to do fact-finding (the retriever handles that). It is not expected to know private or recent information (the knowledge base handles that). Its job is reasoning, synthesis, and writing.

This division of labor is one of the architectural strengths of RAG: each component does what it is best at.

---

## The Advantages This Architecture Provides

### 1. Access to information outside training data
The most fundamental advantage. Private data, recent events, specialized domains — all become available to the LLM through the knowledge base.

### 2. Reduced hallucinations
Hallucinations are most common when an LLM is asked about topics it wasn't well-trained on. By providing relevant information directly in the prompt, the LLM has less reason to fabricate.

### 3. Easy knowledge updates
Updating an LLM's knowledge through retraining is expensive and slow — it can take weeks and cost millions of dollars. Updating a knowledge base takes minutes. New documents go in; the system immediately benefits.

### 4. Source citation
Because you know exactly which documents were retrieved, you can include that information in the augmented prompt and the LLM can produce citations in its response. This is critical for trust and verifiability.

### 5. Scalability and modularity
Each component — retriever, knowledge base, LLM — can be upgraded independently. Swap in a better LLM without touching the retriever. Improve the retrieval algorithm without changing the LLM. This modularity makes RAG systems maintainable.

---

## A Minimal Code Structure

Even at this early stage, it helps to see how these components map to code. The structure is simple:

```python
def retrieve(query: str) -> list[str]:
    """
    Search the knowledge base.
    Returns a list of relevant document strings.
    """
    # ... retriever logic ...
    return relevant_documents

def generate(prompt: str) -> str:
    """
    Call the LLM with a prompt.
    Returns the LLM's response.
    """
    # ... LLM API call ...
    return response

def rag_pipeline(user_query: str) -> str:
    # Step 1: Retrieve relevant documents
    documents = retrieve(user_query)
    
    # Step 2: Build the augmented prompt
    context = "\n".join(documents)
    augmented_prompt = f"""
    Answer the following question using the provided context.
    
    Context:
    {context}
    
    Question: {user_query}
    """
    
    # Step 3: Generate the response
    response = generate(augmented_prompt)
    return response
```

This is the skeleton of every RAG system. The sophistication lives in the implementation of `retrieve()` and `generate()` — and in how the augmented prompt is constructed.

---

## What Can Go Wrong

Understanding the failure modes of RAG architecture is as important as understanding how it works.

| Failure Mode | What Happens | Root Cause |
|---|---|---|
| **Retrieval miss** | The right documents exist but aren't retrieved | Poor retriever, bad indexing, wrong query processing |
| **Irrelevant retrieval** | Wrong documents are retrieved and confuse the LLM | Retriever ranks irrelevant documents too high |
| **Context window overflow** | Too much retrieved text; LLM can't process all of it | Retrieving too many or too long documents |
| **LLM ignores context** | Relevant documents are retrieved but LLM doesn't use them | Poor prompt construction, model limitations |
| **Stale knowledge base** | Documents in the knowledge base are outdated | Poor knowledge base maintenance |

Recognizing these failure modes is the first step toward building systems that avoid them.

---

## Summary

| Component | Input | Output | Lives In |
|-----------|-------|--------|---------|
| Knowledge Base | — | Documents to retrieve from | Database (often vector DB) |
| Retriever | User query | Top-N relevant documents | Search system |
| Prompt Augmenter | Query + documents | Augmented prompt | Application code |
| LLM | Augmented prompt | Grounded response | Model API |

The user sees only the input (their question) and the output (the response). Everything in between is the RAG pipeline.

---

## Related Topics

- [What Is RAG](../rag/what_is_rag.md) — The motivation and core idea
- [How LLMs Work](../llms/how_llms_work.md) — Why the LLM can use information it wasn't trained on
- [Introduction to Information Retrieval](../retrieval/information_retrieval_intro.md) — How the retriever finds relevant documents
