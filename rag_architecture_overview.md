# RAG Architecture Overview

> Understanding how the components of a RAG system connect is essential before building one. This file covers the full data flow — from user question to final response — and explains why each component exists.

---

## The User Experience Is Identical

This is important: from the user's perspective, interacting with a RAG system looks exactly like interacting with a regular LLM. You type a question. You get an answer.

All the additional complexity happens *inside the system*, invisibly.

---

## Full System Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                     RAG SYSTEM                          │
│                                                         │
│  User Prompt                                            │
│      │                                                  │
│      ▼                                                  │
│  ┌───────────┐      ┌──────────────────┐                │
│  │ RETRIEVER │─────▶│  KNOWLEDGE BASE  │                │
│  │           │◀─────│  (Document DB)   │                │
│  └───────────┘      └──────────────────┘                │
│       │                                                 │
│       │  Retrieved Documents                            │
│       ▼                                                 │
│  ┌──────────────────────────┐                           │
│  │    PROMPT AUGMENTATION   │                           │
│  │  Original Question +     │                           │
│  │  Retrieved Documents     │                           │
│  └──────────────────────────┘                           │
│       │                                                 │
│       │  Augmented Prompt                               │
│       ▼                                                 │
│  ┌───────────┐                                          │
│  │    LLM    │                                          │
│  └───────────┘                                          │
│       │                                                 │
│       ▼                                                 │
│   Final Response                                        │
└─────────────────────────────────────────────────────────┘
```

---

## The Three Core Components

### Component 1: The Knowledge Base

The knowledge base is a collection of documents — the trusted source of information your system uses. Think of it as a curated library.

**What can live in a knowledge base:**
- Company policy documents (PDF, Word)
- Product catalogs and FAQs
- News articles or research papers
- Code files and documentation
- Legal case files or medical records
- Web pages

**Key properties:**
- It is separate from the LLM — it can be updated at any time
- It is indexed so the retriever can search it efficiently
- It can be private — the LLM never stores this data, it only reads it

**Practical implementation:** In most production systems, the knowledge base is a **vector database** — a specialized database optimized for finding documents that are semantically similar to a query. You will go deep on this in Module 3.

---

### Component 2: The Retriever

The retriever is the search engine of the RAG system. Given the user's question, it searches the knowledge base and returns the most relevant documents.

**What the retriever does, step by step:**
1. Receives the user's prompt
2. Processes it to understand its meaning
3. Searches the indexed knowledge base
4. Scores each document by relevance
5. Returns the top-ranked documents

**The retriever faces a real tradeoff:**

| Return too many documents | Return too few documents |
|---|---|
| Prompt gets too long | Miss relevant information |
| Increases cost | LLM lacks context to answer well |
| May confuse the LLM | |

Finding the right number is a tuning problem — something you monitor and adjust over time.

**Types of retrieval (covered in depth in later modules):**
- **Keyword search (BM25)** — matches exact words
- **Semantic search** — matches meaning, not just words
- **Hybrid** — combines both for better results

---

### Component 3: The LLM

The LLM is the reasoning and generation engine. It receives the augmented prompt and produces the final response.

**What the LLM contributes:**
- Language understanding and generation
- Reasoning over the provided documents
- Writing quality and coherence
- Synthesizing multiple sources into a single answer

**What the LLM does NOT do in RAG:**
- It does not search the knowledge base — the retriever does that
- It does not store new information permanently — it just reads what is in the prompt
- It does not fact-check itself — the retriever provides grounding

This division of labor is intentional. Each component does what it is best at.

---

## The Augmented Prompt — What It Looks Like

The augmented prompt is the bridge between the retriever and the LLM. In practice it looks something like this:

```
Answer the following question using the provided information.

QUESTION:
Why are hotels in Vancouver so expensive this weekend?

RETRIEVED INFORMATION:
[Document 1] Vancouver Event Calendar - May 2025:
Taylor Swift Eras Tour: May 9-10 at BC Place Stadium.
Expected attendance: 55,000 per night. Hotels within
5km have seen 300% price increases...

[Document 2] Hotel Pricing Analysis - Pacific Northwest:
Event-driven demand spikes typically cause price increases
of 200-400% in surrounding areas...

Answer based on the above information:
```

The LLM reads this and produces a grounded, specific, accurate answer — because it has the right information in front of it.

---

## Why RAG Beats Alternatives for This Problem

### vs. Just prompting the LLM

Without retrieval, the LLM only knows what was in its training data. For private, recent, or specialized information — it will hallucinate or simply not know.

### vs. Retraining the model

Retraining is expensive, slow, and static. RAG is cheap, fast, and dynamic. Update a document in the knowledge base and the system immediately uses the new information.

### vs. Stuffing everything in the context

You could theoretically put all your documents directly in every prompt. But context windows have limits, longer prompts cost more, and it is slow. RAG retrieves only what is *relevant* — keeping prompts lean and targeted.

---

## Advantages RAG Provides

| Advantage | How it works |
|---|---|
| **Grounded answers** | Retrieved documents anchor the LLM to real information |
| **Reduced hallucinations** | LLM is less likely to invent facts when real ones are provided |
| **Source citations** | The system knows exactly which documents were used |
| **Live updates** | Update the knowledge base without touching the model |
| **Scalability** | The LLM focuses on generation; the retriever handles search |
| **Division of labor** | Each component focuses on its strength |

---

## Simple Code Shape of a RAG System

This is the essential skeleton — what RAG looks like in code at its simplest:

```python
def rag_pipeline(user_question):
    # Step 1: Retrieve
    relevant_docs = retriever.search(user_question)

    # Step 2: Augment
    augmented_prompt = f"""
    Answer the following question using the provided documents.
    
    Question: {user_question}
    
    Documents:
    {relevant_docs}
    """

    # Step 3: Generate
    response = llm.generate(augmented_prompt)

    return response
```

Everything else in a production RAG system — vector databases, reranking, chunking, evaluation — is built on top of this skeleton.

---

## What to Carry Forward

- RAG = **Retriever + Knowledge Base + LLM** working in sequence
- The user sees nothing different — the complexity is internal
- The retriever finds relevant documents; the LLM generates from them
- The augmented prompt is the handoff point between the two
- Each component does what it is best at — search vs. reasoning vs. generation

---

*Related: [How LLMs Work →](../llms/how_llms_work.md) | [Information Retrieval →](../retrieval/information_retrieval_fundamentals.md)*
