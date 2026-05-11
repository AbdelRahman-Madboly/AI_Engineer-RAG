# What is Retrieval Augmented Generation (RAG)?

> **Core idea:** RAG is a technique that improves an LLM's responses by first *retrieving* relevant information from a knowledge base, then using that information to *augment* the prompt before *generating* a response.

---

## The Problem RAG Solves

Large Language Models are trained on massive amounts of text from the internet. This gives them broad general knowledge — but it comes with hard limits:

| Limitation | Why it happens |
|---|---|
| **No private data** | Your company's documents were never in the training set |
| **No recent events** | Training has a cutoff date; the world keeps moving |
| **Hallucinations** | When the model doesn't know something, it sometimes *sounds* like it does |
| **No domain-specific depth** | Niche fields (legal, medical, internal systems) need specialized knowledge |

RAG was built specifically to solve these problems — without retraining the model.

---

## The Core Idea — A Simple Analogy

Imagine you are a consultant asked a question about a client's business. You have general business knowledge — that is your background, like an LLM's training data. But to answer *this specific question* well, you quickly look up the client's files, reports, and relevant documents before you respond.

That is RAG:
1. You receive the question
2. You look up relevant documents
3. You answer using both your background knowledge and what you just found

The LLM is the consultant. The knowledge base is the filing cabinet. RAG is the process of looking things up before answering.

---

## The Three Phases — Where the Name Comes From

```
User Question
      │
      ▼
 ┌─────────────┐
 │  RETRIEVAL  │  ← Find relevant documents from the knowledge base
 └─────────────┘
      │
      ▼
 ┌─────────────┐
 │ AUGMENTATION│  ← Combine those documents with the original question
 └─────────────┘
      │
      ▼
 ┌─────────────┐
 │ GENERATION  │  ← LLM generates a grounded, accurate response
 └─────────────┘
      │
      ▼
   Response
```

- **Retrieval** — Search a knowledge base for documents relevant to the user's question
- **Augmentation** — Build an enriched prompt that includes the question + retrieved documents
- **Generation** — The LLM uses that enriched prompt to generate an accurate, grounded response

---

## Why Not Just Retrain the Model?

This is the first question most people ask. If the model doesn't know your data, why not train it on your data?

| Approach | Cost | Time | Flexibility |
|---|---|---|---|
| **Full Retraining** | Extremely high | Weeks to months | Low — must retrain again for every update |
| **Fine-tuning** | High | Days | Medium — but still static, no live updates |
| **RAG** | Low | Minutes to set up | High — update the knowledge base like a database |

RAG wins on every practical dimension for keeping an LLM current with new or private information. You update your knowledge base the same way you would update a database — no GPU clusters required.

---

## What RAG Enables That Was Not Possible Before

Beyond fixing limitations, RAG opens doors:

- **Source citations** — Every answer can reference exactly which document it came from
- **Auditability** — You can trace every response back to a real source
- **Access control** — Different users or roles can have access to different knowledge bases
- **Real-time updates** — Add a document to the knowledge base, it is immediately usable
- **Dramatically reduced hallucinations** — Grounding the LLM in real documents pulls it away from making things up

---

## Key Terms

| Term | Definition |
|---|---|
| **RAG** | Retrieval Augmented Generation — the full technique |
| **Knowledge Base** | The collection of documents the retriever searches through |
| **Retriever** | The component responsible for searching the knowledge base |
| **Augmented Prompt** | The original question combined with retrieved documents |
| **Grounding** | When an LLM's response is anchored to provided, real information |
| **Hallucination** | When an LLM generates text that sounds plausible but is not true |

---

## What to Carry Forward

- RAG = give the LLM the information it needs *at the time of the question*, not at training time
- The three steps are always: **Retrieve → Augment → Generate**
- It is practical, fast to update, cost-effective, and reduces hallucinations
- It does not replace model training — it complements it

---

*Next: [Applications of RAG →](./rag_applications.md)*
