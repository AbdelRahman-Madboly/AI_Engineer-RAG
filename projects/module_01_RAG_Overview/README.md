# Module 01 — RAG Overview

> **Course:** Retrieval Augmented Generation by DeepLearning.AI  
> **Instructor:** Zain Hasan  
> **Status:** ✅ Complete

---

## What This Module Is About

This module is the foundation. Before you can build a RAG system, you need a clear mental model of what it is, why it exists, and how its components work together.

By the end of this module, you should be able to:

- Explain what RAG is and why it was developed
- Describe the three core components of a RAG system and what each one does
- Recognize situations where a RAG approach is appropriate
- Explain how LLMs work at a high level — tokens, prediction, training, hallucinations, context windows
- Explain how information retrieval works — indexing, similarity scoring, precision vs. recall
- Write a minimal RAG pipeline in Python using retrieve and generate functions
- Build and pass an augmented prompt to an LLM and observe the difference in response quality

---

## Topics in This Module

| Topic | File | Description |
|-------|------|-------------|
| What Is RAG | [`knowledge_base/rag/what_is_rag.md`](../../knowledge_base/rag/what_is_rag.md) | Core concept, motivation, retrieval-generation loop |
| Applications of RAG | [`knowledge_base/rag/rag_applications.md`](../../knowledge_base/rag/rag_applications.md) | Real-world use cases and how to recognize opportunities |
| RAG Architecture | [`knowledge_base/architecture/rag_architecture.md`](../../knowledge_base/architecture/rag_architecture.md) | Full pipeline, component roles, failure modes |
| How LLMs Work | [`knowledge_base/llms/how_llms_work.md`](../../knowledge_base/llms/how_llms_work.md) | Tokens, prediction, training, hallucinations, context windows |
| Information Retrieval | [`knowledge_base/retrieval/information_retrieval_intro.md`](../../knowledge_base/retrieval/information_retrieval_intro.md) | Indexing, scoring, search methods, precision-recall |

---

## Labs

| Lab | File | Focus |
|-----|------|-------|
| Lab 1 — Python Refresher | [`labs/lab1_python_refresher.ipynb`](labs/lab1_python_refresher.ipynb) | Lists, dicts, f-strings, list comprehensions |
| Lab 2 — LLM Calls & Augmented Prompts | [`labs/lab2_llm_calls_augmented_prompts.ipynb`](labs/lab2_llm_calls_augmented_prompts.ipynb) | Calling LLMs, building augmented prompts |

---

## Key Concepts — Quick Reference

### RAG in Three Words
**Retrieve. Augment. Generate.**

### The Three Components
1. **Knowledge Base** — collection of trusted, relevant documents
2. **Retriever** — finds the most relevant documents for a given query
3. **LLM** — generates a response grounded in the retrieved information

### Why RAG Works
LLMs can use information provided in the prompt even if that information was not in their training data. RAG exploits this by putting the right information in the prompt before generation.

### When to Use RAG
- The LLM needs private or proprietary data
- The LLM needs recent information (post training cutoff)
- The domain is too specialized to be well-covered in training data
- Source citation is important
- The information changes frequently

### How LLMs Generate Text
1. Process all current text (prompt + prior tokens)
2. Calculate probability for every token in vocabulary
3. Randomly sample next token from that distribution
4. Repeat until done

### Hallucinations
Not a malfunction — LLMs generate *probable* text, not *true* text. When the right information isn't in training data, the model generates something that sounds probable but may be wrong. RAG solves this by providing accurate information in the prompt.

### Context Window
Maximum tokens an LLM can process at once. Retrieved documents must fit within this limit alongside the question and instructions. Larger = more expensive. Good retrieval = only relevant documents are included.

### How Retrieval Works
1. Documents are indexed ahead of time
2. Query arrives → similarity scores are calculated
3. Top-ranked documents are returned
4. Precision-recall trade-off: more documents = higher recall, lower precision

---

## Module Summary

RAG is built on one elegant insight: **LLMs don't need to know everything — they just need to be shown the right information at the right time.**

The knowledge base stores that information. The retriever finds the right piece of it. The LLM synthesizes and responds.

Every module that follows builds on this foundation — better retrieval, smarter indexing, more capable LLM usage, and systems that can be deployed and monitored in production.

---

## Module Quiz

Ready to test your understanding?

→ [`quiz/module_01_quiz.md`](quiz/module_01_quiz.md)

The quiz covers all topics in this module in an exam and interview-ready format.
