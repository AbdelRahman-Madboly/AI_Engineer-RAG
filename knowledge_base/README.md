# Knowledge Base

This is the core of the repository — a topic-organized reference for mastering Retrieval Augmented Generation. Written to be useful to any reader, independent of any course.

---

## How to Read This

The topics are organized by **subject**, not by course module. If you are new to RAG, follow the suggested reading order below. If you are looking for something specific, use the topic map.

---

## Suggested Reading Order

### Foundation Level — Start Here

1. [`rag_fundamentals/01_what_is_rag.md`](./rag_fundamentals/01_what_is_rag.md)  
   The problem RAG solves, the three phases, core intuition

2. [`llms/01_how_llms_work.md`](./llms/01_how_llms_work.md)  
   Tokens, generation, context windows, hallucination — why LLMs need grounding

3. [`rag_fundamentals/02_rag_applications.md`](./rag_fundamentals/02_rag_applications.md)  
   Where RAG is used, how to recognize a RAG opportunity

4. [`rag_architecture/01_rag_architecture_overview.md`](./rag_architecture/01_rag_architecture_overview.md)  
   Full system design — components, data flow, the augmented prompt

### Intermediate Level — Build Understanding

5. [`retrieval/01_dense_retrieval_and_vector_databases.md`](./retrieval/01_dense_retrieval_and_vector_databases.md)  
   From keyword matching to semantic search — embeddings, vector databases, hybrid retrieval

6. [`rag_architecture/02_building_the_rag_pipeline.md`](./rag_architecture/02_building_the_rag_pipeline.md)  
   Environment setup, LLM API calls, building a working pipeline in Python

7. [`llms/02_prompt_engineering.md`](./llms/02_prompt_engineering.md)  
   System messages, grounding instructions, chain-of-thought, citations, structured output

### Deep Dive — Complete the Picture

8. [`rag_fundamentals/03_rag_overview_deep.md`](./rag_fundamentals/03_rag_overview_deep.md)  
   Full academic treatment — formal architecture, research history, agentic RAG, foundational papers

---

## Topic Map

### RAG Fundamentals
| File | What It Covers |
|---|---|
| `01_what_is_rag.md` | Core concept, the problem, the 3 phases, why not retrain |
| `02_rag_applications.md` | Industry applications, recognizing RAG opportunities |
| `03_rag_overview_deep.md` | Academic depth — knowledge problem, full architecture, agentic RAG, paper references |
| `MODULE_01_QUIZ.md` | Interview and exam prep for Module 1 topics |

### RAG Architecture
| File | What It Covers |
|---|---|
| `01_rag_architecture_overview.md` | Components, data flow, augmented prompt structure |
| `02_building_the_rag_pipeline.md` | Environment, LLM API, prompt construction, minimal pipeline in Python |

### LLMs
| File | What It Covers |
|---|---|
| `01_how_llms_work.md` | Tokens, autoregressive generation, context window, hallucination, temperature |
| `02_prompt_engineering.md` | System messages, grounding, CoT, citations, structured output, few-shot prompting |

### Retrieval
| File | What It Covers |
|---|---|
| `01_dense_retrieval_and_vector_databases.md` | TF-IDF, BM25, embeddings, dense retrieval, vector DBs, chunking, hybrid retrieval |
| `02_information_retrieval_fundamentals.md` | *(coming soon)* Indexing, relevance scoring, precision-recall |

---

## Coverage by Course Module

| Module | Title | Files |
|---|---|---|
| 1 | RAG Overview | `rag_fundamentals/01`, `rag_fundamentals/02`, `rag_architecture/01`, `llms/01` |
| 2 | Information Retrieval & Search | `retrieval/01`, `retrieval/02` |
| 3 | Vector Databases | `retrieval/01` (vector DB sections) |
| 4 | LLMs & Text Generation | `llms/01`, `llms/02`, `rag_architecture/02` |
| 5 | RAG in Production | *(coming soon)* |

---

## Contributing to This Knowledge Base

When adding a new file, follow the standards in [`../SKILL_repo_standards.md`](../SKILL_repo_standards.md).

Key rules:
- Every file must have a Key Terms table and a "What to Carry Forward" section
- Files are numbered for reading order within their folder
- Write for a smart reader who is new to the topic — explain from first principles
