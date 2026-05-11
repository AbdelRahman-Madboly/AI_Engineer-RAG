# AI Engineer — RAG

A knowledge base and implementation portfolio for **Retrieval Augmented Generation (RAG)** — covering foundations, architecture, retrieval systems, and production techniques.

---

## Overview

This repository documents a deep, structured study of RAG systems — from the core concepts to working implementations. It is organized as a standalone reference that any engineer or researcher can use to understand, build, and improve RAG pipelines.

The work here covers:
- How RAG works and why it exists
- The full system architecture and its components
- How large language models generate text and where they fail
- Information retrieval — from keyword search to semantic vector search
- Prompt engineering for grounded, reliable LLM output
- Implementations ranging from minimal prototypes to more complete systems

---

## Repository Structure

```
AI_Engineer-RAG/
│
├── knowledge_base/          ← Topic-organized reference files
│   ├── rag_fundamentals/    ← Core concepts: what RAG is, why it works, where it applies
│   ├── rag_architecture/    ← System design, data flow, pipeline construction
│   ├── llms/                ← LLM internals, generation mechanics, prompt engineering
│   └── retrieval/           ← Information retrieval, embeddings, vector databases
│
├── projects/                ← Implementations and experiments
│   ├── 01_RAG_Overview/     ← Building a working RAG pipeline from scratch
│   └── 02_Information_Retrieval_and_Search/  ← Search techniques and retrieval systems
│
└── resources/               ← Deep reference documents, papers, articles, books
    ├── 01_RAG_Overview.md
    ├── 02_Building_the_RAG_Pipeline.md
    ├── 03_Dense_Retrieval_and_Vector_Databases.md
    ├── 04_Advanced_Prompt_Engineering.md
    ├── papers/
    ├── articles/
    └── books/
```

---

## Knowledge Base

The knowledge base is organized by **topic**, not by project or implementation. Each file is written to explain a concept from first principles — with the depth needed to actually use it, not just describe it.

| Area | Topics Covered |
|---|---|
| [rag_fundamentals/](./knowledge_base/rag_fundamentals/) | What RAG is, the problem it solves, real-world applications |
| [rag_architecture/](./knowledge_base/rag_architecture/) | System components, data flow, augmented prompt construction |
| [llms/](./knowledge_base/llms/) | Tokens, autoregressive generation, context windows, hallucination, temperature |
| [retrieval/](./knowledge_base/retrieval/) | Indexing, relevance scoring, BM25, embeddings, dense retrieval, vector databases |

See the [knowledge base index](./knowledge_base/README.md) for the full reading order and topic map.

---

## Projects

Each project folder contains working code with documented explanations of what is implemented and why.

| Project | What It Builds |
|---|---|
| [01_RAG_Overview](./projects/01_RAG_Overview/) | A complete RAG pipeline — retrieval, prompt augmentation, LLM generation |
| [02_Information_Retrieval_and_Search](./projects/02_Information_Retrieval_and_Search/) | Keyword search, semantic search, hybrid retrieval, evaluation |

---

## Resources

The [`resources/`](./resources/) folder contains four long-form reference documents — formal, detailed treatments of the core topics with academic references and review questions. The `papers/`, `articles/`, and `books/` subfolders hold external material added over time.

---

## Setup

**Prerequisites:** [Anaconda](https://www.anaconda.com/), [VS Code](https://code.visualstudio.com/), [Ollama](https://ollama.com/) (optional — for local inference)

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate rag-env
```

### 2. Configure your LLM backend

Create a `.env` file at the repo root:

```bash
# Choose one: ollama | gemini | together
LLM_BACKEND=ollama
```

### Supported backends

| Backend | Cost | Notes |
|---|---|---|
| **Ollama** | Free | Runs locally — no API key needed. Run `ollama pull qwen2.5:7b` first. |
| **Gemini** | Free tier | API key from [aistudio.google.com](https://aistudio.google.com/apikey) |
| **Together.ai** | Paid | Add `TOGETHER_API_KEY` to `.env` |

### 3. Daily workflow

```bash
cd D:\AI_Engineer-RAG
conda activate rag-env
jupyter notebook        # or: code .
```

After working:
```bash
git add .
git commit -m "describe what you did"
git push
```

---

## Progress

| Area | Status |
|---|---|
| RAG fundamentals and architecture | ✅ Complete |
| LLM internals and prompt engineering | ✅ Complete |
| Information retrieval and vector databases | 🔄 In progress |
| RAG in production — evaluation, monitoring | 🔜 Upcoming |

---

## Author

**AbdelRahman Madboly**

[![GitHub](https://img.shields.io/badge/GitHub-AbdelRahman--Madboly-181717?logo=github)](https://github.com/AbdelRahman-Madboly/AI_Engineer-RAG)