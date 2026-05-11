# AI Engineer — Retrieval Augmented Generation (RAG)

> A structured learning repository documenting the journey through RAG — from foundational concepts to production systems.  
> Based on the **Retrieval Augmented Generation** course by [DeepLearning.AI](https://www.deeplearning.ai/) — taught by Zain Hasan.

---

## What This Repo Is

This is not just a course folder. It is a **living knowledge base** designed to:

- Document RAG concepts in a clear, reusable format that anyone can learn from
- Track hands-on code projects and lab implementations
- Store and organize research papers, articles, and books for deep mastery
- Serve as a portfolio showing progression from basics to advanced RAG systems

---

## Repository Structure

```
AI_Engineer-RAG/
│
├── knowledge_base/          ← Topic-based learning material (MD files)
│   ├── rag/                 ← What RAG is, why it matters, how it works
│   ├── llms/                ← How Large Language Models work
│   ├── retrieval/           ← Information retrieval, search, vector DBs
│   └── architecture/        ← RAG system design and components
│
├── projects/                ← Code, labs, assignments per module
│   └── module_01_RAG_Overview/
│       ├── labs/            ← Ungraded lab notebooks
│       └── assignments/     ← Graded assignment code
│
├── resources/               ← External material for deeper understanding
│   ├── papers/              ← Scientific papers
│   ├── articles/            ← Blog posts and technical articles
│   └── books/               ← Books and long-form references
│
├── notes/                   ← Personal reflections and learning notes
│
├── environment.yml          ← Conda environment (reproducible setup)
└── .gitignore
```

---

## Modules Covered

| Module | Title | Status |
|--------|-------|--------|
| 01 | RAG Overview | ✅ In Progress |
| 02 | Information Retrieval and Search Foundations | 🔜 Upcoming |
| 03 | Information Retrieval with Vector Databases | 🔜 Upcoming |
| 04 | LLMs and Text Generation | 🔜 Upcoming |
| 05 | RAG Systems in Production | 🔜 Upcoming |

---

## Knowledge Base Topics

The `knowledge_base/` folder is organized by **topic**, not by module. This makes it easy to look up a concept without knowing which module it came from.

| Topic Area | Description |
|------------|-------------|
| `rag/` | Core RAG concepts, motivation, and workflow |
| `llms/` | LLM internals — tokens, training, hallucinations, context windows |
| `retrieval/` | Search methods, indexing, ranking, vector databases |
| `architecture/` | RAG system design, components, and trade-offs |

---

## How to Set Up the Environment

This repo uses **Conda** for environment management.

```bash
# 1. Clone the repository
git clone https://github.com/AbdelRahman-Madboly/AI_Engineer-RAG.git
cd AI_Engineer-RAG

# 2. Create the environment
conda env create -f environment.yml

# 3. Activate it
conda activate rag-course

# 4. Launch Jupyter
jupyter notebook
```

---

## Tech Stack

- **Python 3.11**
- **Together AI** — LLM hosting (open-source models)
- **Weaviate** — Vector database (used in Module 3+)
- **Jupyter Notebook** — Lab environment
- **OpenAI SDK** (Together-compatible)

---

## Author

**AbdelRahman Madboly**  
AI Engineer in progress — documenting the path.

---

*"RAG may be the most commonly built type of LLM-based application in the world today."* — Andrew Ng