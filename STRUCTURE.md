# Repository Structure

A complete map of every folder and file — what it contains and why it exists.

---

## Root

```
AI_Engineer-RAG/
├── README.md              ← Project overview, setup, progress
├── STRUCTURE.md           ← This file
├── environment.yml        ← Conda environment (name: rag-env, Python 3.11)
├── .env                   ← API keys and backend config — never committed
├── .gitignore             ← Ignores .env, large files, local-only folders
└── Quick-Start.txt        ← Daily workflow cheatsheet
```

---

## `knowledge_base/`

The core of the repository. Topic-organized files written for genuine understanding — not summaries, but first-principles explanations that any engineer can learn from and reference.

```
knowledge_base/
│
├── README.md                        ← Full topic index and reading order
│
├── rag_fundamentals/
│   ├── what_is_rag.md               The problem RAG solves, the three phases,
│   │                                why not retrain, core intuition
│   └── rag_applications.md          Where RAG is used across industries,
│                                    how to recognize a RAG opportunity
│
├── rag_architecture/
│   └── rag_architecture_overview.md Components and roles, full data flow,
│                                    augmented prompt structure, code skeleton
│
├── llms/
│   └── how_llms_work.md             Tokens, autoregressive generation, context
│                                    windows, hallucination, temperature —
│                                    why LLMs need grounding to be reliable
│
└── retrieval/
    └── information_retrieval_intro.md  The retriever's role, indexing, relevance
                                        scoring, precision-recall tradeoff,
                                        connection to vector databases
```

**Every file includes:**
- The core idea stated clearly at the top
- Concept built from first principles with analogies
- Tables and ASCII diagrams where they help
- A Key Terms section with precise definitions
- A "What to Carry Forward" section — the things that must stick

**Naming:** `NN_topic_name.md` — numbers preserve reading order within each folder.

---

## `resources/`

Long-form reference documents — more exhaustive than the knowledge base files, with formal definitions, academic citations, and review questions. Intended for deep study and research.

```
resources/
│
├── README.md
│
├── 01_RAG_Overview.md
│       Complete treatment of RAG foundations: LLM architecture, the knowledge
│       problem, full system design, IR fundamentals, all major applications,
│       agentic RAG, foundational paper references, and 12 review questions.
│
├── 02_Building_the_RAG_Pipeline.md
│       Pipeline construction in depth: Python environment setup, the .env
│       pattern, all three LLM backends, the API message format, prompt
│       engineering fundamentals, and a complete working pipeline walkthrough.
│
├── 03_Dense_Retrieval_and_Vector_Databases.md
│       Retrieval from scratch to production: TF-IDF, BM25, the semantic gap,
│       word and sentence embeddings, DPR, cosine similarity, vector databases
│       (Weaviate, Pinecone, Qdrant, ChromaDB), hybrid retrieval, chunking,
│       retrieval evaluation metrics.
│
├── 04_Advanced_Prompt_Engineering.md
│       Production prompt design: system message anatomy, grounding instructions,
│       chain-of-thought over retrieved context, structured output, source
│       citation, few-shot prompting, handling conflicting or insufficient context,
│       prompt versioning.
│
├── papers/      ← Academic papers with reading notes
├── articles/    ← Blog posts and technical articles
└── books/       ← Long-form reference material
```

**Relationship with `knowledge_base/`:**
- `knowledge_base/` — Concise, topic-focused, built for learning and retention
- `resources/` — Exhaustive, formally referenced, built for research and depth

---

## `projects/`

Working implementations. Each folder is a self-contained project with code, documentation, and self-assessment.

```
projects/
│
├── 01_RAG_Overview/
│   │
│   ├── README.md                    What this project covers, how to run it,
│   │                                what each notebook implements
│   │
│   ├── labs/
│   │   ├── lab1_python_refresher.ipynb
│   │   │       Python patterns used throughout RAG codebases: strings, lists,
│   │   │       dicts, functions, list comprehensions, f-strings
│   │   │
│   │   ├── lab2_llm_calls_augmented_prompts.ipynb
│   │   │       LLM API calls, the message format, building augmented prompts,
│   │   │       the complete Retrieve → Augment → Generate loop in code
│   │   │
│   │   └── utils.py                 Unified backend interface —
│   │                                same code works with Ollama, Gemini, Together.ai
│   │
│   ├── assignments/
│   │   ├── C1M1_Assignment.ipynb    Implementation work
│   │   ├── C1M1_Assignment_Solution.ipynb
│   │   ├── unittests.py             Automated tests
│   │   ├── utils.py
│   │   ├── embeddings.joblib        Pre-computed embeddings (binary)
│   │   └── images/                  Diagrams used in the notebook
│   │
│   └── quiz/
│       └── module_01_quiz.md        Conceptual, architectural, comparative,
│                                    and application questions with full answers —
│                                    for interview and self-assessment prep
│
└── 02_Information_Retrieval_and_Search/
    └── [in progress]
```

---

## `notes/`

Working notes — decisions made, setup steps, things to revisit.

```
notes/
└── git_setup.md     Steps taken to connect the local repo to GitHub,
                     commands run, any issues encountered and resolved
```

---

## Environment

| Item | Value |
|---|---|
| Environment name | `rag-env` |
| Python | 3.11 |
| Key packages | `sentence-transformers`, `weaviate-client`, `scikit-learn`, `tiktoken`, `together`, `openai` |
| LLM backends | Ollama (local, free) · Gemini (cloud, free tier) · Together.ai (cloud, paid) |
| Backend config | `.env` at repo root — gitignored, never committed |
| Daily workflow | `Quick-Start.txt` |

---

## What Is Coming Next

| Item | Location |
|---|---|
| Semantic search and BM25 implementation | `projects/02_Information_Retrieval_and_Search/` |
| Retrieval evaluation (precision, recall, MRR) | `projects/02_Information_Retrieval_and_Search/` |
| Vector database deep dive | `knowledge_base/retrieval/` |
| RAG in production — monitoring, evaluation, tradeoffs | `knowledge_base/` + `projects/` |
| Paper reading notes | `resources/papers/` |