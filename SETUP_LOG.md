# Setup Log — AI Engineer RAG Repository

> A complete record of everything built, configured, and connected so far.  
> Use this as a reference if you ever need to rebuild the environment from scratch.

---

## The Repository

**Name:** AI_Engineer-RAG  
**Local path:** `D:\AI_Engineer-RAG`  
**GitHub:** https://github.com/AbdelRahman-Madboly/AI_Engineer-RAG  
**Course:** Retrieval Augmented Generation — DeepLearning.AI, taught by Zain Hasan  
**Author:** AbdelRahman Madboly

This is not just a course folder. It is a living knowledge base that documents the full learning journey through RAG — from foundational concepts to production systems.

---

## Current Repository Structure

```
AI_Engineer-RAG/
│
├── content/                              ← Course-provided material (gitignored)
│   ├── RAG Overview/
│   │   ├── C1M1_Ungraded_Lab_1/
│   │   │   └── C1M1_Ungraded_Lab_1.ipynb
│   │   ├── C1M1_Ungraded_Lab_2/
│   │   │   ├── C1M1_Ungraded_Lab_2.ipynb
│   │   │   └── utils.py
│   │   └── lecture transcripts (01–07 .txt files)
│   └── Information Retrieval and Search Foundations/
│       ├── C1M2_Ungraded_Lab_1/
│       │   ├── C1M2_Ungraded_Lab_1.ipynb
│       │   ├── utils.py
│       │   └── embeddings.joblib
│       └── C1M2_Ungraded_Lab_2/
│           ├── C1M2_Ungraded_Lab_2.ipynb
│           └── embeddings.joblib
│
├── knowledge_base/                       ← Topic-organized learning notes (MD)
│   ├── llms/
│   ├── rag_architecture/
│   ├── rag_fundamentals/
│   └── retrieval/
│
├── projects/                             ← Our own code and notebooks
│   ├── 01_RAG_Overview/
│   │   └── 00_Python_RAG_Prep.ipynb     ← ✅ Created by us (see below)
│   ├── assignments/
│   └── labs/
│
├── resources/
│   ├── articles/
│   ├── books/
│   └── papers/
│
├── notes/
│
├── .env                                  ← API key (gitignored — never committed)
├── .gitignore
├── README.md
├── SETUP_LOG.md                          ← This file
├── START_HERE.txt
├── environment.yml
├── how_llms_work.md
├── rag_applications.md
├── rag_architecture_overview.md
└── what_is_rag.md
```

---

## Files Committed to GitHub

These files are live at https://github.com/AbdelRahman-Madboly/AI_Engineer-RAG

| File | Description |
|------|-------------|
| `README.md` | Repository overview, structure, module tracker, setup instructions |
| `environment.yml` | Conda environment definition — all packages listed |
| `.gitignore` | Protects API keys, content folder, large files, OS junk |
| `how_llms_work.md` | Deep reference: tokens, context windows, hallucinations, temperature |
| `rag_applications.md` | Real-world RAG use cases across industries |
| `rag_architecture_overview.md` | Full RAG system diagram, components, augmented prompt anatomy |
| `what_is_rag.md` | Core RAG concept: Retrieve → Augment → Generate |
| `START_HERE.txt` | Quick-start cheat sheet for every new terminal session |
| `projects/01_RAG_Overview/00_Python_RAG_Prep.ipynb` | Our custom prep notebook (see below) |

---

## Conda Environment

**Environment name:** `rag-env`  
**Python version:** 3.11  
**Kernel display name:** RAG Environment

### How it was created

```bash
cd /d D:\AI_Engineer-RAG
conda env create -f environment.yml
conda activate rag-env
python -m ipykernel install --user --name rag-env --display-name "RAG Environment"
```

### Full package list (`environment.yml`)

```yaml
name: rag-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - jupyter
  - notebook
  - ipywidgets
  - ipykernel
  - matplotlib
  - numpy
  - scikit-learn
  - pip:
      - together
      - requests
      - python-dotenv
      - sentence-transformers
      - adjustText
      - tiktoken
      - weaviate-client
      - tqdm
      - pandas
      - openai
```

### Why each package is needed

| Package | Used for |
|---------|----------|
| `together` | Calling Together.ai LLM API — the main LLM provider in this course |
| `python-dotenv` | Loading the API key from `.env` safely |
| `sentence-transformers` | Generating text embeddings (`model.encode()`) |
| `scikit-learn` | PCA for visualizing embeddings in 2D |
| `matplotlib` | Plotting vector spaces and embedding visualizations |
| `numpy` | Vector math — cosine similarity, euclidean distance |
| `ipywidgets` | Interactive word embedding widget in the notebooks |
| `adjustText` | Auto-adjusts text labels in embedding plots |
| `tiktoken` | Counting tokens (OpenAI tokenizer, used for context window math) |
| `weaviate-client` | Vector database client — used in Module 3+ |
| `openai` | OpenAI-compatible SDK (Together.ai uses the same API format) |
| `tqdm` | Progress bars for long loops |
| `pandas` | Data manipulation |

### Every-session commands

```bash
# Open new Anaconda Prompt, then:
cd /d D:\AI_Engineer-RAG
conda activate rag-env

# Launch Jupyter
jupyter notebook

# OR open VS Code in this folder
code .
```

---

## Together.ai API Key

**Provider:** Together.ai — https://www.together.ai  
**Purpose:** Hosting open-source LLMs (Qwen, Llama, Mistral, etc.) via API  
**Model used in labs:** `Qwen/Qwen2.5-7B-Instruct-Turbo` (fast, cheap, good quality)

### How it was set up

1. Created account at https://www.together.ai
2. Generated API key at https://api.together.ai/settings/api-keys
3. Created `.env` file at `D:\AI_Engineer-RAG\.env`:

```
TOGETHER_API_KEY=tgp_v1_s...
```

4. Verified it loads correctly in Python:

```python
from dotenv import load_dotenv
import os

load_dotenv()
key = os.environ.get("TOGETHER_API_KEY", "")
print(key[:8] + "...")   # shows: tgp_v1_s...
```

**Output confirmed:** `✅ TOGETHER_API_KEY loaded: tgp_v1_s...`

### Security

- `.env` is listed in `.gitignore` — it will **never** be committed to GitHub
- Never print the full key — always slice `key[:8]` when verifying
- If the key is ever accidentally exposed, regenerate it immediately at https://api.together.ai/settings/api-keys

---

## Notebooks

### Course notebooks (in `content/` — gitignored)

| Notebook | Module | Topic |
|----------|--------|-------|
| `C1M1_Ungraded_Lab_1.ipynb` | Module 1 | Python refresher: lists, dicts, f-strings |
| `C1M1_Ungraded_Lab_2.ipynb` | Module 1 | Calling the LLM API with `utils.py` |
| `C1M2_Ungraded_Lab_1.ipynb` | Module 2 | Embeddings, cosine similarity, euclidean distance |
| `C1M2_Ungraded_Lab_2.ipynb` | Module 2 | Information retrieval and search evaluation |

### Our notebooks (in `projects/` — committed to GitHub)

#### `00_Python_RAG_Prep.ipynb`
**Location:** `projects/01_RAG_Overview/00_Python_RAG_Prep.ipynb`  
**Purpose:** Extended Python prep that goes beyond Lab 1 — covers everything needed to work confidently in the course

| Section | What it covers |
|---------|---------------|
| 1 — Environment Setup | `load_dotenv()`, verifying API key loaded |
| 2 — Lists | Basics, slicing, `append` vs `extend`, list comprehensions with RAG context |
| 3 — Dictionaries | Basics, `.get()` for safe access, `.items()` for iteration |
| 4 — f-strings | Multi-line prompts, building context blocks from lists of dicts |
| 5 — Functions & Type Hints | Reading `List[Dict]`, `Optional[str]`, writing clean functions |
| 6 — Message Format | The `{role, content}` dict, all three roles explained, conversation history |
| 7 — Prompt Augmentation | The complete Retrieve → Augment → Generate pattern |
| 8 — Live API Test | Direct Together.ai call, `utils.py` usage |
| 9 — Mini RAG Skeleton | Full pipeline in ~20 lines — the foundation of everything in the course |

---

## Knowledge Base Files

These are our own reference documents — written for clarity and reuse.

| File | What it covers |
|------|---------------|
| `what_is_rag.md` | Core concept, the three phases, why not retrain, key terms |
| `rag_applications.md` | Code assistants, enterprise chatbots, healthcare, web search, personal assistants |
| `rag_architecture_overview.md` | Full system diagram, retriever, knowledge base, LLM, augmented prompt anatomy, code skeleton |
| `how_llms_work.md` | Tokens, autoregressive generation, context windows, hallucinations, temperature, training |

---

## Git Workflow

### First push (already done)

```bash
git init
git remote add origin https://github.com/AbdelRahman-Madboly/AI_Engineer-RAG.git
git branch -M main
git add .
git commit -m "initial commit: project structure, environment, and knowledge base"
git push -u origin main
```

### Every time you work

```bash
# After making changes:
git add .
git commit -m "describe what you did"
git push
```

### Commit message pattern used in this repo

```
module 01: add Python prep notebook
module 01 lab 1: complete basic LLM calls
module 02: add embeddings notes
module 02 lab 1: complete cosine similarity exercises
```

---

## Module Progress

| Module | Title | Status |
|--------|-------|--------|
| 01 | RAG Overview | ✅ In Progress |
| 02 | Information Retrieval and Search Foundations | 🔜 Upcoming |
| 03 | Information Retrieval with Vector Databases | 🔜 Upcoming |
| 04 | LLMs and Text Generation | 🔜 Upcoming |
| 05 | RAG Systems in Production | 🔜 Upcoming |

---

## Rebuild from Scratch (if needed)

If you ever need to set this up on a new machine:

```bash
# 1. Clone the repo
git clone https://github.com/AbdelRahman-Madboly/AI_Engineer-RAG.git
cd AI_Engineer-RAG

# 2. Create the environment
conda env create -f environment.yml
conda activate rag-env

# 3. Register the Jupyter kernel
python -m ipykernel install --user --name rag-env --display-name "RAG Environment"

# 4. Create the .env file
echo TOGETHER_API_KEY=your_key_here > .env

# 5. Launch Jupyter
jupyter notebook
```

Then get your API key from https://api.together.ai/settings/api-keys and paste it into `.env`.

---

*Last updated: May 2026 — Module 01 complete*
