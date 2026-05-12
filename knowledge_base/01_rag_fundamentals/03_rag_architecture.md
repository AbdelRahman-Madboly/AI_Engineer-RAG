# RAG Architecture — Components, Data Flow, and the Five Advantages

> **Series:** AI_Engineer-RAG Knowledge Base  
> **Module:** 01 — RAG Fundamentals  
> **File:** 3 of 4 in `01_rag_fundamentals/`  
> **Prerequisites:** [`01_what_is_rag.md`](./01_what_is_rag.md), [`02_rag_applications.md`](./02_rag_applications.md)

---

## The Core Idea

A RAG system has three components — an LLM, a knowledge base, and a retriever — connected by a data flow that transforms a user's raw question into an augmented prompt before anything reaches the LLM. The user sees none of this. From the outside, a RAG system looks exactly like a regular LLM; only the internals differ.

## The Problem It Solves

Understanding the architecture concretely matters because the design choices at each component determine system quality. Knowing what flows where, and why, is the prerequisite for making good engineering decisions about chunk size, retrieval depth, prompt structure, and LLM selection. Architecture also explains the five advantages RAG has over a bare LLM — and why those advantages are structural, not incidental.

---

## The Three Components

A RAG system is built from three components, each with a distinct responsibility:

**The LLM (Generator):** The language model that produces the final response. Its job in a RAG system is focused: receive the augmented prompt and reason over it to produce a coherent, accurate reply. The LLM is not responsible for knowing facts — the retriever handles that.

**The Knowledge Base:** A corpus of trusted, curated documents that contains information the LLM was not trained on. In practice this is a database — often a vector database — of indexed documents. It may contain plain text, PDFs, HTML, code, structured records, or any other content that has been preprocessed into a searchable form.

**The Retriever:** The component that bridges the user's query and the knowledge base. Given a query, it searches the knowledge base, scores documents by relevance, and returns the top-ranked results. The retriever is the architectural heart of RAG; retrieval quality is the primary determinant of overall system quality.

---

## Data Flow: End to End

```
User submits a prompt
        │
        ▼
┌───────────────────┐
│     Retriever     │
│  (receives query) │
└───────────────────┘
        │
        ▼
┌───────────────────────────────┐
│        Knowledge Base         │
│  (indexed document database)  │
│                               │
│  Score all documents by       │
│  relevance to the query       │
│  Return top-k documents       │
└───────────────────────────────┘
        │
        │  Retrieved documents
        ▼
┌──────────────────────────────────┐
│       Augmented Prompt           │
│  = Original Query                │
│  + Retrieved Document 1          │
│  + Retrieved Document 2          │
│  + ... (top-k documents)         │
└──────────────────────────────────┘
        │
        ▼
┌───────────────┐
│      LLM      │──────▶ Final Response to User
│  (Generator)  │
└───────────────┘
```

The user experiences only the beginning and end: they send a prompt, they receive a response. The retrieval step, document scoring, and prompt augmentation are invisible. The added latency is the only externally observable difference — a small cost for substantially improved accuracy.

---

## What an Augmented Prompt Looks Like

The augmented prompt is the central artifact of the RAG architecture. Its structure is straightforward: a system-level instruction telling the LLM to use the provided context, the user's original question, and the retrieved document text. A representative example:

```
Respond to the following question using the retrieved information provided below.
If the retrieved information does not contain a sufficient answer, say so clearly.

Question: Why are hotel prices in Vancouver super expensive this weekend?

Retrieved Context:

[Document 1 — Source: Vancouver Events Calendar, May 2025]:
Taylor Swift's Eras Tour is scheduled for two performances at Rogers Arena
in Vancouver on Saturday and Sunday. Estimated attendance: 80,000 per night.

[Document 2 — Source: Tourism Vancouver Hotel Report, Q2 2025]:
Hotel occupancy in Vancouver typically reaches 95%+ during major arena events.
Room rates have historically increased 3–5x over baseline during such periods.

Answer:
```

The LLM now has everything it needs. It does not need to have been trained on Taylor Swift's tour schedule; it reads it in the prompt and reasons from it. This is in-context learning at work.

---

## The Five Structural Advantages

The retriever's addition to the standard LLM pipeline is a modest architectural change with outsized consequences. These five advantages are structural properties of the architecture, not configuration choices.

**1. Access to otherwise unavailable information.** Private enterprise documents, proprietary databases, content published after the training cutoff — none of these can be memorized by an LLM, but all of them can be indexed in a knowledge base and retrieved at query time. RAG is often the only viable way to make this class of information available to a language model.

**2. Hallucination reduction.** Hallucination typically occurs when a model is asked about something outside its training distribution and generates plausible-sounding text to fill the gap. When the relevant information is explicitly present in the augmented prompt, the model can ground its response in concrete evidence rather than statistical inference over training patterns. Hallucination rates in well-designed RAG systems have declined consistently as both model quality and retrieval quality improve.

**3. Easy knowledge updates.** Retraining an LLM is a months-long, multi-million-dollar undertaking. Updating a RAG knowledge base is a database operation — add a document, re-index, done. From that point forward, the system responds based on the new information. This is the most practically significant advantage for any application where the relevant knowledge changes faster than a training cycle.

**4. Source citation.** Because the retrieved documents are explicitly included in the augmented prompt, the LLM can reference them in its response. Citation is built into the architecture: the system always knows where its context came from. This enables human verification and substantially increases trustworthiness in high-stakes domains.

**5. Separation of concerns.** The LLM and the retriever each do what they are best at. The retriever handles fact-finding — scanning a large corpus, scoring relevance, filtering noise. The LLM handles synthesis and reasoning — taking structured context and producing coherent, well-articulated responses. Neither component is forced to operate outside its strength. This separation also makes the system easier to debug: retrieval failures and generation failures have distinct signatures and are addressed with different interventions.

---

## A Minimal Code Illustration

The architecture maps directly to code. In its simplest form, the RAG pipeline is three functions:

```python
def retrieve(query: str) -> list[str]:
    # Search the knowledge base; return the top-k most relevant documents
    ...

def build_augmented_prompt(query: str, docs: list[str]) -> str:
    context = "\n\n".join(docs)
    return (
        f"Respond to the following question using the context below.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )

def generate(prompt: str) -> str:
    # Send the augmented prompt to the LLM; return the response
    ...

# The full RAG pipeline
query = "Why are hotel prices in Vancouver super expensive this weekend?"
docs = retrieve(query)
augmented_prompt = build_augmented_prompt(query, docs)
response = generate(augmented_prompt)
```

The abstractions here hide significant engineering depth — how the retriever scores documents, how the knowledge base is indexed, how the prompt is structured for best performance — but the logical flow is exactly this simple.

---

## Key Terms

| Term | Definition |
|---|---|
| **LLM (Generator)** | The language model component of a RAG system; responsible for synthesizing the augmented prompt into a final response. |
| **Knowledge base** | The indexed document corpus from which the retriever draws; contains information the LLM was not trained on. |
| **Retriever** | The component that searches the knowledge base and returns the most relevant documents for a given query. |
| **Augmented prompt** | The input to the LLM: the user's original query plus the retrieved documents, structured to direct the LLM to use the provided context. |
| **Top-k retrieval** | Returning the k highest-scoring documents from the knowledge base for a given query; k is a tunable hyperparameter. |
| **Relevance score** | A numerical measure of how closely a document matches a query; the basis for ranking and selection in the retriever. |
| **Separation of concerns** | The architectural principle that each component handles the task it is best at: the retriever finds information, the LLM reasons over it. |
| **In-context learning** | The LLM's ability to use information provided in its prompt — not just training weights — to answer questions and perform tasks. |
| **Latency** | The added response time introduced by the retrieval step; a practical engineering trade-off for improved accuracy. |
| **Chunking** | Dividing long documents into smaller segments before indexing; a prerequisite for retrieval and a critical hyperparameter for quality. |

---

## What to Carry Forward

- Three components: LLM, knowledge base, retriever. Each has a single well-defined responsibility; the quality of the overall system depends on the quality of each.
- The data flow is sequential: query → retriever → knowledge base → retrieved docs → augmented prompt → LLM → response. The user sees only the first and last steps.
- The augmented prompt is the architectural core: it is how the retriever's output becomes the LLM's input, and it is where retrieval quality and prompt engineering intersect.
- Five structural advantages: access to unavailable information, hallucination reduction, easy knowledge updates, source citation, and separation of concerns. These follow from the architecture, not from configuration.
- The user experience of a RAG system is identical to a bare LLM; the engineering difference is entirely internal.
- Even a minimal code implementation reveals the structure clearly: retrieve, augment, generate. Everything else is optimization of those three steps.

---

## Navigation

- **Previous:** [`02_rag_applications.md`](./02_rag_applications.md) — the five industry domains where RAG is deployed.
- **Next:** [`04_why_rag_over_retraining.md`](./04_why_rag_over_retraining.md) — why RAG is preferred over fine-tuning and long-context approaches.
- **Related:** [`02_llms/01_how_llms_work.md`](../02_llms/01_how_llms_work.md) — how the LLM component processes the augmented prompt; [`02_llms/02_llm_limitations.md`](../02_llms/02_llm_limitations.md) — the context window and hallucination constraints the architecture addresses.
- **Academic depth:** `01_RAG_Overview.md` — Section 5 (RAG System Architecture). Lewis et al. (2020), [`arXiv:2005.11401`](https://arxiv.org/abs/2005.11401).