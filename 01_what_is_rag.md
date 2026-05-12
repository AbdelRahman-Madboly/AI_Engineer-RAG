# What Is RAG — The Knowledge Gap Problem and the Two-Phase Model

> **Series:** AI_Engineer-RAG Knowledge Base  
> **Module:** 01 — RAG Fundamentals  
> **File:** 1 of 4 in `01_rag_fundamentals/`  
> **Prerequisites:** None — this is the entry point for the series.

---

## The Core Idea

An LLM knows only what was in its training data. RAG solves the knowledge gap by doing what any careful person does before answering a hard question: retrieving relevant information first, then reasoning over it to respond. The name says it exactly — **Retrieval**-Augmented **Generation**.

## The Problem It Solves

Every LLM is a snapshot. Training ends, weights freeze, and the model's knowledge stops updating. This creates three categories of questions it cannot reliably answer: questions about events that happened after training ended, questions requiring private or proprietary information that was never in the training corpus, and questions demanding deep specialization in niche domains that were sparsely represented. For many real-world applications, these are precisely the questions that matter most. Deploying a bare LLM in these contexts leads to hallucination — the model generating plausible-sounding but factually incorrect responses — because it fills knowledge gaps with statistical probability rather than verified fact.
---

## The Hotel Analogy: Three Levels of Question Difficulty

A concrete way to see the problem is through three questions of escalating specificity:

**Level 1 — General knowledge:** *"Why are hotels expensive on weekends?"*  
You can answer this immediately from general knowledge: weekend travel demand is higher, so room competition drives up prices. An LLM handles this comfortably — the answer was almost certainly in its training data, many times over.

**Level 2 — Context-specific:** *"Why are hotels in Vancouver super expensive this coming weekend?"*  
You cannot answer this without current information. A quick search reveals Taylor Swift is performing two nights at Rogers Arena. With that information in hand, the answer follows easily. Without it, any response is a guess. An LLM in this situation either confabulates a plausible-sounding reason or refuses to answer — neither is acceptable.

**Level 3 — Specialized/private:** *"Why doesn't Vancouver have more hotel capacity close to downtown?"*  
Answering this requires deep research into Vancouver's urban planning history, zoning law, and land economics — specialized, possibly proprietary information that even a capable LLM likely encountered only sparsely if at all.

The pattern across all three is the same: the harder the question, the more information must first be gathered before reasoning can begin. Recognizing this pattern is the core insight of RAG.

---

## The Two-Phase Model

RAG makes explicit two cognitive phases that are naturally intertwined in human reasoning:

**Phase 1 — Retrieval:** Before answering, the system searches an external knowledge base for documents relevant to the question. This knowledge base contains information the LLM was not trained on: recent documents, private company data, specialized corpora, live databases. The retriever finds the most relevant pieces and returns them.

**Phase 2 — Generation:** The LLM receives an **augmented prompt** — the user's original question plus the retrieved documents — and reasons over this combined input to produce a response.

```
User Question
      │
      ▼
┌─────────────┐       ┌──────────────────┐
│  Retriever  │──────▶│  Knowledge Base  │
│             │◀──────│  (Documents)     │
└─────────────┘       └──────────────────┘
      │
      │  Retrieved Documents
      ▼
┌──────────────────────────┐
│  Augmented Prompt        │
│  = Question + Documents  │
└──────────────────────────┘
      │
      ▼
┌─────────────┐
│     LLM     │──▶  Response
│ (Generator) │
└─────────────┘
```

The user experience is unchanged — they submit a question and receive an answer. The internal difference is that the LLM now has access to information it could not have memorized during training.

---

## The Augmented Prompt: Why "Just Put It in the Prompt" Works

The mechanism is simpler than it might seem. Instead of sending the user's raw question to the LLM, the system constructs an augmented prompt:

```
Answer the following question using the retrieved context below.

Question: Why are hotels in Vancouver so expensive this coming weekend?

Retrieved Context:
[Document 1]: Taylor Swift's Eras Tour is scheduled for two nights at Rogers
Arena in Vancouver this Saturday and Sunday, drawing an estimated 80,000
attendees per night...

[Document 2]: Vancouver hotel occupancy rates typically spike during major
concert events, with room prices increasing 3–5x over baseline...

Answer:
```

This works because LLMs are remarkably good at synthesizing and reasoning over information provided in their context window, even when that information was never part of their training data. The model does not need to have "memorized" the fact about Taylor Swift — it needs only to read it in the prompt and reason from it. RAG exploits this in-context reasoning capability directly.

The paper that formally named and described this technique — Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — showed that this simple mechanism substantially outperforms both vanilla LLMs and retrieval-only systems on knowledge-intensive tasks.

---

## Key Terms

| Term | Definition |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | A technique that augments an LLM's prompt with documents retrieved from an external knowledge base, enabling the model to answer questions it was not trained on. |
| **Knowledge gap** | The structural mismatch between what an LLM knows from training and what a specific application requires it to know. |
| **Retrieval** | The first phase of RAG: searching the knowledge base for documents relevant to the user's question. |
| **Generation** | The second phase of RAG: the LLM reasoning over the augmented prompt to produce a response. |
| **Augmented prompt** | The input sent to the LLM in a RAG system, consisting of the original user query plus retrieved documents. |
| **Knowledge base** | A curated corpus of documents — private, recent, or specialized — that the retriever searches. |
| **Retriever** | The component that searches the knowledge base and returns the most relevant documents for a given query. |
| **Hallucination** | The generation of fluent but factually incorrect text; a consequence of optimizing for token probability when the model lacks relevant training signal. |
| **In-context learning** | The LLM's ability to use information provided in its prompt — not just its training weights — to perform tasks and answer questions. |
| **Training cutoff** | The date after which no information was included in an LLM's training corpus; the hard boundary of its static knowledge. |

---

## What to Carry Forward

- LLMs are knowledge snapshots: their information is frozen at training time and cannot be updated without retraining. This creates systematic failure modes for private, recent, or specialized queries.
- The hotel analogy establishes three tiers: questions an LLM can answer alone, questions requiring current context, and questions requiring deep specialized knowledge. RAG addresses the latter two.
- RAG makes explicit what good reasoners already do naturally: gather relevant information first, then reason over it.
- The augmented prompt — original question plus retrieved documents — is the central mechanism. Its power rests on the LLM's ability to reason over context it was never trained on.
- "Just put it in the prompt" is not a hack; it is the correct architectural response to the knowledge gap problem, and it is grounded in how LLMs process context.
- The user experience of a RAG system is identical to using a bare LLM; only the internals differ.

---

## Navigation

- **Previous:** None — this is the first file in the series.
- **Next:** [`02_rag_applications.md`](./02_rag_applications.md) — where RAG is deployed in practice across five industry domains.
- **Related:** [`02_llms/01_how_llms_work.md`](../02_llms/01_how_llms_work.md) — the mechanics of LLMs that make the augmented prompt work; [`02_llms/02_llm_limitations.md`](../02_llms/02_llm_limitations.md) — hallucination, context windows, and knowledge cutoff in depth.
- **Academic depth:** `01_RAG_Overview.md` — Section 3 (The Knowledge Problem) and Section 4 (The Core Idea of RAG). The original RAG paper: Lewis et al. (2020), [`arXiv:2005.11401`](https://arxiv.org/abs/2005.11401).
