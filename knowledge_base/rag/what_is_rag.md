# What Is RAG — Retrieval Augmented Generation

> **Topic Area:** RAG Fundamentals  
> **Covers:** Definition, motivation, the retrieval-generation loop, and why RAG matters

---

## The Core Problem RAG Solves

Large Language Models are trained on massive datasets — trillions of tokens scraped from the internet. That training gives them remarkable general knowledge: they can write code, summarize documents, answer questions, and reason through problems.

But that knowledge is **frozen at a point in time** and **limited to what was publicly available**.

Ask an LLM about your company's internal policy? It doesn't know.  
Ask about something that happened last week? It doesn't know.  
Ask about a niche domain with sparse public documentation? It will guess — and sometimes confidently get it wrong.

This is not a bug. LLMs are not designed to be databases. They are designed to generate **probable, coherent text** based on patterns in their training data. When the right information isn't in that training data, the model will still generate something — it just might not be true.

**RAG solves this by giving the model the information it needs, at query time, before it generates a response.**

---

## The Analogy That Makes It Click

Think about how you answer questions.

If someone asks: *"Why are hotels expensive on weekends?"* — you can answer immediately. General knowledge.

If someone asks: *"Why are hotels in Vancouver super expensive this particular weekend?"* — you'd need to look something up. Maybe Taylor Swift is in town. Once you have that fact, you can answer well.

If someone asks: *"Why doesn't Vancouver have more downtown hotel capacity?"* — you'd need deep research into urban planning, zoning laws, and development history.

Notice the pattern:
1. You **collect information** (retrieval)
2. You **reason over it** and respond (generation)

RAG works the same way. It adds a **retrieval step** before the LLM generates a response.

---

## What RAG Actually Is

**Retrieval Augmented Generation (RAG)** is a technique that improves LLM responses by:

1. Receiving a user's question
2. **Retrieving** relevant documents from a knowledge base
3. **Augmenting** the prompt — adding those documents alongside the original question
4. Sending the augmented prompt to the LLM
5. The LLM **generates** a response grounded in the retrieved information

The name maps directly to these three steps:
- **Retrieval** — find relevant information
- **Augmented** — add it to the prompt
- **Generation** — the LLM responds

---

## The Key Insight: Just Put It In the Prompt

LLMs are remarkably good at using information you provide in the prompt — even if that information was never in their training data. The model doesn't need to have been *trained* on a fact to use it. It just needs to *see* it in the prompt.

This is the elegant simplicity of RAG. You don't retrain the model. You don't fine-tune it. You just put the right information in front of it before it answers.

```
Without RAG:
  User: "What is our refund policy for international orders?"
  LLM: [guesses or says it doesn't know]

With RAG:
  User: "What is our refund policy for international orders?"
  System: [retrieves relevant policy document]
  Augmented Prompt: "Answer the following question using this context:
                     [policy document text]
                     Question: What is our refund policy for international orders?"
  LLM: [gives accurate answer grounded in the policy]
```

---

## Three Components of Every RAG System

Every RAG system — from the simplest prototype to the most complex production system — has these three components:

| Component | Role |
|-----------|------|
| **Knowledge Base** | A collection of trusted documents the system can retrieve from |
| **Retriever** | Finds and returns the most relevant documents for a given query |
| **LLM** | Reads the augmented prompt and generates a grounded response |

These components and how they interact are covered in detail in the [RAG Architecture](../architecture/rag_architecture.md) topic.

---

## Why RAG Matters

RAG has become the most widely used technique for improving LLM applications. Here's why it works so well:

**Reduces hallucinations.** When the LLM has relevant, accurate information in its prompt, it is less likely to fabricate answers. Its response is *grounded* in real content.

**Keeps information up to date.** Retraining a model is expensive and slow. With RAG, you just update your knowledge base — the model immediately benefits from new information.

**Enables private data.** Your company's internal documents, databases, customer records — none of this was in the LLM's training data. RAG is often the *only* way to make this information available to an LLM.

**Enables source citation.** Because you know exactly which documents were retrieved, you can include citation information in the prompt, and the LLM can reference its sources in the response.

**Division of labor.** The retriever does what it's good at (finding relevant documents quickly). The LLM does what it's good at (reasoning over and synthesizing information). Neither is asked to do the other's job.

---

## Where RAG Is Used Today

RAG is everywhere in production AI systems:

- **Customer service chatbots** — grounded in a company's product docs and policies
- **Code generation tools** — grounded in your actual codebase and style guide
- **AI web search** — the knowledge base is the entire internet
- **Healthcare assistants** — grounded in recent medical journals or case notes
- **Legal research tools** — grounded in case documents and statutes
- **Personal AI assistants** — grounded in your emails, calendar, and documents
- **Enterprise internal tools** — answering questions about HR policies, engineering docs, etc.

---

## The Evolution of RAG

RAG started as a relatively straightforward technique — retrieve a few documents, inject them into a prompt. But the field has evolved rapidly:

- **Better LLMs** have gotten much better at actually using the retrieved context, leading to fewer hallucinations
- **Larger context windows** mean you can now retrieve and inject far more text before hitting limits
- **Multimodal RAG** extends retrieval beyond text — to PDFs, slides, images
- **Agentic RAG** is the frontier — AI agents that decide *what* to retrieve, *when* to retrieve it, and whether the retrieved information is good enough before generating a final answer

The core idea remains the same. The systems built on top of it keep getting more sophisticated.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| The problem | LLMs have frozen, incomplete knowledge |
| The solution | Retrieve relevant information and add it to the prompt before generation |
| How it works | Retrieval → Augmentation → Generation |
| Why it works | LLMs can use information in the prompt even if not in training data |
| Main benefit | Accurate, grounded, up-to-date, source-citable responses |
| Main use case | Any application that needs an LLM to reason over private, recent, or specialized data |

---

## Related Topics

- [RAG Architecture](../architecture/rag_architecture.md) — How the components connect
- [Applications of RAG](rag_applications.md) — Concrete production examples
- [Introduction to LLMs](../llms/how_llms_work.md) — Why LLMs need grounding
- [Introduction to Information Retrieval](../retrieval/information_retrieval_intro.md) — How the retriever finds relevant documents
