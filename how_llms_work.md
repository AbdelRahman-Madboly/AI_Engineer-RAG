# How Large Language Models Work

> LLMs are the generation engine in RAG. Understanding how they work — and where they fail — explains every design decision in the retrieval layer.

---

## The One-Sentence Description

An LLM is a mathematical model that predicts the most probable next token given all the tokens that came before it.

That is it. Everything else — the apparent intelligence, coherent paragraphs, the ability to reason — emerges from doing this prediction at massive scale on massive data.

---

## The "Fancy Autocomplete" Mental Model

You have used autocomplete on your phone. You type "I will be there in" and it suggests "five minutes." It learned from millions of messages what words typically follow other words.

LLMs do the same thing, but at a scale that produces something that *looks* like understanding:

- Trained on trillions of words from the internet, books, and code
- With billions of internal numerical parameters
- Using a Transformer neural network that captures long-range relationships

The result: a model that writes essays, debugs code, translates languages — all by predicting the next word, over and over.

---

## Tokens — The Unit LLMs Actually Use

LLMs do not work with full words. They work with **tokens** — sub-word pieces of text.

**Why tokens?** Words like "unhappy" or "programmatically" are compound. Instead of storing every possible word, the model builds words from reusable pieces.

| Text | Tokens | Count |
|---|---|---|
| `London` | `London` | 1 |
| `unhappy` | `un` + `happy` | 2 |
| `programmatically` | `program` + `mat` + `ically` | 3 |
| `California is beautiful` | `California` + ` is` + ` beautiful` | 3 |

**Rule of thumb:** 1 token ≈ 0.75 words ≈ 4 characters

Most LLMs have a vocabulary of 50,000 to 100,000+ tokens.

---

## How an LLM Generates Text — Step by Step

Given the prompt: *"What a beautiful day, the sun is..."*

**Step 1: Process the entire prompt**
The model reads every token simultaneously, building understanding of meaning and word relationships.

**Step 2: Calculate probabilities across the entire vocabulary**
For every token in its vocabulary (~50,000+), it computes how likely that token is to come next:

```
"shining"    → 80% probability
"rising"     → 8% probability
"warm"       → 5% probability
"exploding"  → 0.001% probability
```

**Step 3: Randomly sample one token from that distribution**
It does not always pick the most probable token. It samples — so 80% of the time it picks "shining," but sometimes "rising." This is what makes each run different.

**Step 4: Repeat**
Add the new token, then repeat the entire process for the next token. This continues until the model generates a stop signal.

This behavior is called **autoregressive** — each new token is influenced by all previous tokens, including the ones the model just generated.

---

## The Context Window

Every LLM has a **context window** — the maximum number of tokens it can process at once (both input and output combined).

| Model Era | Typical Context Window |
|---|---|
| Early models (GPT-2) | ~1,000 tokens |
| GPT-3 | ~4,000 tokens |
| Modern models (GPT-4o, Claude 3.5) | 128,000 – 1,000,000+ tokens |

**Why this matters for RAG:** When the retriever adds documents to the prompt, it consumes context window space. The retriever must be selective — returning only what is most relevant to stay within limits and keep costs reasonable.

---

## How LLMs Are Trained

**Pre-training:**
The model sees billions of text samples. For each incomplete sentence, it predicts the next word. Based on how wrong it was, its parameters are adjusted. After trillions of adjustments, it encodes knowledge about language, facts, code, and reasoning.

**What the model learns during training:**
- Factual knowledge present in the training data
- Grammar, style, and writing conventions
- Reasoning patterns
- How to follow instructions

**What the model does NOT learn:**
- Anything published after its training cutoff date
- Private or proprietary information
- Real-time or live data

This is exactly the gap that RAG fills.

---

## Hallucinations — Why They Happen

Hallucinations are when an LLM generates confident but incorrect text.

**Root cause:** The model is designed to generate *probable* text, not *true* text. When asked about something it does not know, it generates what *sounds like* a reasonable answer based on patterns — even if specific facts are wrong.

**Example:** Ask an LLM "Why do carrots improve night vision?" It has seen thousands of articles repeating this myth (originally WWII British propaganda to hide radar technology). It confidently explains the vitamin A connection. The answer sounds authoritative. It is wrong. The model reproduced a pattern it saw frequently — it did not check facts.

**Why RAG reduces hallucinations:** When the augmented prompt contains the actual correct information, the LLM reads it and generates from it. It is much harder to hallucinate when the right answer is sitting in the context.

---

## Temperature — Controlling Randomness

When sampling the next token, a parameter called **temperature** controls how random the selection is:

| Temperature | Behavior | Use case |
|---|---|---|
| `0.0` | Always picks the highest-probability token | Factual QA, RAG |
| `0.3 – 0.7` | Balanced coherence with variation | General assistants |
| `1.0+` | Highly random and creative | Creative writing |

For RAG systems where accuracy is important, lower temperatures (0.0–0.3) keep the model anchored to the retrieved facts.

---

## What LLMs Are Good At vs. Where They Struggle

| LLMs Excel At | LLMs Struggle With |
|---|---|
| Language generation and fluency | Private or recent information |
| Reasoning over text given to them | Knowing when they do not know |
| Summarization and synthesis | Large arithmetic |
| Code generation | Citing sources reliably |
| Following complex instructions | Real-time or changing facts |

This table is also a map of what RAG compensates for. The retriever handles information access. The LLM handles reasoning and generation.

---

## Key Terms

| Term | Definition |
|---|---|
| **Token** | The basic unit of text an LLM processes — sub-word pieces |
| **Context Window** | Maximum tokens the model can handle in a single interaction |
| **Temperature** | Controls randomness in token sampling |
| **Autoregressive** | Each new token depends on all previous ones |
| **Hallucination** | Confident but incorrect text generated by an LLM |
| **Pre-training** | Initial training on internet-scale data |
| **Fine-tuning** | Further training to align behavior with human preferences |
| **Grounding** | Anchoring LLM responses to real, provided information |

---

## What to Carry Forward

- LLMs predict the next token, one at a time — everything emerges from this simple mechanism
- They work with tokens (~0.75 words each), not whole words
- They have a context window that limits how much can be in one prompt
- They hallucinate when they lack reliable training data on a topic
- RAG fixes hallucinations by putting the correct information directly into the prompt
- Temperature controls the creativity vs. accuracy tradeoff

---

*Related: [RAG Architecture →](../rag_architecture/rag_architecture_overview.md) | [Information Retrieval →](../retrieval/information_retrieval_fundamentals.md)*
