# How LLMs Work — Tokens, Autoregressive Generation, and the Nature of Learned Knowledge

> **Series:** AI_Engineer-RAG Knowledge Base  
> **Module:** 01 — LLM Foundations  
> **File:** 1 of 2 in `02_llms/`  
> **Prerequisites:** None — can be read independently, but benefits from [`01_rag_fundamentals/01_what_is_rag.md`](../01_rag_fundamentals/01_what_is_rag.md)

---

## The Core Idea

An LLM is a statistical model of language. At every step, it does one thing: compute a probability distribution over its entire vocabulary and sample the next token. Repeat until done. Everything else — coherent paragraphs, accurate code, nuanced reasoning — emerges from applying this single operation billions of times, guided by patterns learned from trillions of tokens of training data.

## The Problem It Solves

Understanding how LLMs work mechanically is not academic. It directly explains why RAG works (LLMs reason well over context), why hallucination happens (the model generates probable text, not verified truth), why the same prompt produces different answers (stochastic sampling), and why context window size is a hard constraint. Each of RAG's design choices is a response to a mechanical property of LLMs. Knowing the mechanism means understanding the system, not just operating it.

---

## Tokens and Vocabulary

LLMs do not process words — they process **tokens**, which are variable-length subword units produced by algorithms such as Byte Pair Encoding (BPE). Common words like *London* or *door* may map to a single token. Longer or rarer words like *programmatically* or *unfalsifiable* are split into multiple tokens. Punctuation, spaces, and special characters also receive tokens.

Most LLMs maintain a vocabulary of 10,000 to 100,000+ tokens. Subword tokenization is what allows a finite vocabulary to represent any word in any language: unfamiliar compound words are assembled from familiar smaller pieces. The model never encounters an unknown token; it encounters an unfamiliar combination of known ones.

Think of it like Scrabble tiles. No matter what word appears in a sentence, you can spell it from the available set of letter tiles. The grammar emerges from the arrangement; the vocabulary is just the finite set of pieces available.

---

## The Generation Loop

When an LLM generates text, it executes the same procedure for each new token:

```
Current state of text (prompt + all previously generated tokens)
        │
        ▼
┌──────────────────────────────────────────────┐
│  LLM processes full context via self-attention│
│  Builds deep representation of meaning        │
│  and relationships across all tokens          │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  For every token in vocabulary (10k–100k+):  │
│  compute P(token | all preceding tokens)      │
│  → Probability distribution over vocabulary  │
└──────────────────────────────────────────────┘
        │
        ▼
   Sample next token from distribution
        │
        ▼
   Append token to text
        │
        ▼
   Repeat until end-of-sequence token is generated
```

For the prompt *"what a beautiful day, the sun is ___"*, the model might assign: *shining* → 80%, *rising* → 10%, *out* → 5%, *warming* → 3%, *exploding* → 0.001%. It then samples from this distribution. Most of the time it picks *shining*; occasionally it picks *rising*; very rarely it picks something unexpected.

---

## Autoregressive Behavior and Its Consequences

The generation process is **autoregressive**: each generated token is appended to the context and becomes part of the input for the next step. Earlier choices constrain later ones. If the model generates *warming* instead of *shining*, subsequent tokens will be conditioned on *warming* — leading to a completion like *"the sun is warming our faces"* rather than *"the sun is shining in the sky"*.

Two consequences follow directly:

**The same prompt produces different completions.** Because sampling is stochastic, running the same prompt twice through the same model produces different outputs. This is not a bug — it is the mechanism that enables fluent, non-repetitive generation. But it also means LLM outputs are not deterministic and cannot be tested like traditional software.

**Early token choices propagate through the rest of the completion.** A model that starts down an incorrect path tends to continue down it, because each subsequent token is conditioned on all previous ones. This amplifies small early errors into large downstream ones — a key contributor to hallucination.

---

## Temperature: Controlling the Distribution

The **temperature** parameter controls how peaked or flat the probability distribution is at each sampling step:

- **Temperature = 0:** Always sample the most probable token (greedy). Deterministic, but potentially repetitive and less creative.
- **Temperature = 1:** Sample directly from the model's learned distribution. The baseline.
- **Temperature > 1:** Flatten the distribution — less probable tokens are sampled more often. Increases creativity and diversity, at the cost of coherence.

In RAG systems, lower temperatures are often preferred: when the model has been given specific context to reason from, you want it to follow that context faithfully rather than wander creatively away from it.

---

## Training: How the Model Learns

Before training, an LLM's billions of numerical parameters are initialized randomly — the model produces gibberish. Training is an iterative process of showing the model incomplete text sequences and asking it to predict what comes next:

1. Show the model a sequence of tokens from the training corpus with the last token hidden.
2. The model produces a probability distribution over all possible next tokens.
3. Compare the model's prediction to the actual next token; compute a loss.
4. Backpropagate: update the parameters to make the correct token more probable.
5. Repeat across trillions of tokens.

Over this process, the model's parameters gradually encode the statistical patterns of language: which words follow which, what sequences are coherent, what facts about the world appear consistently across many contexts. Modern LLMs are trained on trillions of tokens drawn from the open internet, books, code repositories, scientific papers, and other sources.

---

## What "Knowledge" Means for an LLM

The knowledge an LLM accumulates during training is **implicit and distributed**. It is not stored as a database of facts that can be queried or updated. Instead, facts that appeared frequently and consistently in the training data get encoded across the billions of parameters of the network — manifesting as increased probability for certain token sequences in relevant contexts.

This has two important implications:

**Frequency matters.** A fact mentioned thousands of times across training data is reliably encoded. A fact mentioned once, in an obscure document, may not be reliably accessible. This is why LLMs are strong on general knowledge and unreliable on niche or rare topics.

**Knowledge cannot be surgically updated.** You cannot go into an LLM's parameters and change a fact the way you would update a database row. The only way to change what a model knows is to retrain it. This is the core motivation for keeping factual knowledge in a knowledge base rather than in model weights.

The analogy: an LLM's "knowledge" is like a person's intuitive expertise after reading widely for years — fluid, contextual, hard to audit, impossible to update with a specific fact without sustained practice. A knowledge base is like a well-maintained reference library — auditable, updatable, citable.

---

## Key Terms

| Term | Definition |
|---|---|
| **Token** | The basic unit of text processed by an LLM; roughly a word piece, produced by subword tokenization algorithms such as BPE. |
| **Vocabulary** | The complete set of tokens an LLM recognizes; typically 10,000–100,000+ tokens. |
| **Byte Pair Encoding (BPE)** | A subword tokenization algorithm that iteratively merges frequent character pairs; produces a finite vocabulary that can represent any text. |
| **Next-token prediction** | The training objective of language models: predict the next token given all preceding tokens. |
| **Autoregressive** | A generation property where each output token is conditioned on all previously generated tokens; earlier choices shape all later ones. |
| **Probability distribution** | At each generation step, the LLM computes a probability for every token in its vocabulary; the next token is sampled from this distribution. |
| **Temperature** | A parameter that controls the flatness of the sampling distribution; low temperature = deterministic/peaked; high temperature = random/flat. |
| **Stochasticity** | The randomness inherent in token sampling; the reason identical prompts produce different completions. |
| **Context window** | The maximum number of tokens the LLM can process in a single forward pass; the hard limit on input length. |
| **Parameters (weights)** | The billions of numerical values learned during training that collectively define the model's behavior and encoded knowledge. |
| **Implicit knowledge** | Knowledge encoded across model parameters through training; not stored as explicit facts but as statistical patterns over token sequences. |

---

## What to Carry Forward

- LLMs are next-token predictors: at each step they compute a probability distribution over the vocabulary and sample from it. Everything else — reasoning, knowledge, creativity — emerges from applying this operation at scale.
- Generation is autoregressive: each token choice conditions all future choices. This creates coherent completions and propagates early errors downstream.
- Temperature controls the distribution's shape: lower temperature for grounded, deterministic behavior; higher for creative, exploratory generation.
- Training encodes knowledge implicitly across billions of parameters. Knowledge is not stored in an auditable or updatable form — it is distributed through the network as statistical associations.
- "Knowledge" for an LLM means high probability for certain token sequences in certain contexts. Facts that appeared frequently in training are reliable; rare or absent facts are not.
- These mechanical properties explain both why RAG works (in-context reasoning over provided text) and why it is needed (knowledge gaps lead to hallucination).

---

## Navigation

- **Previous:** [`01_rag_fundamentals/03_rag_architecture.md`](../01_rag_fundamentals/03_rag_architecture.md) — the RAG architecture that the LLM operates within.
- **Next:** [`02_llm_limitations.md`](./02_llm_limitations.md) — the three structural limitations that RAG is designed to address.
- **Related:** [`01_rag_fundamentals/01_what_is_rag.md`](../01_rag_fundamentals/01_what_is_rag.md) — why the LLM's knowledge gap motivates the RAG architecture.
- **Academic depth:** `01_RAG_Overview.md` — Section 2 (What Is a Large Language Model?). Foundational paper on Transformer architecture: Vaswani et al. (2017), [`arXiv:1706.03762`](https://arxiv.org/abs/1706.03762). GPT-3 in-context learning: Brown et al. (2020), [`arXiv:2005.14165`](https://arxiv.org/abs/2005.14165).