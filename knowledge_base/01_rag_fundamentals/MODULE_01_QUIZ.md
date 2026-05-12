# MODULE 01 QUIZ — RAG Fundamentals and LLM Foundations

> **Module:** 01 — RAG Fundamentals + LLM Foundations  
> **Covers:** All six files in `01_rag_fundamentals/` and `02_llms/`

---

## Module Coverage

This quiz tests understanding across all six knowledge base files for Module 1:

- [`01_rag_fundamentals/01_what_is_rag.md`](./01_what_is_rag.md) — the knowledge gap, two-phase model, augmented prompt
- [`01_rag_fundamentals/02_rag_applications.md`](./02_rag_applications.md) — five industry domains
- [`01_rag_fundamentals/03_rag_architecture.md`](./03_rag_architecture.md) — three components, data flow, five advantages
- [`01_rag_fundamentals/04_why_rag_over_retraining.md`](./04_why_rag_over_retraining.md) — RAG vs fine-tuning, long context, agentic RAG
- [`02_llms/01_how_llms_work.md`](../02_llms/01_how_llms_work.md) — tokens, generation loop, training, autoregression
- [`02_llms/02_llm_limitations.md`](../02_llms/02_llm_limitations.md) — hallucination, context window, knowledge cutoff

Aim to answer every question in your own words before consulting the answer key. The goal is to reconstruct the argument, not to recall a sentence.

---

## Section 1 — Conceptual Questions (6 questions)

**Q1.** Explain in your own words why an LLM trained on trillions of tokens from the open internet still cannot reliably answer questions about your company's internal HR policies — even if those policies are written in plain English.

**Q2.** The hotel analogy in `01_what_is_rag.md` describes three levels of question difficulty. What is the structural property that distinguishes Level 1 questions (answerable from general knowledge) from Level 2 and Level 3 questions? What does this distinction reveal about where RAG adds value?

**Q3.** LLM generation is described as "autoregressive." What does this mean mechanically? Give a concrete example of how autoregression produces coherent output, and explain how the same property contributes to hallucination.

**Q4.** A colleague says hallucination is a bug that will be fixed in the next model version. Explain precisely why this framing is incorrect. What would "fixing" hallucination actually require, and why is that not feasible?

**Q5.** Why does the same prompt submitted twice to the same LLM produce different completions? What parameter controls the degree of this variation, and how does that parameter affect RAG system behavior?

**Q6.** LLM "knowledge" is described as implicit and distributed. What does this mean? Contrast it with how knowledge is stored in a relational database, and explain why the difference matters for the RAG architecture.

---

## Section 2 — Architecture and Design Questions (4 questions)

**Q7.** Draw the end-to-end data flow of a RAG system from memory — from the moment a user submits a prompt to the moment a response is returned. Label every component and every data handoff. What does the user observe versus what happens internally?

**Q8.** The RAG architecture is described as implementing "separation of concerns." What does each component — the retriever and the LLM — contribute, and what does each component *not* do? Why is this division architecturally valuable rather than just conceptually tidy?

**Q9.** An augmented prompt has a specific structure. Describe that structure in full. What are the three elements it must contain, and why does each element matter for the LLM's ability to produce a grounded response?

**Q10.** A team building a RAG system is deciding how many documents to retrieve per query (the top-k parameter). What are the trade-offs of setting k too high versus too low? What two properties of LLMs make this trade-off technically consequential rather than just a performance preference?

---

## Section 3 — Comparative Questions (3 questions)

**Q11.** A startup is building a customer service bot for a rapidly evolving product line. Their CTO proposes fine-tuning a large LLM on the product documentation. Explain specifically why this approach has a structural flaw for this use case. What would a RAG-based alternative look like, and what problem does it solve that fine-tuning cannot?

**Q12.** A researcher argues: "Context windows are now so large that retrieval is unnecessary — just include all the documents in every prompt." Identify at least three concrete problems with this approach, including one that involves a specific empirical finding from the research literature.

**Q13.** RAG reduces hallucination, but does not eliminate it. Distinguish between the two types of hallucination described in `02_llm_limitations.md`. Which type does RAG directly address, and which does it not? What does addressing the second type require?

---

## Section 4 — Application Questions (2 questions)

**Q14.** A law firm wants to deploy an LLM-based assistant to help attorneys search case law and draft initial responses to client queries. The case law database is updated daily; many documents are confidential. Describe the RAG architecture you would recommend: what goes in the knowledge base, what the retriever must handle, and why the two main alternatives (fine-tuning, long-context) would both fail for this use case.

**Q15.** A software team is considering an agentic RAG system for a complex enterprise workflow. Describe in concrete terms how agentic RAG differs from classical RAG. Give one specific example of a query where an agentic system would outperform a classical RAG system, and explain what capability makes the difference.

---

---

## Answer Key

---

### A1. Why an LLM Cannot Answer Questions About Private HR Policies

The LLM's knowledge comes entirely from its training data. Internal HR policies are private documents — they were never published on the internet, never included in any public corpus, and therefore never encountered during training. The model has zero exposure to these documents. It does not know they exist, let alone what they say.

More fundamentally: even if the policies were written in perfectly clear English, that clarity is irrelevant. The model cannot learn what it was never shown. No amount of scale or capability changes this — a model trained on the public internet has a knowledge boundary defined by the public internet.

RAG solves this by putting the HR policy documents in the knowledge base. When an employee asks a policy question, the retriever finds the relevant policy sections and places them in the augmented prompt. The LLM reads the policy in context and reasons from it — a task it does well. The model does not need to have memorized the policy; it needs only to read it and apply it.

---

### A2. The Structural Property Distinguishing the Three Question Levels

The distinguishing property is whether answering the question requires **information external to the LLM's training distribution**. Level 1 questions (why are hotels expensive on weekends?) can be answered from general world knowledge that was widely represented in training data. Level 2 questions (why are hotels in Vancouver expensive *this* weekend?) require current, event-specific information that postdates or was too specific to appear in training. Level 3 questions require deep specialized or private knowledge that may have been entirely absent from training.

The practical implication: RAG adds value specifically when the required knowledge is private, recent, or specialized — the exact conditions that define Levels 2 and 3. For Level 1 questions, a bare LLM may be sufficient.

---

### A3. Autoregressive Generation: Coherence and Hallucination

Autoregressive means that each generated token is conditioned on all previously generated tokens, including those the model itself just produced. When the model generates *shining* as the next word in *"the sun is ___"*, the next step conditions on *"the sun is shining"* — making *in the sky* a natural continuation.

This produces coherence: early choices shape later ones into a consistent direction. A paragraph about a summer day remains about a summer day because each new sentence is conditioned on a context full of summer imagery.

The same property produces hallucination when the initial direction is wrong. If the model, lacking training signal for a question, generates a plausible-sounding but incorrect first claim, subsequent tokens are conditioned on that incorrect claim — leading to a confident, internally consistent, but factually wrong response. The model "commits" to an incorrect path and follows it where it leads.

---

### A4. Why Hallucination Is Not a Bug

Hallucination is not a malfunction — it is a direct consequence of the model's design objective. LLMs are trained to maximize next-token probability: to generate the most statistically likely continuation of a sequence given training data. This objective has no explicit truth term. There is no loss function component that penalizes factually incorrect output; there is only a component that penalizes low-probability output.

Within the training distribution — topics well-represented in training data — statistically probable text is usually factually accurate, because accurate descriptions appear repeatedly in training. Outside the training distribution, probability and truth decouple. The model generates what sounds right, not what is right.

"Fixing" hallucination in the traditional bug-fix sense would require changing the training objective or training data in a way that guarantees truth alignment. The former is an active research problem in AI alignment and safety. The latter requires knowing, for every possible query the model will ever encounter, whether the training data provides sufficient and accurate signal — an intractable condition. RAG is the practical engineering response: instead of solving an intractable training problem, provide the model with ground-truth information at inference time.

---

### A5. Stochastic Sampling and Temperature

The same prompt produces different completions because the model samples the next token from a probability distribution rather than always selecting the most probable token. Sampling introduces randomness: even with high probability, *shining* is not selected 100% of the time — and downstream token choices then diverge based on whatever was selected first.

The **temperature** parameter controls the shape of the distribution. At temperature 0, the model greedily selects the most probable token at every step, producing deterministic output. At temperature 1, sampling follows the model's learned distribution. At temperatures above 1, the distribution is flattened, making lower-probability tokens more likely to be selected.

In RAG systems, lower temperatures are generally preferred. When the model has been given specific context to reason from — the retrieved documents — you want it to follow that context faithfully. High temperature encourages the model to wander toward low-probability completions, which may depart from the provided context and increase the risk of confabulation. Most production RAG deployments use temperatures in the 0.0–0.3 range.

---

### A6. Implicit Distributed Knowledge vs. Explicit Database Storage

When an LLM "knows" that Paris is the capital of France, this knowledge is not stored as a row in a table with columns `city` and `country`. It is encoded implicitly across billions of numerical weights — manifesting as increased probability for the token *Paris* following contexts like *"the capital of France is ___"*. The knowledge is distributed, contextual, and pattern-based.

A relational database stores facts explicitly: a specific row, with specific values, in a specific table. You can retrieve it directly, update it in place, or delete it. The storage is transparent, auditable, and editable.

For RAG, this distinction is architectural. An LLM's implicit knowledge cannot be surgically updated — you cannot change "the capital of France" to something else without retraining. A knowledge base stores facts explicitly and can be updated at any time. This is why RAG externalizes factual knowledge from model weights into a managed corpus: it converts the intractable (retraining) into the tractable (database operations).

---

### A7. End-to-End RAG Data Flow

```
User submits: "Why are hotels in Vancouver expensive this weekend?"
                        │
                        ▼
              ┌──────────────────┐
              │    Retriever     │  ← receives the query
              └──────────────────┘
                        │ queries
                        ▼
              ┌──────────────────────────────┐
              │       Knowledge Base         │
              │  (indexed document database) │
              │  scores all documents by     │
              │  relevance; returns top-k    │
              └──────────────────────────────┘
                        │ returns retrieved documents
                        ▼
              ┌──────────────────────────────┐
              │      Augmented Prompt        │
              │  = original query            │
              │  + retrieved doc 1           │
              │  + retrieved doc 2 ...       │
              └──────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │       LLM        │ → Response
              └──────────────────┘
```

The user observes: they submit a question and receive a response, possibly with slightly more latency than a bare LLM. Everything between — retrieval, scoring, augmentation — is invisible. The user experience is identical to a bare LLM interaction.

---

### A8. Separation of Concerns

The **retriever** is responsible for finding relevant information: querying the knowledge base, scoring document relevance, filtering down a potentially enormous corpus to the few documents most pertinent to the current query. It does not generate text, reason over content, or produce responses.

The **LLM** is responsible for synthesis and reasoning: reading the augmented prompt, understanding the user's question, reasoning over the retrieved context, and producing a coherent, accurate response. It does not search databases or evaluate document relevance.

This division is architecturally valuable for three reasons. First, each component is used for the task it is best at — retrieval systems are optimized for fast, accurate search; LLMs are optimized for language understanding and generation. Forcing the LLM to "remember" facts from training is asking it to do something it does unreliably; asking it to reason over provided context is what it does best. Second, failures are diagnosable: when a RAG system produces a bad answer, you can determine whether the retriever returned irrelevant documents (retrieval failure) or the LLM failed to reason correctly over good documents (generation failure). Third, each component can be improved independently — a better retriever improves the system without touching the LLM and vice versa.

---

### A9. Structure of the Augmented Prompt

The augmented prompt has three elements:

1. **A system instruction** telling the LLM to use the provided context and to indicate when context is insufficient. This grounds the model's behavior — without explicit instruction, the model may ignore the retrieved documents and rely on training-time associations instead.

2. **The user's original question**, unchanged. The LLM needs to know what is being asked.

3. **The retrieved documents**, with source attribution where possible. This is the factual grounding that prevents hallucination and enables citation. The documents must be present verbatim (not paraphrased) so the LLM can reason from specific content rather than a summary.

Each element is necessary. Without the instruction, the model may hallucinate despite having context. Without the original question, the model lacks a target. Without the retrieved documents, it is just a bare LLM prompt.

---

### A10. Trade-offs of High vs. Low k in Retrieval

Setting k too **low** (too few retrieved documents) risks **recall failures**: the most relevant document may not be among the top-ranked results, and the LLM generates a response without the specific information it needs. This is particularly costly in precision-critical domains.

Setting k too **high** (too many retrieved documents) creates two problems rooted in LLM mechanics. First, longer prompts are computationally expensive due to O(n²) attention scaling — adding many documents increases cost and latency nonlinearly. Second, the "lost in the middle" problem (Liu et al., 2023) means the LLM attends well to information at the beginning and end of a long context but poorly to information buried in the middle — so adding more documents does not guarantee the relevant content is used. Worse, irrelevant documents introduce noise that can degrade response quality.

The practical consequence: k is a tunable hyperparameter that must be calibrated for each deployment. It cannot be set arbitrarily high as a safety measure.

---

### A11. Fine-Tuning Fails for Rapidly Evolving Product Documentation

Fine-tuning bakes knowledge into static model weights. Once the fine-tuning run ends, the model's knowledge of the product line is frozen at the training data cutoff. When new products launch, specifications change, or policies are updated, the model does not know. The only remedy is re-fine-tuning — collect the new data, run another training cycle, evaluate, deploy. This cycle takes weeks to months and costs significant compute. For a rapidly evolving product line, knowledge will be stale almost immediately.

The structural flaw: the update cadence of the product line (potentially weekly) far exceeds the fine-tuning update cadence (months). The system will consistently be out of date.

A RAG-based alternative: maintain a knowledge base of product documentation, specifications, inventory data, and FAQs. When documentation changes, update the knowledge base — a database operation. The next query to the system immediately benefits from the updated information. No retraining, no redeployment of the model.

---

### A12. Three Problems with "Just Use a Large Context Window"

**Problem 1 — Cost.** LLM inference cost scales approximately with the square of the sequence length (O(n²) for self-attention). Including all documents in every prompt, at production query volumes, is financially prohibitive. RAG's selective retrieval keeps prompts concise and cost-effective.

**Problem 2 — The "lost in the middle" problem.** Liu et al. (2023) demonstrated empirically that LLMs attend well to information at the beginning and end of long contexts but systematically underutilize information positioned in the middle. A knowledge base included wholesale in every prompt does not guarantee that the relevant sections will be used — they may be effectively invisible to the model depending on their position.

**Problem 3 — Latency.** Longer prompts take longer to process. In interactive applications requiring sub-second or low-second response times, prompts of millions of tokens are not viable.

Large context windows do not eliminate the value of retrieval — they change the optimal settings for it (larger chunks, more documents can be included per query). The retriever's role in filtering relevance and managing cost remains architecturally essential.

---

### A13. Two Types of Hallucination and RAG's Coverage

**Extrinsic hallucination:** The model introduces information not present in or verifiable against the provided context. It confabulates facts from training-data associations when context is absent or insufficient. RAG directly addresses this: when relevant documents are in the augmented prompt, the model has a reliable factual basis and is less likely to confabulate.

**Intrinsic hallucination:** The model contradicts or distorts the context that was explicitly provided. It generates a response that conflicts with the retrieved documents — ignoring, misreading, or inverting their content. RAG does not directly address this. The documents are present; the model fails to use them correctly.

Addressing intrinsic hallucination requires prompt engineering (clear instructions to use the provided context and flag uncertainty), model selection (models with stronger instruction-following and context-adherence), and evaluation (testing the system on cases where the correct answer is present in the retrieved documents and checking whether the model uses it). These are the subjects of `04_Advanced_Prompt_Engineering.md` and `05_Evaluation_and_Monitoring.md`.

---

### A14. RAG Architecture for a Legal Knowledge Assistant

**Knowledge base:** The corpus should include the firm's full case law database (statutes, precedents, court filings), updated daily as new rulings are published. Confidential client documents should be included with appropriate access controls — segmented by client or matter to prevent cross-contamination. Source metadata (case name, jurisdiction, date) should be indexed alongside document content to enable citation.

**Retriever requirements:** The retriever must handle both semantic search (finding conceptually relevant case law even when terminology differs) and keyword search (finding cases with specific legal citations or party names). A hybrid retriever combining dense and sparse methods is appropriate. It must also handle access control — only returning documents the querying attorney is permitted to see.

**Why fine-tuning fails:** Legal knowledge changes daily with new rulings. Fine-tuning cannot keep up; the model would be immediately stale. Confidential documents cannot safely be included in a training run.

**Why long-context fails:** Legal corpora are enormous. Including the full case law database in every prompt is computationally infeasible. The "lost in the middle" problem is particularly dangerous here — missing a relevant precedent because it was buried in a long context could constitute professional error.

**RAG's advantage:** The knowledge base is updated as new rulings are published. Retrieved documents are the sources for cited responses. Confidential documents never leave the controlled knowledge base environment.

---

### A15. Agentic RAG vs. Classical RAG

In **classical RAG**, a human engineer designs a fixed retrieval strategy: which knowledge base to query, how many documents to retrieve, how the prompt is structured. This strategy is applied identically to every query, regardless of what the query actually needs.

In **agentic RAG**, an AI agent makes those decisions dynamically at query time. The agent receives the query, decides whether retrieval is needed, chooses which source to query (internal database, web, code repository), evaluates whether the first retrieval round was sufficient, and if not, reformulates the query and retrieves again. Multiple specialized agents may collaborate, each handling a different part of a complex workflow.

**Concrete example:** A user asks, *"Summarize the current regulatory status of mRNA vaccine approvals in the EU and compare it to our internal compliance documentation."* A classical RAG system would retrieve from one knowledge base with a fixed query. An agentic system would: (1) query an external news/regulatory database for current EU vaccine approval status, (2) query the internal compliance knowledge base for the company's documentation, (3) evaluate whether the retrieved information from both sources is sufficient to support a comparison, (4) if not, run targeted follow-up queries on specific regulatory bodies or documents, and (5) synthesize across sources in the final prompt.

The capability that makes the difference is **dynamic decision-making about retrieval strategy**: the agent can choose its sources, evaluate its results, and iterate — rather than executing a single fixed retrieval plan designed in advance.

---

*End of Module 01 Quiz*