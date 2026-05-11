# Module 01 Quiz — RAG Overview

> Test your understanding of Module 1 before the graded quiz or an interview.  
> Try to answer each question before reading the answer.  
> Questions are grouped by type: **Conceptual**, **Application**, and **Technical**.

---

## Part 1 — Conceptual Questions

These test whether you understand the *why* behind RAG.

---

**Q1. What is the primary goal of a RAG system?**

<details>
<summary>Show Answer</summary>

**To improve the quality and accuracy of LLM responses by providing relevant external information at query time.**

RAG does not replace training, reduce token counts as a primary goal, or permanently modify the model. It improves responses by retrieving and injecting relevant information into the prompt before the LLM generates a response.

</details>

---

**Q2. Why do LLMs hallucinate?**

<details>
<summary>Show Answer</summary>

LLMs generate *probable* text, not *true* text. When a model is asked about something that was absent from or rare in its training data, it still generates a confident-sounding response — because that is what the training process taught it to do. It has no mechanism to say "I don't know" unless specifically trained to do so.

Hallucination is not a bug or malfunction. It is what happens when a text-probability model is asked about information outside its knowledge.

</details>

---

**Q3. In what situations is RAG particularly useful? Name at least three.**

<details>
<summary>Show Answer</summary>

RAG is particularly useful when:
1. **Private or proprietary data** — the information was never in the LLM's training data (company documents, internal policies, patient records)
2. **Recent events** — information that post-dates the model's training cutoff
3. **Specialized domains** — niche topics with limited public documentation
4. **Source citation required** — the application needs to show where information came from
5. **Frequently changing information** — knowledge bases can be updated without retraining the model

</details>

---

**Q4. What is the difference between retrieval and generation in a RAG system?**

<details>
<summary>Show Answer</summary>

- **Retrieval** — the process of finding and collecting relevant information from the knowledge base in response to a query. Handled by the retriever.
- **Generation** — the process of reasoning over the retrieved information and producing a coherent, grounded response. Handled by the LLM.

The name RAG maps directly: **R**etrieval → **A**ugmented → **G**eneration.

</details>

---

**Q5. How does RAG reduce the likelihood of hallucinations?**

<details>
<summary>Show Answer</summary>

By placing accurate, relevant information directly in the prompt, RAG gives the LLM correct content to generate from. Instead of generating based on potentially incomplete or inaccurate training patterns, the model reads the retrieved documents and builds its response from that grounded content.

The model still does what it always does — generates probable next tokens — but now the most probable tokens are the ones that accurately describe the retrieved facts, not a hallucination.

</details>

---

**Q6. Why is it easier to keep a RAG system up-to-date compared to retraining an LLM?**

<details>
<summary>Show Answer</summary>

Retraining an LLM requires massive compute resources, weeks of training time, and significant cost. You cannot quickly update an LLM's knowledge.

In a RAG system, the knowledge base is just a database. Updating it is as simple as adding, modifying, or removing documents. Once the new documents are indexed, the system immediately uses the new information. No model retraining required.

</details>

---

## Part 2 — Application Questions

These test whether you can apply RAG concepts to real scenarios.

---

**Q7. A hospital wants to build a system where doctors can ask questions and get answers grounded in the specific patient's history and the latest treatment guidelines. Is RAG a good fit? Explain why.**

<details>
<summary>Show Answer</summary>

Yes, RAG is an excellent fit for several reasons:

1. **Patient data is private** — it was never in any LLM's training data. RAG allows this data to be used as a knowledge base at query time without exposing it during model training.
2. **Medical guidelines change** — the knowledge base can be updated with new guidelines immediately, without retraining the model.
3. **High precision required** — RAG allows the system to cite the specific documents it drew from, enabling doctors to verify the source.
4. **Specialized domain** — medical literature is vast and specialized; an off-the-shelf LLM would not reliably have the specific knowledge needed.

</details>

---

**Q8. A developer asks: "Can't I just train the LLM on my company's data instead of using RAG?" What would you tell them?**

<details>
<summary>Show Answer</summary>

Fine-tuning (training on company data) and RAG solve related but different problems:

- **Fine-tuning** teaches the model a *style*, *format*, or *domain-specific behavior*. It's good for "respond like our company" or "always format answers this way."
- **RAG** makes specific *facts* and *documents* available to the model at query time. It's good for "answer this specific question using this specific document."

Fine-tuning is also expensive, slow, and doesn't update dynamically. If your company data changes, you'd have to fine-tune again.

For most enterprise use cases that require answering questions from internal documents, **RAG is the right choice** — often with some fine-tuning on top for tone and format.

</details>

---

**Q9. A user asks a RAG-powered customer service bot: "What is your return policy?" The retriever finds and returns the return policy document. The LLM then responds with generic information about return policies instead of your company's policy. What is the most likely cause?**

<details>
<summary>Show Answer</summary>

The most likely causes are:

1. **Poor prompt construction** — the augmented prompt may not have clearly instructed the LLM to use the provided context. If the instructions don't emphasize "use the following information," the LLM may default to its training data.
2. **LLM ignoring context** — some models need explicit instruction to prioritize the context over their own knowledge.

Fix: Improve the augmented prompt with clearer instructions, such as: *"Answer ONLY using the provided context. Do not use your general knowledge."*

</details>

---

## Part 3 — Technical Questions

These test your understanding of how LLMs and retrievers work internally.

---

**Q10. What is a token, and why does tokenization matter for RAG?**

<details>
<summary>Show Answer</summary>

A **token** is the fundamental unit an LLM processes — roughly corresponding to sub-word chunks. Common words may be one token; longer or rarer words may be split into multiple tokens. Punctuation often has its own token.

Tokenization matters for RAG because:
- The LLM's **context window** (maximum input size) is measured in tokens
- Every document you retrieve and add to the prompt consumes tokens
- You must ensure the question + retrieved documents + instructions fit within the context window
- Longer prompts cost more and take longer to process

</details>

---

**Q11. When an LLM is generating a response, what does it do to decide the next token?**

<details>
<summary>Show Answer</summary>

1. **Processes the full current text** — the prompt plus all previously generated tokens
2. **Calculates a probability for every token** in its vocabulary (tens of thousands of options)
3. **Randomly samples** the next token from that probability distribution (not always the highest-probability token — this randomness produces varied responses)
4. **Adds that token** to the completion and repeats the process

This is why the same prompt can produce different responses each time — the sampling introduces randomness at each step.

</details>

---

**Q12. The phrase "California is uncommonly beautiful" — approximately how many tokens would this be? Explain your reasoning.**

<details>
<summary>Show Answer</summary>

Approximately **6 tokens**:

- `"California"` → 1 token (common, well-known word)
- `" is"` → 1 token
- `" un"` → 1 token (prefix split off)
- `"comm"` → 1 token
- `"only"` → 1 token
- `" beautiful"` → 1 token

The word "uncommonly" gets split because it's a compound/derivative word. Common short words and common long words may stay as single tokens; less common compound words get split into sub-word pieces.

</details>

---

**Q13. What is the difference between keyword search and semantic search in the context of information retrieval?**

<details>
<summary>Show Answer</summary>

**Keyword search (lexical search):**
- Matches documents based on whether they contain the exact words from the query
- Fast and simple
- Misses synonyms: query "car" won't match document that says "automobile"
- Example algorithm: BM25

**Semantic search (vector search):**
- Converts query and documents into vectors (embeddings) that represent *meaning*
- Finds documents that are conceptually similar even with different words
- "car" and "automobile" would be close in vector space
- Slower to build index, but better at capturing meaning
- May miss exact keyword matches that are clearly relevant

**Hybrid search** combines both and typically gives the best results in practice.

</details>

---

**Q14. Explain the precision-recall trade-off in retrieval. What happens if your retriever returns too many documents? Too few?**

<details>
<summary>Show Answer</summary>

**Precision:** Of the documents returned, what fraction are actually relevant?  
**Recall:** Of all relevant documents in the knowledge base, what fraction were returned?

**Too many documents returned:**
- High recall (you're unlikely to miss relevant documents)
- Low precision (many irrelevant documents are returned)
- The LLM's context window fills with noise
- Prompts become longer, slower, more expensive
- The LLM may be confused by irrelevant content

**Too few documents returned:**
- High precision (returned documents are probably relevant)
- Low recall (you may miss important documents ranked just below the cutoff)
- The LLM may lack information it needs to answer well

The right balance depends on the application. Most systems are tuned through monitoring real user queries and iterating on the retrieval settings.

</details>

---

**Q15. What is the context window, and how does it constrain a RAG system?**

<details>
<summary>Show Answer</summary>

The **context window** is the maximum number of tokens an LLM can process in a single request — it includes the system prompt, instructions, retrieved documents, conversation history, and the user's question combined.

Constraints it creates for RAG:
1. **Limits how many documents can be retrieved** — each document takes tokens
2. **Forces trade-off between coverage and cost** — more documents = closer to the limit = more expensive
3. **Requires smart retrieval** — you must retrieve the *most relevant* documents, not just any documents
4. **Older models** (small context windows) needed very precise retrieval; **newer models** (1M+ token windows) have more flexibility but longer prompts still cost more

</details>

---

## Part 4 — Interview-Style Questions

These are the kind of open-ended questions you might face in an AI engineering interview.

---

**Q16. "Walk me through what happens when a user submits a query to a RAG system."**

<details>
<summary>Show Answer</summary>

Strong answer structure:

1. The user's query arrives at the system
2. The query is sent to the **retriever**, which processes it to understand its meaning
3. The retriever searches the **knowledge base** (often a vector database), scoring documents by relevance to the query
4. The top-N highest-scoring documents are returned
5. The system builds an **augmented prompt** — combining the original query with the retrieved documents and instructions for the LLM
6. The augmented prompt is sent to the **LLM**
7. The LLM generates a response grounded in both its training knowledge and the retrieved context
8. The response is returned to the user

The user sees only their question and the response. Everything in between is the RAG pipeline.

</details>

---

**Q17. "How would you explain RAG to a non-technical stakeholder?"**

<details>
<summary>Show Answer</summary>

Strong non-technical explanation:

*"Think of the AI like a very well-read analyst who has read a lot about the world, but hasn't read your company's specific documents. When you ask them a question about company policy, they might guess — and they'd be wrong.*

*RAG is like giving that analyst a filing cabinet with all your company's documents. Before they answer your question, they look through the filing cabinet, find the relevant documents, and then answer based on what they actually read. Now their answer is grounded in your real information instead of a guess."*

</details>

---

**Q18. "What are the limitations of RAG, and when might it not be the right approach?"**

<details>
<summary>Show Answer</summary>

RAG has real limitations:

1. **Retrieval quality is a bottleneck** — if the right documents aren't retrieved, the LLM has nothing to work with. Bad retrieval = bad answers even with a great LLM.
2. **Knowledge base must be maintained** — it needs to be kept current, properly structured, and relevant. A poorly curated knowledge base hurts performance.
3. **Latency** — the retrieval step adds time before the LLM can respond.
4. **Context window limits** — you can't always retrieve everything that might be relevant.
5. **The LLM must be instructed to use the context** — without proper prompting, the LLM may ignore the retrieved documents.

**When RAG might not be the right approach:**
- The task requires deeply internalizing a particular *style* or *behavior* (fine-tuning may be better)
- The knowledge is small, stable, and could just be in the system prompt
- Real-time performance is critical and retrieval latency is unacceptable
- The task doesn't involve factual grounding (creative writing, general reasoning)

</details>

---

*Good luck on the graded quiz. If you can answer all of these confidently, you're ready.*
