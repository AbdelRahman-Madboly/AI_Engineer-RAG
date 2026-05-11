# Part 2: Building the RAG Pipeline — Data, Prompts, and LLM Calls

> **Series:** Retrieval-Augmented Generation — From Foundations to Production  
> **Part:** 2 of 6  
> **Level:** Foundational–Intermediate  
> **Prerequisites:** Part 1 (RAG Overview); basic Python familiarity (strings, lists, dicts, functions)  
> **Practical Labs:** `00_Python_RAG_Prep.ipynb` · `01_LLM_Calls_and_Augmented_Prompts.ipynb`

---

## Table of Contents

1. [Introduction: From Concept to Code](#1-introduction-from-concept-to-code)
2. [The Environment: Setting Up for RAG Development](#2-the-environment-setting-up-for-rag-development)
3. [Python Foundations for RAG Engineering](#3-python-foundations-for-rag-engineering)
4. [The LLM API Interaction Model](#4-the-llm-api-interaction-model)
5. [Prompt Engineering for RAG](#5-prompt-engineering-for-rag)
6. [Constructing the Augmented Prompt](#6-constructing-the-augmented-prompt)
7. [Building a Minimal RAG Pipeline in Python](#7-building-a-minimal-rag-pipeline-in-python)
8. [The Retrieve → Augment → Generate Loop in Detail](#8-the-retrieve--augment--generate-loop-in-detail)
9. [Common Failure Modes and How to Diagnose Them](#9-common-failure-modes-and-how-to-diagnose-them)
10. [Key Concepts Summary](#10-key-concepts-summary)
11. [Further Reading and Foundational Papers](#11-further-reading-and-foundational-papers)
12. [Review Questions](#12-review-questions)

---

## 1. Introduction: From Concept to Code

Part 1 established the conceptual architecture of RAG: a retriever fetches relevant documents, a prompt augmentation step injects them alongside the user's query, and an LLM synthesizes a grounded response. The architecture is elegant in theory, but building it demands a set of practical engineering skills that are just as important as the conceptual understanding.

Part 2 bridges that gap. We move from the whiteboard to the keyboard, examining:

- **How to structure a Python development environment** for LLM-based projects;
- **The data structures and language features** that appear repeatedly throughout every RAG codebase;
- **The LLM API interaction model** — how modern language models receive input and produce output via an API, and what the standardized message format looks like;
- **Prompt engineering for RAG** — the craft of constructing prompts that reliably elicit accurate, grounded, and well-formatted responses from an LLM;
- **A complete, working, minimal RAG pipeline** built step-by-step from scratch.

By the end of this part, you will have built a functioning RAG system — simple but complete — that represents the skeleton on which every subsequent part of this series builds.

---

## 2. The Environment: Setting Up for RAG Development

### 2.1 Why Environment Management Matters

LLM projects involve multiple dependencies — API clients, numerical libraries, environment variable management, and version-sensitive packages. Without disciplined environment management:

- Package version conflicts corrupt your setup silently;
- API keys are accidentally committed to version control (a serious security failure);
- Code that works on your machine breaks in deployment.

Professional RAG development addresses all three concerns from the start.

### 2.2 The `.env` File Pattern

API keys and configuration values should never be hard-coded in source files. The standard practice in Python development is to store them in a `.env` file (a plain text file of `KEY=VALUE` pairs) and load them at runtime using the `python-dotenv` library.

```
# .env — stored at the project root; NEVER committed to git
LLM_BACKEND=ollama          # which LLM backend to use
OLLAMA_MODEL=qwen2.5:7b     # which model to load via Ollama
GEMINI_API_KEY=your_key     # Google Gemini key (if using Gemini backend)
TOGETHER_API_KEY=your_key   # Together.ai key (if using Together backend)
```

In Python, these values are loaded with:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env and populates os.environ

backend = os.environ.get('LLM_BACKEND', 'ollama')  # 'ollama' is the default
api_key = os.environ.get('TOGETHER_API_KEY', '')
```

> **Security principle:** Add `.env` to your `.gitignore` file immediately when starting any project that uses API keys. A single accidental commit of an API key to a public repository can result in significant financial and security harm.

### 2.3 Supported LLM Backends

A well-engineered RAG course environment abstracts away backend differences behind a unified interface. The course `utils.py` supports three backends:

| Backend | Description | Requires |
|---|---|---|
| **Ollama** | Runs LLMs locally on your machine | `ollama serve` running; model pulled |
| **Gemini** | Google's Gemini API (cloud) | `GEMINI_API_KEY` in `.env` |
| **Together.ai** | Cloud inference for open-source models | `TOGETHER_API_KEY` in `.env` |

The active backend is selected by setting `LLM_BACKEND` in `.env`. The rest of your code — including all prompts and pipeline logic — remains identical regardless of which backend is active. This is the **abstraction principle** at work: changes in infrastructure should not require changes in application logic.

### 2.4 Verifying Your Setup

A reliable setup verification checks three things:

```python
import sys
from dotenv import load_dotenv
import os

# 1. Python version
print(f"Python version: {sys.version}")
assert sys.version_info >= (3, 11), "Python 3.11+ is required"

# 2. Active backend
load_dotenv()
backend = os.environ.get('LLM_BACKEND', 'ollama')
print(f"Active backend: {backend}")

# 3. API key or host
if backend == 'together':
    key = os.environ.get('TOGETHER_API_KEY', '')
    print("Key loaded" if key else "ERROR: TOGETHER_API_KEY missing")
elif backend == 'ollama':
    host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    print(f"Ollama host: {host}")
```

---

## 3. Python Foundations for RAG Engineering

RAG systems are built on a small set of Python constructs that appear constantly. Mastering these five patterns will make every subsequent lab immediately readable.

### 3.1 Lists: The Universal Container for Retrieved Documents

Every retrieved document collection is a Python list. Lists are the native data structure for representing an ordered, mutable sequence of items — exactly the shape of "the top-k documents returned by the retriever."

```python
# A knowledge base — the simplest form: a list of strings
knowledge_base = [
    "RAG stands for Retrieval Augmented Generation.",
    "LLMs have a training cutoff date and cannot access new information.",
    "Vector databases store embeddings for fast similarity search.",
    "Chunking splits long documents into smaller retrievable pieces.",
    "The retriever scores documents by relevance to the query.",
]

# Retrieve the top-3 (simulated here by slicing)
top_3 = knowledge_base[:3]
print(f"Retrieved {len(top_3)} documents")
```

**Key operations used throughout RAG pipelines:**

| Operation | Syntax | Use case |
|---|---|---|
| Access by index | `docs[0]` | First retrieved document |
| Slice top-k | `docs[:k]` | Limit retrieval count |
| Append | `docs.append(d)` | Add a single document |
| Extend | `docs.extend(more)` | Merge two document lists |
| Length | `len(docs)` | Count retrieved documents |

### 3.2 List Comprehensions: Transforming Document Collections

List comprehensions provide a concise, readable syntax for creating a new list by transforming or filtering an existing one. In RAG, they are used constantly for formatting document batches.

**Standard for-loop version:**
```python
labeled_docs = []
for i, doc in enumerate(knowledge_base[:3]):
    labeled_docs.append(f"[Doc {i+1}] {doc}")
```

**Equivalent list comprehension (preferred):**
```python
labeled_docs = [f"[Doc {i+1}] {doc}" for i, doc in enumerate(knowledge_base[:3])]
```

**Conditional filtering** (keeping only relevant documents after a keyword pre-filter):
```python
rag_docs = [doc for doc in knowledge_base if "RAG" in doc]
```

**Why list comprehensions matter:** They reduce syntactic noise and make the *intent* of the transformation immediately apparent. In a RAG pipeline that processes hundreds of documents, code clarity is not cosmetic — it directly affects maintainability.

### 3.3 Dictionaries: The Native Format for LLM Messages

Every message exchanged with a modern LLM API is a Python dictionary with exactly two keys:

```python
message = {
    "role":    "user",       # who is speaking: 'system', 'user', or 'assistant'
    "content": "What is RAG?"  # the actual text content
}
```

This `{role, content}` format is standardized across OpenAI, Anthropic, Together.ai, Google Gemini (with minor adaptation), and Ollama. Understanding it deeply is foundational to all LLM programming.

**Safe key access with `.get()`:**

When parsing API responses — which may have optional fields — use `.get()` instead of direct bracket access to avoid `KeyError` exceptions:

```python
api_response = {
    "role":        "assistant",
    "content":     "RAG stands for Retrieval Augmented Generation.",
    "model":       "qwen2.5:7b",
    "tokens_used": 42
}

# Safe access — returns None if key doesn't exist
finish_reason = api_response.get("finish_reason", "unknown")

# Iteration over all key-value pairs
for key, value in api_response.items():
    print(f"  {key}: {value}")
```

**Building a retrieved document with metadata as a dict:**

```python
retrieved_doc = {
    "id":     1,
    "score":  0.95,
    "text":   "RAG retrieves relevant documents before generating a response.",
    "source": "intro_to_rag.pdf",
    "page":   3
}
```

Storing retrieval scores and source metadata alongside document text enables citation generation and relevance diagnostics — critical features in production RAG systems.

### 3.4 f-Strings: Dynamic Prompt Construction

Every augmented prompt in a RAG system is constructed dynamically at runtime by interpolating a user query and retrieved documents into a template. Python's f-string syntax is the standard tool for this.

**Simple interpolation:**
```python
user_name = "AbdelRahman"
topic     = "RAG"
greeting  = f"Hello, {user_name}! Today we are studying {topic}."
```

**Multi-line f-string for a RAG prompt template:**
```python
question = "What is retrieval augmented generation?"
context  = "RAG is a technique that improves LLM responses by retrieving relevant documents."

prompt = f"""Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:"""
```

The triple-quote multi-line f-string is the idiomatic way to write prompt templates in Python. It preserves whitespace and newlines exactly as written, which matters because LLMs are sensitive to prompt formatting.

**Building the context block from a list of documents:**
```python
retrieved_docs = [
    {"id": 1, "score": 0.95, "text": "RAG retrieves relevant documents before generating."},
    {"id": 2, "score": 0.88, "text": "The retriever searches a vector database."},
    {"id": 3, "score": 0.81, "text": "The LLM reads the retrieved docs and generates an answer."},
]

# Build a formatted context block
context_lines = [
    f"[Document {doc['id']} | relevance: {doc['score']:.2f}]\n{doc['text']}"
    for doc in retrieved_docs
]
context_block = "\n\n".join(context_lines)
```

Output:
```
[Document 1 | relevance: 0.95]
RAG retrieves relevant documents before generating.

[Document 2 | relevance: 0.88]
The retriever searches a vector database.

[Document 3 | relevance: 0.81]
The LLM reads the retrieved docs and generates an answer.
```

This structured formatting is not arbitrary — it directly affects LLM performance. Clearly delineated document boundaries help the model parse which information came from which source, enabling more accurate and citable responses.

### 3.5 Functions with Type Hints

Course utility functions use Python **type hints** to document expected input and output types. These are annotations, not runtime enforcement — but they serve as machine-readable documentation that makes complex codebases dramatically easier to read.

```python
from typing import List, Dict, Optional

def format_documents(docs: List[Dict], max_docs: int = 3) -> str:
    """
    Takes a list of document dicts and returns a single formatted string
    suitable for injection into a RAG prompt.

    Args:
        docs:     List of dicts, each with 'id', 'score', and 'text' keys.
        max_docs: Maximum number of documents to include (default: 3).

    Returns:
        A formatted string ready to insert into a prompt.
    """
    top_docs = docs[:max_docs]
    lines = [f"[Doc {doc['id']}] {doc['text']}" for doc in top_docs]
    return "\n".join(lines)
```

Reading type hints in utility functions immediately tells you:
- What types of arguments the function expects;
- What type of value it returns;
- Which parameters are optional (those with `Optional[T]` or defaults).

---

## 4. The LLM API Interaction Model

### 4.1 The Chat Completions Standard

Modern LLM APIs — regardless of provider — share a common interaction model based on the **chat completions** paradigm, originally introduced by OpenAI and now widely adopted. The key insight is that LLM interactions are modeled not as single prompt-response pairs, but as **conversations**: ordered sequences of messages, each attributed to a specific **role**.

The three roles in the chat completions model are:

| Role | Description | When to Use |
|---|---|---|
| `system` | Instructions that frame the model's behavior and persona | Beginning of every conversation; sets context, constraints, and role |
| `user` | The human's input — questions, commands, content to process | Each turn of the human's contribution |
| `assistant` | The model's prior responses | For multi-turn conversations, to maintain context |

A typical message list for a RAG query looks like:

```python
messages = [
    {
        "role":    "system",
        "content": (
            "You are a precise and helpful research assistant. "
            "Answer questions using only the provided context. "
            "If the answer is not in the context, say 'I cannot find this in the provided documents.' "
            "Always cite the document number you used."
        )
    },
    {
        "role":    "user",
        "content": f"Context:\n{context_block}\n\nQuestion: {user_question}"
    }
]
```

### 4.2 The Two Core API Functions

The course `utils.py` exposes two functions that cover all LLM interaction patterns:

#### `generate_with_single_input` — one prompt, one response

Use this for simple, standalone queries where no conversation history or system message is needed.

```python
from utils import generate_with_single_input

output = generate_with_single_input(
    prompt="What is the capital of France?",
    role="user",           # default; who is sending the message
    max_tokens=500,        # maximum response length
    temperature=0.0,       # 0.0 = deterministic/factual
)

# Return format: always a dict
print(output['role'])      # 'assistant'
print(output['content'])   # 'The capital of France is Paris...'
```

**Parameter reference:**

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `prompt` | `str` | required | The text to send |
| `role` | `str` | `'user'` | The role of this message |
| `max_tokens` | `int` | `500` | Max length of response in tokens |
| `temperature` | `float` | `None` | Randomness: 0.0 = deterministic, 1.0+ = creative |
| `top_p` | `float` | `None` | Nucleus sampling; alternative to temperature |
| `model` | `str` | Backend default | Which specific model to invoke |

#### `generate_with_multiple_input` — full conversation history

Use this when you need a system message, multi-turn history, or the system-user-assistant message structure that production RAG systems require.

```python
from utils import generate_with_multiple_input

messages = [
    {"role": "system",    "content": "You are a helpful assistant. Be concise."},
    {"role": "user",      "content": "What does RAG stand for?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
    {"role": "user",      "content": "What problem does it solve?"},  # new query
]

output = generate_with_multiple_input(
    messages=messages,
    max_tokens=200,
    temperature=0.0,
)

print(output['content'])
```

**Both functions return the same structure:**
```python
{
    "role":    "assistant",
    "content": "<the model's response text>"
}
```

This uniform return format is deliberate. Whether you call `generate_with_single_input` or `generate_with_multiple_input`, downstream code always extracts the response the same way: `output['content']`. Consistency reduces bugs and cognitive load.

### 4.3 Temperature and Determinism in RAG Contexts

Temperature is one of the most important hyperparameters in RAG deployments, and its optimal value depends on the use case:

| Use Case | Recommended Temperature | Rationale |
|---|---|---|
| Factual Q&A (e.g., policy lookup) | 0.0 – 0.2 | High determinism; responses should be grounded in retrieved facts |
| Summarization | 0.3 – 0.5 | Some paraphrasing flexibility is acceptable |
| Creative writing augmented by RAG | 0.7 – 1.0 | Creative generation desired, facts still grounded by context |
| Code generation | 0.0 – 0.1 | Code must be syntactically correct; randomness is harmful |

In most RAG applications, **low temperature (0.0–0.3)** is appropriate because the goal is accurate, grounded responses — not creative variation. The retriever provides the "interesting" specific information; the LLM's job is to synthesize it faithfully.

### 4.4 Token Budget Management

Every LLM API call has a **token budget**: the sum of input tokens (prompt + context) and output tokens (response) must not exceed the model's context window. Managing this budget is a practical engineering concern:

```
Total tokens = |system message| + |retrieved documents| + |user query| + |response|
             ≤ context window limit
```

For a model with a 32,000-token context window:

- System message: ~100 tokens
- 5 retrieved document chunks at ~300 tokens each: ~1,500 tokens
- User query: ~50 tokens
- **Available for response:** ~30,350 tokens

In this example there is ample room, but with long documents or large-k retrieval, the budget fills quickly. This is one reason why **chunking** (Part 3) and **top-k selection** are so important — they control the size of what enters the prompt.

**Practical token estimation:**
> As a rough rule of thumb, 1 token ≈ 0.75 words (in English). A 300-word document chunk is approximately 400 tokens. A 1,000-token system message is approximately 750 words.

---

## 5. Prompt Engineering for RAG

Prompt engineering is the practice of designing, iterating on, and optimizing the text instructions that guide an LLM's behavior. In RAG systems, prompt engineering operates at two levels: the **system message** (static behavioral instructions) and the **user message** (dynamic query + retrieved context).

### 5.1 The System Message: Defining the Model's Role

The system message is the most powerful lever for controlling LLM behavior in a RAG system. An effective system message for RAG should:

1. **Define the model's persona and domain** — what role is it playing, and in what context?
2. **Specify the information constraint** — the model should answer *only* from the provided context;
3. **Handle the "I don't know" case** — instruct the model on what to say when context is insufficient;
4. **Set the response format** — length, style, citation requirements.

**Example: Customer service chatbot system message**
```
You are a professional customer service assistant for Acme Corp.
Answer questions using ONLY the product documentation provided in the user's message.
If the answer is not explicitly present in the provided documentation, respond with:
"I don't have that information in our documentation. Please contact support@acme.com."
Be concise, factual, and professional. Do not invent specifications or policies.
```

**Example: Research assistant system message**
```
You are a rigorous research assistant specializing in machine learning.
Answer questions based strictly on the provided research excerpts.
When citing information, reference the document number (e.g., "[Document 2]").
If the provided excerpts are insufficient to answer fully, state what information is missing
and what additional documents might help.
Do not speculate beyond what the documents support.
```

Notice the explicit instruction to cite document numbers. Without this, the LLM will synthesize an answer from multiple sources without attribution — making it impossible for users to verify claims or dig deeper.

### 5.2 The User Message: Query + Context

The user message in a RAG system carries two pieces of information that are not normally present in a vanilla LLM interaction:

1. **The retrieved context** — the documents selected by the retriever;
2. **The user's original question** — unchanged from what the user typed.

The standard structure is:

```
[Retrieved context block]
[User question]
```

With the context preceding the question. There is research evidence (Liu et al., 2023) that LLMs attend better to information at the beginning and end of long inputs (the "lost in the middle" effect), so placing the most relevant document first in the context block is a practical optimization.

### 5.3 Prompt Templates and Parameterization

Hard-coding prompts inline leads to unmaintainable codebases. Professional RAG systems use **prompt templates** — parameterized string structures that separate the fixed instructional scaffolding from the dynamic data injected at runtime.

**A reusable prompt template function:**

```python
from typing import List, Dict, Optional

def build_rag_prompt(
    question: str,
    retrieved_docs: List[Dict],
    max_docs: int = 5,
    system_note: Optional[str] = None
) -> str:
    """
    Builds a formatted RAG user message from a question and retrieved documents.

    Args:
        question:       The user's original question.
        retrieved_docs: List of dicts with 'id', 'score', and 'text' keys.
        max_docs:       Maximum number of documents to include.
        system_note:    Optional additional instruction to prepend.

    Returns:
        A fully formatted string ready to use as the user message in an LLM call.
    """
    top_docs = retrieved_docs[:max_docs]

    context_lines = [
        f"[Document {doc['id']} | relevance: {doc['score']:.2f}]\n{doc['text']}"
        for doc in top_docs
    ]
    context_block = "\n\n".join(context_lines)

    base_prompt = f"""Retrieved Documents:
{context_block}

Question: {question}

Answer:"""

    if system_note:
        return f"Note: {system_note}\n\n{base_prompt}"
    return base_prompt
```

**Usage:**
```python
docs = [
    {"id": 1, "score": 0.95, "text": "Taylor Swift's Eras Tour arrives in Vancouver this weekend..."},
    {"id": 2, "score": 0.87, "text": "Vancouver hotel occupancy typically spikes during major events..."},
]

prompt = build_rag_prompt(
    question="Why are hotels in Vancouver so expensive this weekend?",
    retrieved_docs=docs,
    max_docs=2
)
```

### 5.4 The System + User Message Pattern (Production Style)

In production RAG systems, instructions and data are cleanly separated:
- The **system message** carries behavioral instructions (static; rarely changes);
- The **user message** carries the data context and question (dynamic; changes every request).

This separation makes the system easier to maintain, audit, and update. Changing the retrieval strategy (which affects only the data context) requires no change to the system message. Refining instructions (which affects only the system message) requires no change to how documents are formatted.

```python
def rag_query(question: str, retrieved_docs: List[Dict]) -> str:
    """Production-style RAG call with separate system and user messages."""
    
    # Format the retrieved context
    context = build_rag_prompt(question, retrieved_docs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual assistant. "
                "Answer only from the documents provided. "
                "If the answer is not in the documents, say so explicitly. "
                "Be concise and cite document numbers."
            )
        },
        {
            "role": "user",
            "content": context
        }
    ]

    response = generate_with_multiple_input(messages=messages, max_tokens=300)
    return response['content']
```

### 5.5 Multi-Turn RAG Conversations

Some RAG applications require **conversational context**: the user asks a follow-up question that implicitly refers to a previous exchange. In these cases, the conversation history is accumulated and passed with each API call.

```python
conversation = [
    {
        "role": "system",
        "content": "You are a real estate assistant. Answer only from the listing data provided."
    }
]

# Turn 1
listing_context = format_house_listings(house_data)
query_1 = "Which house is best value for money?"

conversation.append({
    "role": "user",
    "content": f"House listings:\n{listing_context}\n\nQuestion: {query_1}"
})

response_1 = generate_with_multiple_input(messages=conversation, max_tokens=150)
conversation.append(response_1)  # Add assistant's response to history

# Turn 2 — follow-up question; no need to repeat the listing data
query_2 = "How much does it cost per square foot?"
conversation.append({"role": "user", "content": query_2})

response_2 = generate_with_multiple_input(messages=conversation, max_tokens=100)
```

> **Important:** The retrieved context should be injected in the **first** user turn where it is needed. Subsequent follow-up turns can rely on the LLM's in-context memory of the conversation history, avoiding redundant re-injection of large document blocks.

---

## 6. Constructing the Augmented Prompt

The augmented prompt is where retrieval and generation meet. Its construction is a precise engineering task that directly determines RAG system quality. This section examines the anatomy of a well-constructed augmented prompt in detail.

### 6.1 Anatomy of an Augmented Prompt

A complete, production-quality augmented prompt has six functional zones:

```
┌─────────────────────────────────────────────────────────────┐
│ ZONE 1: SYSTEM MESSAGE (role: system)                        │
│ - Model persona and domain                                   │
│ - Information constraint ("answer only from context")        │
│ - Fallback instruction ("if not in context, say so")         │
│ - Format requirements (length, citations, tone)              │
├─────────────────────────────────────────────────────────────┤
│ ZONE 2: CONTEXT HEADER (role: user)                          │
│ - Signals to the model that documents follow                 │
│ - Example: "Use the following retrieved documents to answer" │
├─────────────────────────────────────────────────────────────┤
│ ZONE 3: RETRIEVED DOCUMENTS                                  │
│ - Each document clearly delimited                            │
│ - Source/ID metadata included for citation                   │
│ - Relevance scores optionally shown                          │
│ - Ordered by relevance (highest first)                       │
├─────────────────────────────────────────────────────────────┤
│ ZONE 4: QUESTION SEPARATOR                                   │
│ - Clear visual break between context and question            │
│ - Prevents the model from conflating context with query      │
├─────────────────────────────────────────────────────────────┤
│ ZONE 5: USER QUESTION                                        │
│ - The original, unmodified query                             │
├─────────────────────────────────────────────────────────────┤
│ ZONE 6: ANSWER PROMPT                                        │
│ - Optional "Answer:" or "Response:" cue                      │
│ - Signals where the model should begin its response          │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 A Complete Augmented Prompt Example

```python
system_message = """You are a precise research assistant specializing in AI and machine learning.
Answer questions based strictly on the documents provided below.
Cite document numbers when referencing specific claims (e.g., "[Doc 2]").
If the answer cannot be found in the provided documents, state:
"This information is not available in the provided documents."
Keep responses concise and factual."""

user_message = """Use the following retrieved documents to answer the question.

---RETRIEVED DOCUMENTS---

[Document 1 | relevance: 0.96]
RAG, or Retrieval-Augmented Generation, is a technique that improves LLM accuracy by
providing the model with relevant external documents at inference time. The documents
are retrieved from a knowledge base using a search component called the retriever.

[Document 2 | relevance: 0.89]
Hallucinations in LLMs occur when a model generates text that sounds plausible but is
factually incorrect. RAG significantly reduces hallucinations by grounding the model's
responses in retrieved, verified documents.

[Document 3 | relevance: 0.74]
The knowledge base in a RAG system is a curated collection of trusted documents.
It may contain company policies, product documentation, research papers, or any
domain-specific information relevant to the application.

---END OF DOCUMENTS---

Question: How does RAG reduce the likelihood of hallucinations?

Answer:"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user",   "content": user_message},
]
```

### 6.3 Formatting Documents for Maximum LLM Comprehension

The way documents are formatted inside the augmented prompt affects how well the LLM can distinguish between them, attribute claims, and avoid conflating information from different sources. Key formatting principles:

**Use consistent delimiters:**
```
[Document 1 | source: policy_handbook.pdf | page: 12]
<document text here>

[Document 2 | source: faq_2024.pdf | page: 3]
<document text here>
```

**Use XML-style tags (increasingly common in modern prompting):**
```xml
<documents>
  <document id="1" source="policy_handbook.pdf" relevance="0.95">
    <text>The company's remote work policy allows employees to...</text>
  </document>
  <document id="2" source="faq_2024.pdf" relevance="0.88">
    <text>Employees requesting equipment must submit a form...</text>
  </document>
</documents>
```

XML-style formatting has been shown to improve LLM performance on multi-document reasoning tasks, particularly with models like Claude that were trained with XML-delimited structure. The structured tags make document boundaries unambiguous, reducing attribution errors.

---

## 7. Building a Minimal RAG Pipeline in Python

We now assemble everything into a complete, working RAG pipeline. This is not a toy — it is a functioning system that implements the core loop of every RAG application, from the most basic to the most sophisticated.

### 7.1 Component 1: The Knowledge Base

```python
# A simple in-memory knowledge base (list of strings)
# In production, this would be a vector database with thousands of documents
knowledge_base = [
    "RAG stands for Retrieval-Augmented Generation. It improves LLM accuracy by "
    "providing relevant documents at inference time.",
    
    "LLMs have a training cutoff date. They cannot access information published "
    "after their training was completed.",
    
    "Hallucinations occur when an LLM generates confident-sounding but incorrect text. "
    "RAG reduces this by grounding responses in retrieved, verified documents.",
    
    "The retriever component searches the knowledge base and returns the most relevant "
    "documents for a given query. It assigns a relevance score to each document.",
    
    "Chunking is the process of splitting long documents into smaller segments "
    "so they fit within the LLM's context window and can be retrieved individually.",
    
    "Vector databases store document embeddings — dense numerical representations "
    "of text — and support fast similarity search over millions of documents.",
    
    "The augmented prompt combines the user's original question with the retrieved "
    "documents. The LLM reads this combined input and generates a grounded response.",
    
    "Temperature controls the randomness of LLM generation. A temperature of 0.0 "
    "produces deterministic, factual responses — ideal for RAG applications.",
]
```

### 7.2 Component 2: The Simple Retriever

For our minimal pipeline, we implement a **keyword-based retriever** using term overlap. This is not production-quality retrieval (Part 3 covers dense vector retrieval), but it captures the essential interface: accept a query, return the top-k most relevant documents.

```python
from typing import List, Tuple

def simple_retriever(query: str, corpus: List[str], top_k: int = 3) -> List[str]:
    """
    A simple keyword-overlap retriever.
    Scores documents by counting how many query words appear in each document.
    Returns the top_k highest-scoring documents.

    Args:
        query:  The user's question.
        corpus: The knowledge base (list of document strings).
        top_k:  How many documents to return.

    Returns:
        List of the top_k most relevant document strings.
    """
    # Normalize the query: lowercase, split into words
    query_words = set(query.lower().split())

    # Score each document: count how many query words it contains
    scored_docs: List[Tuple[float, str]] = []
    for doc in corpus:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)   # set intersection
        score = overlap / len(query_words)        # fraction of query words matched
        scored_docs.append((score, doc))

    # Sort by score (descending), return text of top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]
```

### 7.3 Component 3: The Prompt Builder

```python
def build_augmented_prompt(question: str, retrieved_docs: List[str]) -> List[dict]:
    """
    Constructs the full message list (system + user) for a RAG LLM call.

    Args:
        question:       The user's original question.
        retrieved_docs: The documents returned by the retriever.

    Returns:
        A list of message dicts ready for generate_with_multiple_input().
    """
    # Format the context block
    context_lines = [
        f"[Document {i+1}]\n{doc}"
        for i, doc in enumerate(retrieved_docs)
    ]
    context_block = "\n\n".join(context_lines)

    system_message = (
        "You are a precise and helpful assistant. "
        "Answer questions using ONLY the retrieved documents provided. "
        "If the answer is not in the documents, say so explicitly. "
        "Cite document numbers when relevant (e.g., '[Document 2]')."
    )

    user_message = f"""Retrieved Documents:
{context_block}

Question: {question}

Answer:"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]
```

### 7.4 Component 4: The Generator

```python
from utils import generate_with_multiple_input

def generate_response(messages: List[dict], max_tokens: int = 300) -> str:
    """
    Calls the LLM with the augmented prompt and returns the response text.

    Args:
        messages:   The full message list (system + user, from build_augmented_prompt).
        max_tokens: Maximum response length in tokens.

    Returns:
        The LLM's response as a plain string.
    """
    output = generate_with_multiple_input(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,   # deterministic for factual RAG
    )
    return output['content']
```

### 7.5 The Complete Pipeline: Retrieve → Augment → Generate

```python
def rag_pipeline(
    user_question: str,
    corpus: List[str],
    top_k: int = 3,
    max_tokens: int = 300
) -> str:
    """
    Complete RAG pipeline: Retrieve → Augment → Generate.

    Args:
        user_question: The question to answer.
        corpus:        The knowledge base (list of document strings).
        top_k:         Number of documents to retrieve.
        max_tokens:    Maximum response length.

    Returns:
        The LLM's grounded answer as a string.
    """
    # ── Step 1: RETRIEVE ──────────────────────────────────────────────────────
    retrieved_docs = simple_retriever(user_question, corpus, top_k=top_k)
    print(f"[RAG] Retrieved {len(retrieved_docs)} documents")

    # ── Step 2: AUGMENT ───────────────────────────────────────────────────────
    messages = build_augmented_prompt(user_question, retrieved_docs)

    # ── Step 3: GENERATE ──────────────────────────────────────────────────────
    answer = generate_response(messages, max_tokens=max_tokens)

    return answer


# ── Run the pipeline ──────────────────────────────────────────────────────────
question = "Why does RAG reduce hallucinations?"
answer = rag_pipeline(question, knowledge_base, top_k=3)

print(f"\nQ: {question}")
print(f"\nA: {answer}")
```

**Expected output:**
```
[RAG] Retrieved 3 documents

Q: Why does RAG reduce hallucinations?

A: RAG reduces hallucinations by providing the language model with relevant,
verified documents at inference time [Document 2]. Instead of relying solely on
patterns learned during training — which may be incomplete, outdated, or simply
incorrect — the model reads the actual retrieved text and synthesizes its response
from that grounded information. When the model's answer is anchored to specific
retrieved documents, it is less likely to generate plausible-sounding but false claims.
```

---

## 8. The Retrieve → Augment → Generate Loop in Detail

Having built the pipeline, it is worth examining each step in greater depth — both to understand what is happening computationally and to understand where the design choices are made.

### 8.1 Retrieve: The Computational Challenge

The retriever must solve a **ranking problem at scale**: given a query and thousands (or millions) of documents, find the top-k most relevant ones — ideally in milliseconds.

Our simple keyword retriever works well for tiny corpora but has two fundamental limitations:

**Limitation 1: Vocabulary mismatch.** If the user asks "What is neural search?" but the document says "semantic similarity search powered by embeddings," keyword overlap is low even though the documents are highly relevant. This is the **synonymy problem**, and it motivates dense (embedding-based) retrieval covered in Part 3.

**Limitation 2: No semantic understanding.** The keyword retriever treats "dog bites man" and "man bites dog" as equally relevant to both queries — because they contain the same words. Dense retrieval, which encodes meaning as vectors, solves this.

For the purposes of this part, the simple retriever is adequate. Its interface — `retrieve(query) → List[str]` — is what matters, and this interface is identical whether the underlying implementation is keyword overlap, BM25, or a full vector database with embedding models.

### 8.2 Augment: Why Prompt Structure Matters

The augmentation step is where most of the prompt engineering decisions live. Two systems with identical retrievers and identical LLMs can produce dramatically different outputs based solely on how the augmented prompt is structured.

**The critical decisions in augmentation:**

| Decision | Options | Impact |
|---|---|---|
| Document ordering | Highest relevance first vs. last | LLMs attend better to early and late content |
| Document delimiter | Plain text vs. XML tags vs. JSON | Affects attribution accuracy |
| Score inclusion | Show or hide relevance scores | Can bias model toward high-score docs |
| Instruction specificity | Generic vs. domain-specific constraints | Affects hallucination rate |
| Citation format | Inline vs. footnote vs. none | Determines verifiability of responses |

### 8.3 Generate: What the LLM "Sees"

At the generate step, the LLM receives the complete augmented prompt — system message plus the formatted user message with embedded documents — and generates a completion token by token. Crucially, **the LLM has no privileged access to the knowledge base** — it sees only what was placed in the prompt. This means:

- If a relevant document was not retrieved (retriever failure), the LLM cannot compensate;
- If irrelevant documents were included (retriever noise), the LLM may be misled;
- If the prompt structure is confusing (augmentation failure), the LLM may conflate documents.

This highlights a key principle of RAG system design: **the system is only as good as its weakest component**. A perfect LLM cannot recover from a poor retriever, and a perfect retriever cannot compensate for a poorly constructed augmented prompt.

---

## 9. Common Failure Modes and How to Diagnose Them

Every RAG system will fail in characteristic ways. Knowing these failure modes before you encounter them enables rapid diagnosis.

### 9.1 Retrieval Failure (The Retriever Returns Wrong Documents)

**Symptoms:**
- The LLM's answer is generic or uses information from training data rather than the knowledge base;
- The LLM says "I cannot find this in the documents" when the answer is clearly present in the corpus.

**Diagnosis:**
```python
# Inspect what the retriever actually returns
retrieved = simple_retriever(user_question, knowledge_base, top_k=5)
for i, doc in enumerate(retrieved):
    print(f"[{i+1}] {doc[:100]}...")  # show first 100 chars of each
```

**Likely causes and remedies:**
- Vocabulary mismatch → switch to dense (embedding-based) retrieval;
- `top_k` too small → increase `top_k`;
- Documents too long → implement chunking (Part 3).

### 9.2 Augmentation Failure (Context Is Ignored)

**Symptoms:**
- The LLM generates answers inconsistent with the retrieved documents;
- The LLM answers from training data even when relevant context is present.

**Diagnosis:** Check whether the system message sufficiently constrains the model to the provided context.

**Remedy:** Strengthen the information constraint:
```
# Weak:
"You are a helpful assistant. Use the context if relevant."

# Strong:
"You are a factual assistant. Answer ONLY using the documents provided.
If the answer is not in the documents, say: 'I cannot find this in the provided documents.'
Do not use your training knowledge. Cite document numbers."
```

### 9.3 Generation Failure (Model Ignores or Misattributes Information)

**Symptoms:**
- The answer cites "Document 3" but the relevant content was in "Document 1";
- The model generates plausible-sounding but unverifiable content.

**Remedies:**
- Improve document delimiting (make boundaries unambiguous);
- Use XML-style tags;
- Lower `temperature` toward 0.0;
- Reduce the number of documents (fewer documents reduces attribution confusion).

### 9.4 Token Overflow

**Symptoms:**
- `context_length_exceeded` error from the API;
- Response is truncated mid-sentence.

**Remedy:**
```python
# Estimate token count before calling the API (rough approximation)
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35)  # words × 1.35 ≈ tokens

total_input_tokens = estimate_tokens(system_message + user_message)
print(f"Estimated input tokens: {total_input_tokens}")
# If close to or exceeding context window, reduce top_k or truncate documents
```

---

## 10. Key Concepts Summary

| Concept | Definition |
|---|---|
| **`.env` file** | A plain-text file storing API keys and configuration; loaded at runtime; never committed to version control. |
| **Chat completions model** | The standardized LLM API paradigm: interactions as ordered sequences of `{role, content}` messages. |
| **`system` role** | The message role that defines the model's persona, behavioral constraints, and response format; set once per conversation. |
| **`user` role** | The message role representing the human's input — queries, data, instructions. |
| **`assistant` role** | The message role representing the model's prior responses; included for multi-turn conversation context. |
| **Temperature** | A parameter controlling generation randomness; 0.0 = deterministic, 1.0+ = creative. |
| **Top-k** | The number of documents the retriever returns for each query; a critical hyperparameter. |
| **Augmented prompt** | The combined message (retrieved documents + user query) sent to the LLM; the central artifact of RAG. |
| **Prompt template** | A parameterized string structure that separates static instructional scaffolding from dynamic runtime data. |
| **Token budget** | The constraint imposed by the context window: total tokens (input + output) must not exceed the model's limit. |
| **Information constraint** | The instruction in the system message that restricts the LLM to answering only from provided context. |
| **Vocabulary mismatch** | The failure mode where keyword-based retrieval misses relevant documents because they use different terminology. |
| **Retrieve → Augment → Generate** | The three-step loop that defines every RAG system: fetch relevant documents, construct the augmented prompt, generate the response. |

---

## 11. Further Reading and Foundational Papers

### Prompt Engineering

- **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022).** *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022. [`arXiv:2201.11903`](https://arxiv.org/abs/2201.11903)  
  *(Foundational work showing that structured prompts dramatically improve LLM reasoning — directly applicable to how RAG prompts are designed.)*

- **Brown, T., Mann, B., Ryder, N., et al. (2020).** *Language Models are Few-Shot Learners.* NeurIPS 2020. [`arXiv:2005.14165`](https://arxiv.org/abs/2005.14165)  
  *(The GPT-3 paper; introduced in-context learning — the principle that underlies RAG's ability to use retrieved documents without fine-tuning.)*

- **Anthropic. (2024).** *Claude's Prompt Engineering Guide.* [`docs.anthropic.com`](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)  
  *(Practical, research-backed guidance on system messages, XML structuring, and context injection.)*

### Context Utilization in LLMs

- **Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023).** *Lost in the Middle: How Language Models Use Long Contexts.* [`arXiv:2307.03172`](https://arxiv.org/abs/2307.03172)  
  *(Demonstrates that LLMs attend poorly to middle context — critical for deciding document ordering and how many documents to include.)*

- **Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E., ... & Zhou, D. (2023).** *Large Language Models Can Be Easily Distracted by Irrelevant Context.* ICML 2023. [`arXiv:2302.00093`](https://arxiv.org/abs/2302.00093)  
  *(Shows that irrelevant information in the prompt degrades LLM performance — a direct argument for precise retrieval.)*

### Information Retrieval Foundations

- **Manning, C. D., Raghavan, P., & Schütze, H. (2008).** *Introduction to Information Retrieval.* Cambridge University Press. [`Available free online`](https://nlp.stanford.edu/IR-book/)  
  *(The standard textbook on information retrieval; covers TF-IDF, BM25, evaluation metrics, and indexing. Chapters 1–6 are directly relevant to RAG retrieval.)*

### RAG System Design

- **Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2024).** *Retrieval-Augmented Generation for Large Language Models: A Survey.* [`arXiv:2312.10997`](https://arxiv.org/abs/2312.10997)  
  *(Section 3 of this survey covers the augmentation step and prompt design patterns for RAG in detail.)*

- **Edge, J., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., ... & Larson, J. (2024).** *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* [`arXiv:2404.16130`](https://arxiv.org/abs/2404.16130)  
  *(Microsoft's GraphRAG — demonstrates how prompt design for RAG extends beyond simple document injection to structured graph-based knowledge representation.)*

---

## 12. Review Questions

### Conceptual Understanding

1. Explain the difference between `generate_with_single_input` and `generate_with_multiple_input`. Give a specific example of a RAG use case where each is the appropriate choice.

2. Why is temperature typically set to a low value (0.0–0.2) in RAG applications focused on factual question answering? In what kind of RAG application might a higher temperature be appropriate?

3. The system message in a RAG prompt typically includes an "information constraint" instructing the model to answer only from provided context. Why might a model sometimes ignore this constraint? What prompt engineering techniques can strengthen it?

4. Explain the "lost in the middle" phenomenon (Liu et al., 2023) and its implications for how retrieved documents should be ordered within the augmented prompt.

### Engineering Practice

5. A colleague writes the following code to load an API key:
   ```python
   api_key = "sk-together-abc123xyz789"
   ```
   Identify two specific problems with this approach and rewrite it using best practices.

6. You are building a RAG system and notice that the LLM's responses often cite "Document 3" as the source of a claim that is actually in "Document 1." What is the most likely cause of this attribution error, and what change to the augmented prompt construction would you try first?

7. Your RAG pipeline is producing a `context_length_exceeded` error for queries that retrieve 10 documents of roughly 500 words each. Calculate approximately how many tokens your context is consuming and propose two independent remedies.

8. Rewrite the following weak system message as a strong, production-quality RAG system message for a healthcare information assistant:
   ```
   "You are a helpful medical assistant. Use the context to answer."
   ```

### Design and Analysis

9. The simple keyword retriever in §7.2 scores documents by counting how many query words they contain. Describe a specific query and a specific document where this scoring approach would fail badly (low score for a relevant document) and explain why.

10. A product manager asks: "Can we just skip the retriever and put the entire knowledge base (5,000 documents) in the system message?" Construct a detailed response addressing computational cost, the "lost in the middle" problem, and context window constraints.

11. Design the complete message list (system + user messages) for a RAG application that helps lawyers search case law. Your design should handle: (a) the citation requirement, (b) the instruction to stay within the provided documents, (c) a fallback for when the answer isn't in the retrieved case law, and (d) the response format (brief legal analysis followed by direct answer).

12. Trace through the complete `rag_pipeline()` function from §7.5 for the query `"What is temperature in LLM generation?"` against the knowledge base defined in §7.1. Which documents would the simple retriever return? Is this the correct set? What does this reveal about the limitation of keyword-based retrieval?

---

*End of Part 2 — Building the RAG Pipeline: Data, Prompts, and LLM Calls*

---

> **Up Next:** [Part 3 — Dense Retrieval, Embeddings, and Vector Databases](./03_Dense_Retrieval_and_Vector_Databases.md)  
> In Part 3, we replace the simple keyword retriever with a proper embedding-based dense retriever, introduce the mathematics of semantic similarity, and build a working vector search index.

---

*This document is part of the **Retrieval-Augmented Generation: From Foundations to Production** learning series.*  
*Content adapted from DeepLearning.AI RAG course materials (`00_Python_RAG_Prep.ipynb`, `01_LLM_Calls_and_Augmented_Prompts.ipynb`) and supplemented with academic references.*
