# Part 5: Evaluation, Monitoring, and Continuous Improvement

> **Series:** Retrieval-Augmented Generation — From Foundations to Production  
> **Part:** 5 of 6  
> **Level:** Intermediate–Advanced  
> **Prerequisites:** Parts 1–4; understanding of the full RAG pipeline (retrieve → augment → generate)  
> **Key Frameworks:** RAGAS, TREC Eval, BEIR, Human Evaluation Protocols

---

> **Prerequisites Check ✓**  
> Before starting this part, you should be able to:  
> - Describe Recall@k, Precision@k, and MRR as retrieval metrics  
> - Build a complete RAG pipeline (retriever + augmented prompt + LLM call)  
> - Explain what a hallucination is and why it occurs  
> - Define "faithfulness" in the context of LLM-generated text  
>
> If any of these feel uncertain, revisit Parts 3 and 4 before continuing.

---

## Table of Contents

1. [Introduction: You Cannot Improve What You Do Not Measure](#1-introduction-you-cannot-improve-what-you-do-not-measure)
2. [The RAG Evaluation Framework: Three Distinct Evaluation Targets](#2-the-rag-evaluation-framework-three-distinct-evaluation-targets)
3. [Evaluating Retrieval Quality](#3-evaluating-retrieval-quality)
4. [Evaluating Generation Quality: Faithfulness and Correctness](#4-evaluating-generation-quality-faithfulness-and-correctness)
5. [End-to-End RAG Evaluation with RAGAS](#5-end-to-end-rag-evaluation-with-ragas)
6. [Human Evaluation Protocols](#6-human-evaluation-protocols)
7. [LLM-as-Judge: Automated Quality Assessment](#7-llm-as-judge-automated-quality-assessment)
8. [Building a Comprehensive Evaluation Dataset](#8-building-a-comprehensive-evaluation-dataset)
9. [Production Monitoring: Observing a Live RAG System](#9-production-monitoring-observing-a-live-rag-system)
10. [Continuous Improvement: The Feedback Loop](#10-continuous-improvement-the-feedback-loop)
11. [Common Failure Patterns and Systematic Diagnosis](#11-common-failure-patterns-and-systematic-diagnosis)
12. [Key Concepts Summary](#12-key-concepts-summary)
13. [Further Reading and Foundational Papers](#13-further-reading-and-foundational-papers)
14. [Review Questions](#14-review-questions)

---

## 1. Introduction: You Cannot Improve What You Do Not Measure

A RAG system that has never been formally evaluated is an unknown system. It may perform well on the examples its developers tested during construction. It may fail in completely different ways on the distribution of queries that real users actually ask. Without measurement, there is no way to know which is true — and no principled way to improve.

This part addresses the discipline of **RAG evaluation**: the systematic, rigorous measurement of system quality across its components. Evaluation in RAG is particularly nuanced because there is no single metric that captures overall quality. A system can retrieve excellently but generate poorly. It can generate fluently but hallucinate. It can be faithful to the retrieved context but retrieve the wrong context in the first place. Understanding which component is failing — and measuring each independently — is the foundation of systematic improvement.

We begin with a fundamental architectural observation: a RAG system has **three distinct evaluation targets**, each requiring different measurement instruments.

---

## 2. The RAG Evaluation Framework: Three Distinct Evaluation Targets

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RAG System: Three Evaluation Zones                   │
├───────────────┬────────────────────────────────┬────────────────────────┤
│  ZONE 1       │  ZONE 2                        │  ZONE 3                │
│  RETRIEVAL    │  AUGMENTATION + GENERATION     │  END-TO-END            │
│               │                                │                        │
│  Are the      │  Is the LLM faithfully         │  Does the final        │
│  right docs   │  using what was retrieved?     │  answer satisfy        │
│  retrieved?   │  Is the answer grounded?       │  the user's need?      │
│               │                                │                        │
│  Metrics:     │  Metrics:                      │  Metrics:              │
│  Recall@k     │  Faithfulness                  │  Answer Correctness    │
│  Precision@k  │  Answer Relevance              │  User Satisfaction     │
│  MRR, NDCG    │  Context Utilization           │  Task Completion Rate  │
└───────────────┴────────────────────────────────┴────────────────────────┘
```

These three zones must be evaluated independently because they fail independently. A failure in Zone 1 (wrong documents retrieved) will appear as a failure in Zone 3 (wrong answer), but the fix is entirely in the retriever — not the LLM. Treating the whole system as a black box and optimizing Zone 3 directly makes it impossible to identify the actual root cause.

**The evaluation-before-optimization principle:**  
Always diagnose which zone is failing before making changes. Tuning the LLM when retrieval is the problem wastes resources. Tuning the retriever when the prompt is the problem achieves nothing.

---

## 3. Evaluating Retrieval Quality

Retrieval quality evaluation requires a **labeled dataset**: for each test query, you need to know which documents in the knowledge base are actually relevant. This dataset is called the **ground truth** and its construction is one of the most labor-intensive (and most important) steps in RAG evaluation.

### 3.1 Core Retrieval Metrics

#### Recall@k

Of all relevant documents that exist in the corpus, what fraction does the retriever return in its top-k results?

```
Recall@k = |{relevant documents} ∩ {top-k retrieved}| / |{relevant documents}|
```

**Range:** 0 (none found) to 1.0 (all found).  
**In RAG:** Recall@k is the single most important retrieval metric. If a relevant document is not retrieved (Recall@k < 1.0), the LLM has zero chance of answering correctly from context. Typical targets: Recall@5 ≥ 0.85 for production systems.

#### Precision@k

Of the k documents retrieved, what fraction is actually relevant?

```
Precision@k = |{relevant documents} ∩ {top-k retrieved}| / k
```

**In RAG:** Lower precision means more irrelevant documents enter the prompt, increasing cost, noise, and the risk of the LLM being misled (Shi et al., 2023). However, precision is secondary to recall — a missed relevant document is almost always more harmful than an extra irrelevant one.

#### MRR (Mean Reciprocal Rank)

For each query, the reciprocal rank is 1 divided by the rank position of the first relevant document. MRR is the mean over all queries:

```
MRR = (1/|Q|) × Σ_{q ∈ Q} (1 / rank_q)
```

**In RAG:** MRR rewards systems that place the most important document first. Since LLMs attend better to early-positioned context (Liu et al., 2023), high MRR translates directly to better generation quality.

#### NDCG@k (Normalized Discounted Cumulative Gain)

NDCG is a graded metric that gives higher credit for relevant documents appearing earlier in the ranked list, using a logarithmic discount:

```
DCG@k = Σ_{i=1}^{k} rel_i / log_2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

where `rel_i` is the relevance grade of the document at rank `i` (binary: 0 or 1, or graded: 0, 1, 2), and `IDCG@k` is the ideal DCG (what you'd get from a perfect ranker). NDCG@10 is the standard metric on BEIR and MTEB benchmarks.

### 3.2 Python Implementation

```python
import numpy as np
from typing import List, Set, Dict

def compute_recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    """Compute Recall@k for a single query."""
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant) if relevant else 0.0


def compute_precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    """Compute Precision@k for a single query."""
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / k


def compute_mrr(
    retrieved: List[str],
    relevant: Set[str]
) -> float:
    """Compute Reciprocal Rank for a single query."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0  # no relevant document found


def compute_ndcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    """Compute NDCG@k for a single query (binary relevance)."""
    dcg = sum(
        1.0 / np.log2(i + 2)          # log_2(rank + 1), 0-indexed rank → +2
        for i, doc in enumerate(retrieved[:k])
        if doc in relevant
    )
    # Ideal DCG: all relevant docs at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retriever(
    test_queries: List[str],
    ground_truth: List[Set[str]],       # relevant doc IDs per query
    retrieve_fn,                         # query → List[doc_id]
    k: int = 5
) -> Dict[str, float]:
    """
    Evaluate a retriever on a labeled test set.

    Args:
        test_queries:  List of test query strings.
        ground_truth:  For each query, set of relevant document IDs.
        retrieve_fn:   Function: query → List[doc_id] (ranked).
        k:             Evaluation depth.

    Returns:
        Dict of mean scores: Recall@k, Precision@k, MRR, NDCG@k.
    """
    recalls, precisions, mrrs, ndcgs = [], [], [], []

    for query, relevant in zip(test_queries, ground_truth):
        retrieved = retrieve_fn(query)

        recalls.append(compute_recall_at_k(retrieved, relevant, k))
        precisions.append(compute_precision_at_k(retrieved, relevant, k))
        mrrs.append(compute_mrr(retrieved, relevant))
        ndcgs.append(compute_ndcg_at_k(retrieved, relevant, k))

    return {
        f"Recall@{k}":    float(np.mean(recalls)),
        f"Precision@{k}": float(np.mean(precisions)),
        "MRR":            float(np.mean(mrrs)),
        f"NDCG@{k}":      float(np.mean(ndcgs)),
    }
```

### 3.3 Interpreting Retrieval Scores

| Score Range | Interpretation | Action |
|---|---|---|
| Recall@5 < 0.60 | Critical retrieval failure | Overhaul retriever: check chunking, switch to dense/hybrid |
| Recall@5 0.60–0.80 | Suboptimal; significant answer gaps | Tune chunking, increase `top_k`, try hybrid retrieval |
| Recall@5 0.80–0.90 | Good baseline | Fine-tune embedding model on domain data |
| Recall@5 > 0.90 | Production-ready retrieval | Focus optimization on generation quality |
| MRR < 0.50 | Relevant docs are buried; LLM cannot use them | Improve re-ranking; reorder prompt context |

---

## 4. Evaluating Generation Quality: Faithfulness and Correctness

Once we know the retriever is working, we evaluate whether the LLM is correctly using the retrieved documents. Generation evaluation has two primary dimensions:

### 4.1 Faithfulness

**Faithfulness** measures whether every claim in the LLM's response is supported by the retrieved context. An unfaithful response contains claims that cannot be verified against — or actively contradict — the retrieved documents. Faithfulness is the primary proxy for hallucination in RAG.

Formally: a response `r` is faithful to context `C` if every atomic claim `c_i` in `r` can be entailed from some passage in `C`.

```
Faithfulness(r, C) = |{claims in r entailed by C}| / |{total claims in r}|
```

Range: 0 (no claims supported) to 1.0 (all claims supported).

### 4.2 Answer Relevance

**Answer relevance** measures whether the response actually addresses the user's question — regardless of whether it is faithful. A response can be faithful (only uses retrieved content) but irrelevant (discusses a different aspect of the topic). Both dimensions must be evaluated.

```
Answer_Relevance(r, q) = semantic_similarity(r, q)  [via embedding cosine similarity]
```

### 4.3 Context Precision and Context Recall

Two additional metrics evaluate how well the retrieved context is being used:

- **Context Precision**: What fraction of the retrieved context is actually relevant to the answer? (Measures retrieval precision from the perspective of what the generation uses.)
- **Context Recall**: What fraction of the information needed to answer the question is present in the retrieved context? (Measures retrieval completeness from the generation's perspective.)

### 4.4 Implementing a Simple Faithfulness Checker

```python
from utils import generate_with_multiple_input

def check_faithfulness(
    answer: str,
    retrieved_context: str,
    verbose: bool = False
) -> dict:
    """
    Use an LLM to assess whether each claim in an answer is supported
    by the retrieved context.

    Returns:
        Dict with 'faithfulness_score' (0.0–1.0) and 'analysis'.
    """
    system_message = """You are a factual accuracy auditor.
Given a CONTEXT and an ANSWER, identify each factual claim in the ANSWER.
For each claim, determine if it is:
- SUPPORTED: directly entailed by the context
- UNSUPPORTED: not present in or contradicts the context
- INFERRED: a reasonable inference from the context (not directly stated)

Return JSON only:
{
  "claims": [
    {"claim": "...", "status": "SUPPORTED|UNSUPPORTED|INFERRED", "evidence": "..."}
  ],
  "faithfulness_score": 0.0-1.0
}
The faithfulness_score = supported_count / total_count."""

    user_message = f"""CONTEXT:
{retrieved_context}

ANSWER:
{answer}

Analyze each claim in the ANSWER:"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message}
    ]

    output = generate_with_multiple_input(messages=messages, max_tokens=600, temperature=0.0)

    import json, re
    raw = output['content']
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if verbose:
                for c in result.get('claims', []):
                    print(f"  [{c['status']}] {c['claim']}")
            return result
        except json.JSONDecodeError:
            pass

    return {"faithfulness_score": None, "error": "Could not parse response"}
```

---

## 5. End-to-End RAG Evaluation with RAGAS

**RAGAS** (Retrieval Augmented Generation Assessment) (Es et al., 2023) is the most widely adopted framework for end-to-end RAG evaluation. It provides a suite of reference-free metrics that can evaluate a RAG system without needing human-labeled ground-truth answers — an enormous practical advantage, as labeling is expensive and slow.

### 5.1 RAGAS Metrics Overview

RAGAS defines four core metrics:

| Metric | Measures | Requires |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? | Answer + Context |
| **Answer Relevance** | Does the answer address the question? | Answer + Question |
| **Context Precision** | Are the retrieved chunks ranked with relevant ones first? | Context + Ground Truth |
| **Context Recall** | Does the retrieved context contain all info needed to answer? | Context + Ground Truth |

The four scores are combined into a single **RAGAS Score**:

```
RAGAS = harmonic_mean(Faithfulness, Answer_Relevance, Context_Precision, Context_Recall)
```

The harmonic mean ensures that a single very low metric cannot be masked by high scores on others — a system with Faithfulness = 0.3 will have a low RAGAS score even if the other three metrics are 1.0.

### 5.2 Using the RAGAS Library

```python
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Build evaluation dataset
eval_data = {
    "question": [
        "What is the capital of France?",
        "How does RAG reduce hallucinations?",
        "What is chunking in RAG?",
    ],
    "answer": [
        "The capital of France is Paris.",
        "RAG reduces hallucinations by grounding LLM responses in retrieved documents.",
        "Chunking splits long documents into smaller segments for retrieval.",
    ],
    "contexts": [
        ["Paris is the capital and largest city of France."],
        ["RAG grounds LLM responses by providing retrieved documents at inference time.",
         "Hallucinations decrease when the LLM has verified context to draw from."],
        ["Chunking is the process of dividing long documents into smaller retrievable segments.",
         "Optimal chunk size depends on document type and the LLM's context window."],
    ],
    "ground_truth": [  # needed only for Context Recall
        "Paris is the capital of France.",
        "RAG reduces hallucinations by providing verified documents that ground the response.",
        "Chunking divides documents into smaller segments for retrieval.",
    ]
}

dataset = Dataset.from_dict(eval_data)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# Output:
# {'faithfulness': 0.94, 'answer_relevancy': 0.91,
#  'context_precision': 0.89, 'context_recall': 0.87}
```

### 5.3 RAGAS Without Ground Truth (Reference-Free Mode)

RAGAS's most powerful feature is that **Faithfulness and Answer Relevance** require no ground-truth labels — they evaluate the answer against the retrieved context and the question alone. This enables continuous evaluation on live user traffic without any human annotation:

```python
# Reference-free evaluation — runs on live traffic
reference_free_metrics = [faithfulness, answer_relevancy]

live_eval_data = {
    "question":  [user_query],
    "answer":    [llm_response],
    "contexts":  [retrieved_docs],
    # No ground_truth needed!
}

live_result = evaluate(
    dataset=Dataset.from_dict(live_eval_data),
    metrics=reference_free_metrics
)
```

### 5.4 Interpreting RAGAS Scores

| Score | Interpretation |
|---|---|
| RAGAS > 0.85 | High-quality system; production ready |
| RAGAS 0.70–0.85 | Acceptable; targeted improvements possible |
| RAGAS 0.55–0.70 | Needs significant improvement before production |
| RAGAS < 0.55 | Critical quality issues; do not deploy |

**Diagnostic patterns:**

| Low Metric | Root Cause | Fix |
|---|---|---|
| Low Faithfulness | LLM ignoring context; strong grounding instruction needed | Strengthen information constraint; lower temperature |
| Low Answer Relevance | Retriever returning off-topic docs | Improve retrieval; strengthen query understanding |
| Low Context Precision | Retrieving many irrelevant chunks | Reduce `top_k`; improve re-ranking |
| Low Context Recall | Missing relevant chunks | Increase `top_k`; review chunking strategy |

---

## 6. Human Evaluation Protocols

Automated metrics are fast and scalable but imperfect. They cannot capture all dimensions of quality that matter to users — tone, appropriateness, helpfulness, clarity. Human evaluation is the ground truth against which automated metrics are calibrated.

### 6.1 The Annotation Schema

A rigorous human evaluation uses a structured annotation schema with clear, operationalized definitions for each rating dimension:

```python
annotation_schema = {
    "faithfulness": {
        "scale": "1–5",
        "definition": "To what degree is every claim in the response supported by the provided documents?",
        "anchors": {
            1: "Multiple major claims contradict or are absent from the documents",
            3: "Most claims supported; 1–2 unsupported or inferred claims",
            5: "Every claim is directly supported by the retrieved documents"
        }
    },
    "answer_completeness": {
        "scale": "1–5",
        "definition": "Does the response fully address all aspects of the question?",
        "anchors": {
            1: "Response misses the main point of the question entirely",
            3: "Response addresses the question partially; key aspects missing",
            5: "Response fully and completely addresses all aspects of the question"
        }
    },
    "answer_correctness": {
        "scale": "Binary: correct / incorrect",
        "definition": "Is the factual content of the response correct?",
        "note": "Requires domain expert annotators or gold-standard reference answers"
    },
    "citation_accuracy": {
        "scale": "1–5",
        "definition": "Do cited document numbers correctly correspond to the claims they support?",
        "anchors": {
            1: "Citations are fabricated or systematically incorrect",
            5: "All citations are accurate and correctly attributed"
        }
    },
    "response_quality": {
        "scale": "1–5",
        "definition": "Overall quality of the response: clarity, appropriateness, conciseness",
        "anchors": {
            1: "Response is unclear, inappropriate, or grossly verbose",
            5: "Response is exceptionally clear, appropriately concise, and well-structured"
        }
    }
}
```

### 6.2 Inter-Annotator Agreement

Any human evaluation must measure **inter-annotator agreement** — the degree to which different annotators assign the same ratings to the same examples. Low agreement indicates that the schema is ambiguous, annotators need more training, or the task is genuinely subjective.

The standard measure for categorical annotation is **Cohen's Kappa (κ)**:

```python
from sklearn.metrics import cohen_kappa_score

# Ratings from two annotators on 50 examples (faithfulness, scale 1–5)
annotator_1_ratings = [5, 4, 3, 5, 2, 4, 5, 3, 4, 5, ...]
annotator_2_ratings = [5, 4, 3, 4, 2, 4, 5, 3, 3, 5, ...]

kappa = cohen_kappa_score(annotator_1_ratings, annotator_2_ratings)
print(f"Cohen's Kappa: {kappa:.3f}")

# Interpretation:
# κ < 0.20: slight agreement — schema needs revision
# κ 0.21–0.40: fair agreement — annotator training needed
# κ 0.41–0.60: moderate agreement — acceptable for many tasks
# κ 0.61–0.80: substantial agreement — good annotation quality
# κ > 0.80: excellent agreement — publication-quality annotation
```

Target κ ≥ 0.60 for production evaluation datasets. Below this threshold, the ratings are too noisy to support reliable conclusions.

### 6.3 Annotation Workflow Design

```
┌──────────────────────────────────────────────────────────────┐
│                 Human Evaluation Workflow                      │
│                                                               │
│  1. Sample 200–500 live queries (stratified by query type)   │
│  2. Run full RAG pipeline on each query                       │
│  3. Assign each (query, retrieved docs, response) triple      │
│     to 2 independent annotators                               │
│  4. Measure inter-annotator agreement (κ)                     │
│     - If κ < 0.60: revise schema + re-annotate               │
│     - If κ ≥ 0.60: resolve disagreements by majority / expert│
│  5. Compute mean scores across all dimensions                 │
│  6. Segment by query type, domain, user cohort                │
│  7. Identify systematic failure patterns                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. LLM-as-Judge: Automated Quality Assessment

A rapidly adopted technique in production RAG evaluation is **LLM-as-judge** (Zheng et al., 2023): using a powerful LLM (typically a frontier model like GPT-4 or Claude) to evaluate the outputs of the RAG system. This approach is faster and cheaper than human evaluation while correlating reasonably well with human judgments on well-designed rubrics.

### 7.1 Single-Answer Grading

```python
def llm_judge_single(
    question: str,
    answer: str,
    context: str,
    criterion: str = "faithfulness"
) -> dict:
    """
    Use an LLM to grade a single RAG response on a specified criterion.

    Args:
        question:  The original user question.
        answer:    The RAG system's response.
        context:   The retrieved documents used to generate the answer.
        criterion: The evaluation criterion (faithfulness, relevance, completeness).

    Returns:
        Dict with 'score' (1–5) and 'reasoning'.
    """
    rubrics = {
        "faithfulness": (
            "Rate how faithfully the ANSWER is grounded in the CONTEXT. "
            "Score 5: all claims directly supported. "
            "Score 3: most claims supported; minor unsupported inferences. "
            "Score 1: multiple claims contradict or are absent from context."
        ),
        "relevance": (
            "Rate how directly and completely the ANSWER addresses the QUESTION. "
            "Score 5: answer fully addresses all aspects of the question. "
            "Score 3: answer partially addresses the question. "
            "Score 1: answer does not address the question."
        ),
        "completeness": (
            "Rate whether the ANSWER covers all information from the CONTEXT "
            "that is relevant to the QUESTION. "
            "Score 5: all relevant context used. "
            "Score 3: some relevant context omitted. "
            "Score 1: most relevant context ignored."
        )
    }

    system_message = f"""You are an expert evaluator of AI assistant responses.
Evaluate the response according to this rubric:
{rubrics.get(criterion, rubrics['faithfulness'])}

Return JSON only:
{{"score": 1-5, "reasoning": "one-sentence justification"}}"""

    user_message = f"""QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate:"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message}
    ]

    output = generate_with_multiple_input(messages=messages, max_tokens=200, temperature=0.0)

    import json, re
    json_match = re.search(r'\{.*\}', output['content'], re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"score": None, "error": "Parse failure"}
```

### 7.2 Pairwise Comparison (A/B Evaluation)

For comparing two system versions, pairwise judgment is more reliable than absolute scoring:

```python
def llm_judge_pairwise(
    question: str,
    context: str,
    answer_a: str,      # e.g., from prompt version 1.0
    answer_b: str,      # e.g., from prompt version 1.1
) -> dict:
    """
    Ask an LLM judge which of two answers is better.
    Returns preference (A/B/TIE) and reasoning.
    """
    system_message = """You are an expert evaluator comparing two AI responses.
Given a QUESTION and its CONTEXT, decide which ANSWER better satisfies the user.
Consider: faithfulness to context, directness, completeness, clarity.
Return JSON: {"preference": "A" | "B" | "TIE", "reasoning": "one sentence"}"""

    user_message = f"""QUESTION: {question}

CONTEXT:
{context}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Which is better?"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message}
    ]
    output = generate_with_multiple_input(messages=messages, max_tokens=150, temperature=0.0)

    import json, re
    json_match = re.search(r'\{.*\}', output['content'], re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"preference": None, "error": "Parse failure"}
```

### 7.3 Known Biases in LLM-as-Judge

LLM judges are powerful but systematically biased in several ways. Awareness of these biases is essential for valid evaluation:

| Bias | Description | Mitigation |
|---|---|---|
| **Position bias** | Judges prefer the first answer presented | Randomize A/B order; average over both orderings |
| **Verbosity bias** | Judges prefer longer, more detailed answers | Include length as an explicit negative criterion |
| **Self-enhancement bias** | A model rates its own outputs higher | Use a different model as judge than the one being evaluated |
| **Anchoring** | Early scores anchor later ones | Randomize evaluation order across sessions |
| **Format bias** | Judges prefer well-formatted responses | Explicitly instruct the judge to ignore formatting |

---

## 8. Building a Comprehensive Evaluation Dataset

An evaluation dataset is the backbone of all systematic RAG improvement. Its quality determines whether your metrics reflect real system quality or merely measure performance on an unrepresentative subset.

### 8.1 Query Taxonomy

A comprehensive evaluation dataset covers multiple query types, each exercising different system capabilities:

```python
query_taxonomy = {
    "factual_simple": {
        "description": "Single-hop factual lookup",
        "example": "What is the company's remote work policy?",
        "primary_metric": "Answer Correctness",
        "target_proportion": 0.30
    },
    "factual_complex": {
        "description": "Multi-hop or multi-document factual synthesis",
        "example": "How does our parental leave policy differ from the FMLA minimum?",
        "primary_metric": "Faithfulness + Answer Completeness",
        "target_proportion": 0.20
    },
    "comparative": {
        "description": "Comparing two or more entities",
        "example": "What are the differences between our Basic and Premium support tiers?",
        "primary_metric": "Answer Completeness",
        "target_proportion": 0.15
    },
    "out_of_scope": {
        "description": "Query about topics not in the knowledge base",
        "example": "What is the current stock price?",
        "primary_metric": "Fallback Quality",
        "target_proportion": 0.15
    },
    "ambiguous": {
        "description": "Queries that are genuinely ambiguous or underspecified",
        "example": "What is the policy on travel?",
        "primary_metric": "Clarification Handling",
        "target_proportion": 0.10
    },
    "adversarial": {
        "description": "Queries designed to induce hallucination or grounding failures",
        "example": "What does [document not in KB] say about X?",
        "primary_metric": "Hallucination Rate",
        "target_proportion": 0.10
    }
}
```

### 8.2 Synthetic Dataset Generation

Manual query creation is expensive. Synthetic data generation using an LLM to create query-answer pairs from documents is a practical alternative:

```python
def generate_eval_pairs_from_document(document_text: str, n_pairs: int = 5) -> list:
    """
    Generate (question, ground_truth_answer) pairs from a document.
    These form the ground-truth labels for Context Recall evaluation.
    """
    system_message = f"""You are an evaluation dataset generator.
Given a DOCUMENT, generate {n_pairs} diverse question-answer pairs that:
- Can be answered from the document content
- Cover different aspects and difficulty levels
- Include at least one multi-hop reasoning question
- Include at least one question where the answer is partially in the document

Return JSON only:
{{"pairs": [{{"question": "...", "answer": "...", "type": "simple|complex|partial"}}]}}"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": f"DOCUMENT:\n{document_text}"}
    ]

    output = generate_with_multiple_input(messages=messages, max_tokens=1000, temperature=0.3)

    import json, re
    json_match = re.search(r'\{.*\}', output['content'], re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group()).get('pairs', [])
        except json.JSONDecodeError:
            pass
    return []
```

> **Critical note on synthetic data:** Synthetic datasets from the same LLM family as your generator tend to be biased toward question types that LLMs answer well. Always supplement with real user queries from production logs, or manually add challenging edge cases.

### 8.3 Dataset Size Guidelines

| System Stage | Minimum Dataset Size | Purpose |
|---|---|---|
| Prototype evaluation | 50–100 examples | Initial quality check |
| Pre-production validation | 300–500 examples | Release decision gate |
| Production monitoring | 1,000+ (ongoing) | Trend detection, regression catching |
| Benchmark comparison | 2,000+ per domain | Publishable comparison |

---

## 9. Production Monitoring: Observing a Live RAG System

Evaluation on a static test set tells you how well the system performs at a point in time. Production monitoring tells you how it performs continuously — detecting drift, regressions, and emerging failure patterns before they materially harm users.

### 9.1 What to Log

Every RAG request in production should generate a structured log entry:

```python
import uuid
import time

def rag_with_logging(
    user_query: str,
    retrieve_fn,
    generate_fn,
    session_id: str = None
) -> dict:
    """
    Production RAG pipeline with comprehensive logging.
    Every field in the log entry is used for monitoring and debugging.
    """
    request_id = str(uuid.uuid4())
    t0 = time.time()

    # Step 1: Retrieve
    t1 = time.time()
    retrieved = retrieve_fn(user_query)
    retrieval_latency_ms = (time.time() - t1) * 1000

    # Step 2 + 3: Augment + Generate
    t2 = time.time()
    response = generate_fn(user_query, retrieved)
    generation_latency_ms = (time.time() - t2) * 1000

    total_latency_ms = (time.time() - t0) * 1000

    log_entry = {
        "request_id":           request_id,
        "session_id":           session_id,
        "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        # Query
        "user_query":           user_query,
        "query_length_tokens":  len(user_query.split()),

        # Retrieval
        "retrieved_doc_ids":    [doc['id'] for doc in retrieved],
        "retrieved_doc_scores": [doc['score'] for doc in retrieved],
        "n_docs_retrieved":     len(retrieved),
        "retrieval_latency_ms": retrieval_latency_ms,

        # Generation
        "response":             response,
        "response_length_tokens": len(response.split()),
        "generation_latency_ms": generation_latency_ms,

        # Total
        "total_latency_ms":     total_latency_ms,

        # To be filled asynchronously by evaluation pipeline
        "faithfulness_score":   None,
        "user_rating":          None,   # filled if user provides thumbs up/down
        "flagged_for_review":   False,
    }

    # Persist log entry (database, data warehouse, log aggregator)
    persist_log(log_entry)

    return {"response": response, "request_id": request_id}
```

### 9.2 Key Production Metrics Dashboard

A production RAG monitoring dashboard should track seven key metrics over time:

```
┌─────────────────────────────────────────────────────────────┐
│               RAG Production Dashboard                        │
├─────────────────────────────────────────────────────────────┤
│  QUALITY METRICS (sample-based, updated daily)              │
│  ┌─────────────────┬───────────────┬──────────────────────┐ │
│  │ Faithfulness    │ 0.89 ↗ +0.02  │ Target: ≥ 0.85       │ │
│  │ Answer Relevance│ 0.83 → +0.00  │ Target: ≥ 0.80       │ │
│  │ RAGAS Score     │ 0.79 ↗ +0.01  │ Target: ≥ 0.75       │ │
│  └─────────────────┴───────────────┴──────────────────────┘ │
│                                                               │
│  OPERATIONAL METRICS (all requests, real-time)              │
│  ┌─────────────────┬───────────────┬──────────────────────┐ │
│  │ P50 Latency     │ 1.2s          │ Target: < 2.0s       │ │
│  │ P99 Latency     │ 4.8s          │ Target: < 8.0s       │ │
│  │ Error Rate      │ 0.3%          │ Target: < 1.0%       │ │
│  │ Fallback Rate   │ 12%           │ Monitor for drift    │ │
│  └─────────────────┴───────────────┴──────────────────────┘ │
│                                                               │
│  USER FEEDBACK (when available)                              │
│  ┌─────────────────┬───────────────┬──────────────────────┐ │
│  │ Thumbs Up Rate  │ 78%           │ Target: ≥ 75%        │ │
│  │ Thumbs Down     │ 22%           │ Monitor topics       │ │
│  └─────────────────┴───────────────┴──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 Drift Detection

Over time, a RAG system's performance can drift for several reasons:
- The **query distribution shifts** (users start asking different types of questions);
- The **knowledge base becomes stale** (documents fall out of date);
- The **LLM model version changes** (provider updates affect behavior).

A simple drift detection approach tracks rolling mean scores and alerts when they deviate significantly:

```python
import numpy as np
from collections import deque

class DriftDetector:
    """
    Detects performance drift in production RAG metrics using a sliding window.
    """
    def __init__(self, window_size: int = 200, alert_threshold: float = 0.05):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.baseline_mean = None
        self.window = deque(maxlen=window_size)

    def set_baseline(self, baseline_scores: list):
        self.baseline_mean = np.mean(baseline_scores)
        print(f"Baseline set: {self.baseline_mean:.3f}")

    def add_score(self, score: float) -> dict:
        self.window.append(score)
        current_mean = np.mean(list(self.window))
        drift = self.baseline_mean - current_mean  # positive = degradation

        alert = drift > self.alert_threshold if self.baseline_mean else False
        return {
            "current_mean": current_mean,
            "baseline_mean": self.baseline_mean,
            "drift": drift,
            "alert": alert
        }

# Usage
faithfulness_monitor = DriftDetector(window_size=200, alert_threshold=0.05)
faithfulness_monitor.set_baseline(historical_faithfulness_scores)

# Add new score from production
status = faithfulness_monitor.add_score(new_faithfulness_score)
if status['alert']:
    send_alert(f"Faithfulness drift detected: {status['drift']:.3f} below baseline")
```

---

## 10. Continuous Improvement: The Feedback Loop

Evaluation is not a one-time activity — it is a recurring process embedded in a **continuous improvement feedback loop**. The loop has four phases:

```
         ┌─────────────────────────────────────────────────────┐
         │                                                     │
    ┌────▼────┐         ┌──────────┐         ┌──────────────┐  │
    │ DEPLOY  │────────▶│ MONITOR  │────────▶│  DIAGNOSE    │  │
    │         │         │          │         │              │  │
    │ Release │         │ Collect  │         │ Identify     │  │
    │ system  │         │ metrics, │         │ which zone   │  │
    │ to prod │         │ logs,    │         │ is failing   │  │
    │         │         │ feedback │         │ and why      │  │
    └─────────┘         └──────────┘         └──────┬───────┘  │
         ▲                                          │          │
         │              ┌──────────┐                │          │
         └──────────────│  IMPROVE │◀───────────────┘          │
                        │          │                           │
                        │ Tune the │                           │
                        │ right    │                           │
                        │ component│───────────────────────────┘
                        └──────────┘
```

### 10.1 The Improvement Decision Tree

When monitoring reveals a quality problem, the diagnosis should be systematic:

```
Quality problem detected (RAGAS score drops / user satisfaction falls)
│
├─▶ Is Recall@k low?
│   YES → Retrieval problem
│         ├─▶ Is chunking too coarse? → Reduce chunk size, add overlap
│         ├─▶ Is retrieval method wrong? → Switch to dense/hybrid
│         └─▶ Is knowledge base stale? → Update and re-index
│
├─▶ Is Faithfulness low (but Recall@k is fine)?
│   YES → Generation problem
│         ├─▶ Is grounding instruction weak? → Strengthen constraint
│         ├─▶ Is temperature too high? → Lower to 0.0–0.1
│         ├─▶ Is context too noisy (too many irrelevant docs)? → Reduce top_k
│         └─▶ Does model need domain fine-tuning? → Consider PEFT/LoRA
│
├─▶ Is Answer Relevance low (but Faithfulness is fine)?
│   YES → Prompt/augmentation problem
│         ├─▶ Is the system message too vague? → Add domain specificity
│         ├─▶ Is context ordering suboptimal? → Reorder by relevance
│         └─▶ Are few-shot examples misaligned? → Update examples
│
└─▶ Is Fallback Rate rising?
    YES → Knowledge base gap
          └─▶ Identify common unanswerable query types → Expand KB coverage
```

### 10.2 Tracking Improvements Over Time

Every experiment should be tracked with consistent metadata, enabling retrospective analysis:

```python
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

@dataclass
class ExperimentRecord:
    """A record of one improvement experiment."""
    experiment_id:     str
    date:              str
    hypothesis:        str              # "Switching to hybrid retrieval will improve Recall@5"
    change_made:       str              # "Replaced BM25-only with BM25+Dense RRF"
    component_changed: str              # "retriever" | "prompt" | "chunking" | "llm"
    baseline_metrics:  Dict[str, float] = field(default_factory=dict)
    new_metrics:       Dict[str, float] = field(default_factory=dict)
    delta:             Dict[str, float] = field(default_factory=dict)
    decision:          Optional[str]    = None  # "deploy" | "reject" | "further-test"
    notes:             str              = ""

    def compute_delta(self):
        self.delta = {
            k: self.new_metrics.get(k, 0) - self.baseline_metrics.get(k, 0)
            for k in self.baseline_metrics
        }

    def summarize(self):
        print(f"\nExperiment: {self.experiment_id}")
        print(f"Hypothesis: {self.hypothesis}")
        print(f"Change: {self.change_made}")
        print("\nMetric Changes:")
        for metric, delta in self.delta.items():
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"  {metric}: {self.baseline_metrics.get(metric):.3f} → "
                  f"{self.new_metrics.get(metric):.3f} ({direction}{abs(delta):.3f})")
        print(f"\nDecision: {self.decision}")
```

---

## 11. Common Failure Patterns and Systematic Diagnosis

The following table catalogs the most frequently observed RAG failure patterns, their diagnostic signatures, and recommended remediation:

| Failure Pattern | Observable Symptoms | Diagnostic Metric | Remediation |
|---|---|---|---|
| **Retrieval miss** | Correct answer not in retrieved docs; LLM says "information not found" when it should exist | Recall@k < 0.70 | Increase `top_k`; improve chunking; switch to dense/hybrid |
| **Retrieval noise** | Irrelevant documents in context; LLM answer wanders off-topic | Precision@k < 0.50 | Reduce `top_k`; add re-ranking; improve metadata filtering |
| **Context flooding** | Very long context; LLM misses relevant information in the middle | "Lost in middle" pattern in responses | Reduce `top_k`; improve chunk ranking; summarize context |
| **Grounding failure** | LLM uses training knowledge instead of context; answers sound right but aren't in docs | Faithfulness < 0.70 | Strengthen grounding instruction; lower temperature; add prohibition |
| **Citation hallucination** | LLM cites "Document 4" when only 3 were retrieved | Citation accuracy < 1.0 | Add citation validation; limit doc numbers in prompt |
| **Fallback hallucination** | LLM invents a plausible answer when context is insufficient instead of saying "I don't know" | Adversarial query recall | Rephrase fallback instruction; add "I cannot find..." as a positive target |
| **Knowledge base staleness** | Answers are correct for a past date but wrong for the current state | Rising user corrections | Implement document freshness scoring; schedule regular KB updates |
| **Vocabulary gap** | Relevant documents retrieved but using different terminology than the query | Low Recall@k for domain-specific queries | Fine-tune embedding model on domain data; add synonym expansion |
| **Chunk boundary artifact** | Answer straddles two chunks; neither chunk contains the complete answer | Low Recall@k for specific question types | Increase overlap; use hierarchical chunking |
| **Latency regression** | Response time increases beyond SLA | P99 latency > threshold | Profile pipeline; optimize ANN index; add caching layer |

---

## 12. Key Concepts Summary

| Concept | Definition |
|---|---|
| **Recall@k** | Fraction of all relevant documents found in the top-k retrieved results; the primary retrieval metric for RAG. |
| **Precision@k** | Fraction of the k retrieved documents that are relevant; measures retrieval noise. |
| **MRR (Mean Reciprocal Rank)** | Mean of 1/rank for the first relevant document; rewards systems that rank the best document highest. |
| **NDCG@k** | Normalized Discounted Cumulative Gain; a graded ranking metric that penalizes relevant documents at lower positions. |
| **Faithfulness** | The degree to which every claim in a generated response is directly supported by the retrieved context; the primary hallucination proxy. |
| **Answer Relevance** | The degree to which the generated response directly addresses the user's question. |
| **Context Precision** | The fraction of retrieved context that is actually relevant to answering the question. |
| **Context Recall** | The fraction of information needed to answer the question that is present in the retrieved context. |
| **RAGAS** | Retrieval Augmented Generation Assessment; an automated framework providing four metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall) requiring no human-labeled ground truth for two of them. |
| **LLM-as-Judge** | Using a frontier LLM to evaluate the outputs of a RAG system; faster than human evaluation but subject to systematic biases. |
| **Inter-annotator agreement (κ)** | Cohen's Kappa; measures consistency between human annotators; a prerequisite for valid human evaluation datasets. |
| **Drift detection** | Monitoring system quality over time and alerting when metrics deviate significantly from a baseline. |
| **Fallback rate** | The proportion of queries where the system's response indicates that the answer was not found in the knowledge base; a proxy for knowledge base coverage. |
| **Ground truth dataset** | A labeled set of (query, relevant_documents, correct_answer) triples used as the reference for retrieval and generation evaluation. |

---

## 13. Further Reading and Foundational Papers

### RAG Evaluation Frameworks

- **Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).** *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* [`arXiv:2309.15217`](https://arxiv.org/abs/2309.15217)  
  *(The RAGAS framework paper — the standard for automated RAG evaluation. Includes metric definitions and validation against human judgments.)*

- **Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2023).** *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* [`arXiv:2311.09476`](https://arxiv.org/abs/2311.09476)  
  *(An alternative to RAGAS; uses trained classifiers rather than LLM-as-judge for faster, cheaper evaluation.)*

### LLM-as-Judge

- **Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Gonzalez, J. E. (2023).** *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. [`arXiv:2306.05685`](https://arxiv.org/abs/2306.05685)  
  *(Systematic study of LLM-as-judge — when it works, when it fails, and how to mitigate biases.)*

### Faithfulness and Hallucination

- **Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020).** *On Faithfulness and Factuality in Abstractive Summarization.* ACL 2020. [`arXiv:2005.00661`](https://arxiv.org/abs/2005.00661)  
  *(Established the "faithfulness" concept — seminal reference for RAG faithfulness evaluation.)*

- **Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W. T., Koh, P. W., ... & Hajishirzi, H. (2023).** *FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. [`arXiv:2305.14251`](https://arxiv.org/abs/2305.14251)  
  *(Atomic claim-level faithfulness evaluation — the basis of production-grade faithfulness checkers.)*

### Information Retrieval Evaluation

- **Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).** *BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of Information Retrieval Models.* NeurIPS 2021. [`arXiv:2104.08663`](https://arxiv.org/abs/2104.08663)  
  *(The BEIR benchmark — the standard for evaluating retrievers across diverse domains.)*

- **Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022).** *MTEB: Massive Text Embedding Benchmark.* [`arXiv:2210.07316`](https://arxiv.org/abs/2210.07316)  
  *(MTEB — 58 tasks for embedding model evaluation; use to select the best embedding model for your domain.)*

### Human Evaluation

- **Artstein, R., & Poesio, M. (2008).** *Inter-Coder Agreement for Computational Linguistics.* Computational Linguistics. [`ACL Anthology`](https://aclanthology.org/J08-4004/)  
  *(Comprehensive reference on inter-annotator agreement measures, including Cohen's Kappa and Krippendorff's Alpha.)*

---

## 14. Review Questions

> **Difficulty guide:** ★ Foundational · ★★ Intermediate · ★★★ Advanced

### Conceptual Understanding

1. ★ A RAG system has three evaluation zones. Without referring to the text, describe what each zone measures, give one metric appropriate for each, and explain why a failure in Zone 1 can masquerade as a failure in Zone 3 on end-to-end evaluation.

2. ★ Compare Recall@k and Precision@k as retrieval metrics. In a RAG system where it is critical that no relevant document is missed (e.g., a legal compliance system), which metric should you optimize primarily? What is the risk of optimizing only that metric while ignoring the other?

3. ★★ RAGAS uses the harmonic mean to combine its four sub-metrics rather than the arithmetic mean. Construct a concrete numerical example demonstrating why the harmonic mean is more appropriate for this use case than the arithmetic mean.

4. ★★ Explain the "verbosity bias" in LLM-as-judge evaluation. How would this bias systematically skew a comparison between a system that gives concise, accurate answers and one that gives verbose answers with similar factual content? Propose a specific modification to the judge's rubric that would mitigate this bias.

5. ★★★ A practitioner argues: "We don't need to evaluate retrieval separately — if the end-to-end answer quality is good, the retriever must be working." Construct a detailed counterargument using a concrete scenario where end-to-end evaluation would fail to reveal a retrieval problem.

### Engineering Practice

6. ★ Implement a `compute_ndcg_at_k` function that handles **graded relevance** (where documents can be rated 0 = not relevant, 1 = partially relevant, 2 = highly relevant) rather than binary relevance. Show an example of a query where graded NDCG would produce meaningfully different results than binary NDCG.

7. ★★ You are building a production monitoring pipeline for a RAG customer service chatbot. Design a sampling strategy that ensures your daily evaluation sample (budget: 100 queries evaluated by LLM-as-judge) is representative of the full query distribution. Consider: query type distribution, time of day, high-error queries, and novel query types.

8. ★★ Your drift detector alerts that Faithfulness has dropped from 0.88 baseline to 0.79 over the past 30 days. List five possible root causes in order of likelihood for a production RAG system. For each, describe the diagnostic check you would perform to confirm or rule it out.

9. ★★★ Design a comprehensive A/B testing protocol to determine whether adding a re-ranking step (using a cross-encoder) after BM25+Dense hybrid retrieval improves end-to-end RAG quality. Your protocol must address: what to measure, how many samples are needed for statistical significance (assume you want power=0.80, α=0.05, and expect an effect size of 0.05 on RAGAS score), how to avoid contamination between variants, and what constitutes a "ship" decision.

10. ★★ Build a simple production logging and alerting system in Python. It should: (a) log each RAG request with the fields defined in §9.1; (b) compute a rolling 7-day Faithfulness score from logged evaluations; (c) trigger an alert function if the rolling score drops more than 0.05 below the 30-day moving average. Use only the Python standard library plus `numpy`.

### Design and Analysis

11. ★★ A company builds two RAG systems for the same knowledge base:
    - System A: dense retrieval only, top_k=10, no re-ranking  
    - System B: hybrid retrieval (BM25 + dense), top_k=5, cross-encoder re-ranking
    
    System A achieves Recall@10=0.92 and RAGAS=0.74. System B achieves Recall@5=0.88 and RAGAS=0.81. The product team wants to ship System A because it has better recall. Make the counterargument for System B. What additional information would you want before making a final recommendation?

12. ★★★ You are asked to design a **regression test suite** for a RAG system — a fixed set of queries that must pass before any change to the system (retriever, prompt, chunking, or LLM version) can be deployed. Design the suite: how many queries, what types, what pass/fail criteria for each metric, and what human review process is triggered when a query fails. Justify each design choice.

---

*End of Part 5 — Evaluation, Monitoring, and Continuous Improvement*

---

> **Up Next:** [Part 6 — Agentic RAG and Advanced Architectures](./06_Agentic_RAG_and_Advanced_Architectures.md)  
> In the final part, we move from single-retrieval RAG to multi-step, self-directed agentic systems — where the AI decides what to retrieve, when to retrieve again, and how to orchestrate multiple retrieval and reasoning steps to solve complex problems.

---

*This document is part of the **Retrieval-Augmented Generation: From Foundations to Production** learning series.*  
*Content adapted from DeepLearning.AI RAG course materials and enriched with academic references in RAG evaluation, LLM-as-judge methodology, human evaluation protocols, and production AI monitoring.*
