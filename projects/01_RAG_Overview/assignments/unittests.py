"""
unittests.py — Local test suite for C1M1 Assignment
=====================================================
Replaces the Coursera dlai_grader with simple self-contained checks.
Run each test by calling it with your function, e.g.:

    unittests.test_get_relevant_data(get_relevant_data)
    unittests.test_format_relevant_data(format_relevant_data)
"""

import pandas as pd
from types import FunctionType

# ── Load data once at import time ─────────────────────────────────────────────
NEWS_DATA = pd.read_csv("./news_data_dedup.csv").to_dict(orient="records")

# ── Minimal test-case helpers ─────────────────────────────────────────────────

class _Case:
    def __init__(self):
        self.failed = False
        self.msg    = ""
        self.want   = ""
        self.got    = ""

def _print_feedback(cases: list):
    failures = [c for c in cases if c.failed]
    if not failures:
        print("✅ All tests passed!")
        return
    for c in failures:
        print(f"❌ {c.msg}")
        if c.want:
            print(f"   Expected : {c.want}")
        if c.got:
            print(f"   Got      : {c.got}")

# ── Test: get_relevant_data ───────────────────────────────────────────────────

def test_get_relevant_data(learner_func):
    """
    Tests that get_relevant_data:
      1. Returns a list
      2. Returns exactly top_k items
      3. Returns the correct documents (matched by guid)
    """
    cases = []

    # Type check
    t = _Case()
    if not isinstance(learner_func, FunctionType):
        t.failed = True
        t.msg    = "get_relevant_data is not a function"
        t.want   = str(FunctionType)
        t.got    = str(type(learner_func))
        _print_feedback([t])
        return

    query  = "This is a test query"
    top_k  = 3

    # Expected guids for this query (determined by the fixed embeddings)
    expected_guids = {
        "e78d129bee161f6416d20ab0ae66f5a9",
        "79c0f5715f341c65c0d9abd4890f35c0",
        "2de17d633142978a5409df1445ad538c",
    }

    # Run learner function
    try:
        output = learner_func(query, top_k=top_k)
    except Exception as e:
        t = _Case()
        t.failed = True
        t.msg    = f"get_relevant_data raised an exception"
        t.got    = f"Exception: {e}"
        _print_feedback([t])
        return

    # Return type
    t = _Case()
    if not isinstance(output, list):
        t.failed = True
        t.msg    = "Return value is not a list"
        t.want   = "list"
        t.got    = str(type(output))
        cases.append(t)
        _print_feedback(cases)
        return
    cases.append(t)

    # Length
    t = _Case()
    if len(output) != top_k:
        t.failed = True
        t.msg    = f"Wrong number of results for top_k={top_k}"
        t.want   = str(top_k)
        t.got    = str(len(output))
    cases.append(t)

    # Correct documents (by guid)
    t = _Case()
    try:
        output_guids = {d["guid"] for d in output}
    except Exception as e:
        t.failed = True
        t.msg    = "Each result dict must have a 'guid' key"
        t.got    = f"Exception: {e}"
        cases.append(t)
        _print_feedback(cases)
        return

    if output_guids != expected_guids:
        t.failed = True
        t.msg    = f"Wrong documents retrieved for query='{query}', top_k={top_k}"
        t.want   = str(expected_guids)
        t.got    = str(output_guids)
    cases.append(t)

    _print_feedback(cases)


# ── Test: format_relevant_data ────────────────────────────────────────────────

def test_format_relevant_data(learner_func):
    """
    Tests that format_relevant_data:
      1. Returns a string
      2. Contains the required keywords: title, url, published_at, description
      3. Each keyword appears exactly once per document
    """
    cases = []

    t = _Case()
    if not isinstance(learner_func, FunctionType):
        t.failed = True
        t.msg    = "format_relevant_data is not a function"
        t.want   = str(FunctionType)
        t.got    = str(type(learner_func))
        _print_feedback([t])
        return

    relevant_data = NEWS_DATA[5:9]   # 4 documents, same slice the grader uses

    try:
        result = learner_func(relevant_data)
    except Exception as e:
        t.failed = True
        t.msg    = "format_relevant_data raised an exception"
        t.got    = f"Exception: {e}"
        _print_feedback([t])
        return

    # Must return a string
    t = _Case()
    if not isinstance(result, str):
        t.failed = True
        t.msg    = "Return value must be a string"
        t.want   = "str"
        t.got    = str(type(result))
        cases.append(t)
        _print_feedback(cases)
        return
    cases.append(t)

    result_lower = result.lower()
    required_keywords = ["title", "url", "published_at", "description"]

    for keyword in required_keywords:
        # Keyword present at all?
        t = _Case()
        if keyword not in result_lower:
            t.failed = True
            t.msg    = f"Keyword '{keyword}' not found in the formatted output"
            t.want   = f"'{keyword}' must appear in the output"
        cases.append(t)

        # Keyword appears once per document
        t = _Case()
        count = result_lower.count(keyword)
        if count != len(relevant_data):
            t.failed = True
            t.msg    = f"'{keyword}' should appear {len(relevant_data)} times (once per document)"
            t.want   = str(len(relevant_data))
            t.got    = str(count)
        cases.append(t)

    _print_feedback(cases)