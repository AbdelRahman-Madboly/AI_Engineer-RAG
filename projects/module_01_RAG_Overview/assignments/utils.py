"""
utils.py — RAG Assignment helpers (C1M1)
=========================================
Adapted to work with your local LLM backend (ollama / gemini / together).
Set LLM_BACKEND in your .env file (defaults to "ollama").

Required files in the same directory:
  - news_data_dedup.csv
  - embeddings.joblib
  - .env  (with LLM_BACKEND and any required API keys)
"""

import json
import os
import numpy as np
import pandas as pd
from dateutil import parser
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import List, Dict

import ipywidgets as widgets
from IPython.display import display, Markdown

load_dotenv(override=True)

# ── Embedding model & pre-computed embeddings ─────────────────────────────────

# Downloads to HuggingFace cache on first run (~440 MB), then reuses it.
print("Loading embedding model... (first run may take a minute)")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

EMBEDDINGS = joblib.load("embeddings.joblib")
print("✅ Embedding model and embeddings loaded.")

# ── Dataset helpers ────────────────────────────────────────────────────────────

def format_date(date_string: str) -> str:
    """Parse any date string and return YYYY-MM-DD."""
    return parser.parse(date_string).strftime("%Y-%m-%d")


def read_dataframe(path: str) -> list:
    """Read the news CSV, normalise date columns, return list of dicts."""
    df = pd.read_csv(path)
    df["published_at"] = df["published_at"].apply(format_date)
    df["updated_at"]   = df["updated_at"].apply(format_date)
    return df.to_dict(orient="records")


def pprint(*args, **kwargs):
    """Pretty-print as indented JSON (matches original course interface)."""
    print(json.dumps(*args, indent=2))


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> list:
    """
    Encode the query and return the indices of the top_k most similar
    documents based on cosine similarity against the pre-computed embeddings.

    Parameters:
        query  : free-text search query
        top_k  : how many results to return

    Returns:
        list of int indices into NEWS_DATA
    """
    query_embedding    = embedding_model.encode(query)
    similarity_scores  = cosine_similarity(query_embedding.reshape(1, -1), EMBEDDINGS)[0]
    similarity_indices = np.argsort(-similarity_scores)
    return list(similarity_indices[:top_k])


# ── LLM backend (mirrors your existing utils.py) ──────────────────────────────

def _backend() -> str:
    return os.environ.get("LLM_BACKEND", "ollama").lower()


def _default_model() -> str:
    b = _backend()
    defaults = {
        "ollama":   os.environ.get("OLLAMA_MODEL",   "qwen2.5:7b"),
        "gemini":   os.environ.get("GEMINI_MODEL",   "gemini-2.0-flash"),
        "together": os.environ.get("TOGETHER_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    }
    return defaults.get(b, "qwen2.5:7b")


def _call_ollama(messages, model, max_tokens, temperature, top_p):
    import requests
    url     = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
    payload = {
        "model":   model,
        "messages": messages,
        "stream":  False,
        "think":   False,
        "options": {"num_predict": max_tokens},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature
    if top_p is not None:
        payload["options"]["top_p"] = top_p

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data    = resp.json()
    content = data["message"].get("content", "")

    if not content.strip():
        import re
        raw     = data["message"].get("thinking", "") or content
        content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return {"role": data["message"]["role"], "content": content}


def _call_gemini(messages, model, max_tokens, temperature, top_p):
    import google.generativeai as genai  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "\n❌ GEMINI_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at https://aistudio.google.com/apikey\n"
        )
    genai.configure(api_key=api_key)

    system_text, gemini_history = "", []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            gemini_history.append({"role": "user",  "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            gemini_history.append({"role": "model", "parts": [msg["content"]]})

    gen_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)
    if temperature is not None:
        gen_config.temperature = temperature
    if top_p is not None:
        gen_config.top_p = top_p

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_text or None,
        generation_config=gen_config,
    )

    if gemini_history and gemini_history[-1]["role"] == "user":
        last_user = gemini_history[-1]["parts"][0]
        history   = gemini_history[:-1]
    else:
        last_user, history = "", gemini_history

    chat     = gemini_model.start_chat(history=history)
    response = chat.send_message(last_user)
    return {"role": "assistant", "content": response.text}


def _call_together(messages, model, max_tokens, temperature, top_p):
    from together import Together

    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "\n❌ TOGETHER_API_KEY not set. Add it to your .env file.\n"
        )
    client  = Together(api_key=api_key)
    payload = {
        "model":      model,
        "messages":   messages,
        "max_tokens": max_tokens,
        "reasoning":  {"enabled": False},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    response = client.chat.completions.create(**payload).model_dump()
    role     = response["choices"][-1]["message"]["role"]
    role     = role.name.lower() if hasattr(role, "name") else str(role).lower()
    return {"role": role, "content": response["choices"][-1]["message"]["content"]}


def _dispatch(messages, model, max_tokens, temperature, top_p):
    backend = _backend()
    if backend == "ollama":
        return _call_ollama(messages, model, max_tokens, temperature, top_p)
    elif backend == "gemini":
        return _call_gemini(messages, model, max_tokens, temperature, top_p)
    elif backend == "together":
        return _call_together(messages, model, max_tokens, temperature, top_p)
    else:
        raise ValueError(f"Unknown LLM_BACKEND='{backend}'. Choose: ollama, gemini, together")


def generate_with_single_input(
    prompt: str,
    role: str = "user",
    max_tokens: int = 500,
    temperature: float = None,
    top_p: float = None,
    model: str = None,
) -> dict:
    """
    Send a single prompt to the configured LLM backend.

    Parameters:
        prompt      : text to send
        role        : 'user', 'system', or 'assistant'
        max_tokens  : max response tokens
        temperature : randomness (None = model default)
        top_p       : nucleus sampling (None = model default)
        model       : override model name

    Returns:
        {'role': 'assistant', 'content': '<response>'}
    """
    if model is None:
        model = _default_model()
    messages = [{"role": role, "content": prompt}]
    return _dispatch(messages, model, max_tokens, temperature, top_p)


# ── Interactive widget (side-by-side RAG vs no-RAG) ───────────────────────────

def display_widget(llm_call_func):
    """Display an interactive widget to compare RAG vs non-RAG responses."""

    def on_button_click(b):
        output1.clear_output()
        output2.clear_output()
        status_output.clear_output()
        status_output.append_stdout("Generating...\n")
        query  = query_input.value
        top_k  = slider.value
        prompt = prompt_input.value.strip() or None

        response1 = llm_call_func(query, use_rag=True,  top_k=top_k, prompt=prompt)
        response2 = llm_call_func(query, use_rag=False, top_k=top_k, prompt=prompt)

        with output1:
            display(Markdown(response1))
        with output2:
            display(Markdown(response2))
        status_output.clear_output()

    query_input = widgets.Text(
        description="Query:",
        placeholder="Type your query here",
        layout=widgets.Layout(width="100%"),
    )
    prompt_input = widgets.Textarea(
        description="Augmented prompt layout:",
        placeholder=(
            "Optional: type a custom prompt layout.\n"
            "Use {query} and {documents} as placeholders.\n"
            "Example:\nAnswer this: {query}\nContext: {documents}"
        ),
        layout=widgets.Layout(width="100%", height="100px"),
        style={"description_width": "initial"},
    )
    slider = widgets.IntSlider(
        value=5, min=1, max=20, step=1,
        description="Top K:",
        style={"description_width": "initial"},
    )
    output1      = widgets.Output(layout={"border": "1px solid #ccc", "width": "45%"})
    output2      = widgets.Output(layout={"border": "1px solid #ccc", "width": "45%"})
    status_output = widgets.Output()

    submit_button = widgets.Button(description="Get Responses")
    submit_button.on_click(on_button_click)

    label1 = widgets.Label(value="With RAG",    layout={"width": "45%"})
    label2 = widgets.Label(value="Without RAG", layout={"width": "45%"})

    for out in (output1, output2):
        out.layout.margin  = "5px"
        out.layout.height  = "300px"
        out.layout.padding = "10px"
        out.layout.overflow = "auto"

    display(query_input, prompt_input, slider, submit_button, status_output)
    display(widgets.HBox([label1,   label2],   layout={"justify_content": "space-between"}))
    display(widgets.HBox([output1, output2],   layout={"justify_content": "space-between"}))


# ── Compatibility stubs (kept so old imports don't break) ─────────────────────

def get_together_key():  return os.environ.get("TOGETHER_API_KEY", "")
def get_proxy_url():     return os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/")
def get_proxy_headers(): return {"Authorization": os.environ.get("TOGETHER_API_KEY", "")}