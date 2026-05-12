"""
utils.py — LLM helper functions for the RAG course
====================================================
Supports three backends:
    - ollama    → Local Ollama server (default, no API key needed)
    - gemini    → Google Gemini API   (free tier available)
    - together  → Together.ai API     (paid)

HOW TO CONFIGURE — edit the .env file at D:\\AI_Engineer-RAG\\.env:

    LLM_BACKEND=ollama              ← pick one

    OLLAMA_MODEL=qwen2.5:7b         ← model to use with Ollama
    OLLAMA_HOST=http://localhost:11434

    GEMINI_API_KEY=your_key_here    ← required only for gemini backend
    GEMINI_MODEL=gemini-2.0-flash

    TOGETHER_API_KEY=your_key_here  ← required only for together backend
    TOGETHER_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo

FUNCTIONS PROVIDED:
    generate_with_single_input(prompt, ...)   → one question, one answer
    generate_with_multiple_input(messages, .) → full conversation history
    Both return: {'role': 'assistant', 'content': '<response text>'}
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

# Re-load .env every import so changes take effect without restarting Jupyter
load_dotenv(override=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _backend() -> str:
    """Read LLM_BACKEND from environment (re-read each call so .env changes work)."""
    return os.environ.get("LLM_BACKEND", "ollama").lower().strip()


def _default_model() -> str:
    """Return the default model name for the active backend."""
    b = _backend()
    defaults = {
        "ollama":   os.environ.get("OLLAMA_MODEL",   "qwen2.5:7b"),
        "gemini":   os.environ.get("GEMINI_MODEL",   "gemini-2.0-flash"),
        "together": os.environ.get("TOGETHER_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    }
    return defaults.get(b, "qwen2.5:7b")


# ── Ollama backend ────────────────────────────────────────────────────────────

def _call_ollama(messages: list, model: str, max_tokens: int,
                 temperature, top_p, **kwargs) -> dict:
    """
    Call a local Ollama server.
    Requires: `ollama serve` running in a separate terminal.
    Install a model first: ollama pull qwen2.5:7b
    """
    import requests

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url  = host.rstrip("/") + "/api/chat"

    options = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p

    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  options,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "\n\n❌ Cannot reach Ollama at " + host + "\n"
            "Fix: open a terminal and run:  ollama serve\n"
            "Then re-run this cell.\n"
        )

    data    = resp.json()
    content = data["message"].get("content", "").strip()

    # Some Qwen3 thinking models wrap output in <think>…</think> — strip it
    if "<think>" in content:
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    return {"role": data["message"]["role"], "content": content}


# ── Gemini backend ────────────────────────────────────────────────────────────

def _call_gemini(messages: list, model: str, max_tokens: int,
                 temperature, top_p, **kwargs) -> dict:
    """
    Call Google Gemini via the google-generativeai SDK.
    Install: pip install google-generativeai
    Get a free key at: https://aistudio.google.com/apikey
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        raise ImportError(
            "\n\n❌ google-generativeai is not installed.\n"
            "Fix: run this in your terminal:\n"
            "    conda activate rag-env\n"
            "    pip install google-generativeai\n"
        )

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_key_here":
        raise EnvironmentError(
            "\n\n❌ GEMINI_API_KEY not set.\n"
            "Fix: open D:\\AI_Engineer-RAG\\.env and set:\n"
            "    GEMINI_API_KEY=your_actual_key\n"
            "Get a free key at: https://aistudio.google.com/apikey\n"
        )

    genai.configure(api_key=api_key)

    # Separate system message from conversation history
    system_text    = ""
    gemini_history = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "system":
            system_text = content
        elif role == "user":
            gemini_history.append({"role": "user",  "parts": [content]})
        elif role == "assistant":
            gemini_history.append({"role": "model", "parts": [content]})

    gen_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)
    if temperature is not None:
        gen_config.temperature = temperature
    if top_p is not None:
        gen_config.top_p = top_p

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_text if system_text else None,
        generation_config=gen_config,
    )

    # The last message must be a user turn
    if gemini_history and gemini_history[-1]["role"] == "user":
        last_user_text = gemini_history[-1]["parts"][0]
        history        = gemini_history[:-1]
    else:
        last_user_text = ""
        history        = gemini_history

    chat     = gemini_model.start_chat(history=history)
    response = chat.send_message(last_user_text)

    return {"role": "assistant", "content": response.text}


# ── Together.ai backend ───────────────────────────────────────────────────────

def _call_together(messages: list, model: str, max_tokens: int,
                   temperature, top_p, **kwargs) -> dict:
    """
    Call Together.ai.
    Install: pip install together
    Get a key at: https://api.together.ai
    """
    try:
        from together import Together
    except ImportError:
        raise ImportError(
            "\n\n❌ together package is not installed.\n"
            "Fix: run this in your terminal:\n"
            "    conda activate rag-env\n"
            "    pip install together\n"
        )

    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "\n\n❌ TOGETHER_API_KEY not set.\n"
            "Fix: open D:\\AI_Engineer-RAG\\.env and set:\n"
            "    TOGETHER_API_KEY=your_actual_key\n"
        )

    client  = Together(api_key=api_key)
    payload = {
        "model":      model,
        "messages":   messages,
        "max_tokens": max_tokens,
        "reasoning":  {"enabled": False},
        **kwargs,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    result  = client.chat.completions.create(**payload).model_dump()
    message = result["choices"][-1]["message"]

    role = message["role"]
    if hasattr(role, "name"):        # Together returns an enum on some versions
        role = role.name.lower()
    role = str(role).lower()

    return {"role": role, "content": message["content"]}


# ── Internal dispatcher ───────────────────────────────────────────────────────

def _dispatch(messages: list, model: str, max_tokens: int,
              temperature, top_p, **kwargs) -> dict:
    """Route the request to the correct backend based on LLM_BACKEND in .env."""
    backend = _backend()
    if backend == "ollama":
        return _call_ollama(messages, model, max_tokens, temperature, top_p, **kwargs)
    elif backend == "gemini":
        return _call_gemini(messages, model, max_tokens, temperature, top_p, **kwargs)
    elif backend == "together":
        return _call_together(messages, model, max_tokens, temperature, top_p, **kwargs)
    else:
        raise ValueError(
            f"\n\n❌ Unknown LLM_BACKEND='{backend}' in your .env file.\n"
            "Valid options: ollama, gemini, together\n"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_with_single_input(
    prompt: str,
    role: str = "user",
    max_tokens: int = 500,
    temperature: float = None,
    top_p: float = None,
    model: str = None,
    **kwargs,
) -> dict:
    """
    Send a single prompt to the LLM and receive one response.

    Args:
        prompt      : The text to send.
        role        : Who is sending — 'user' (default), 'system', or 'assistant'.
        max_tokens  : Maximum response length in tokens. Default 500.
        temperature : 0.0 = deterministic · 1.0 = creative. None = model default.
        top_p       : Nucleus sampling cutoff. None = model default.
        model       : Model name override. Defaults to backend default in .env.

    Returns:
        {'role': 'assistant', 'content': '<response text>'}

    Example:
        output = generate_with_single_input("What is the capital of France?")
        print(output['content'])   # Paris
    """
    if model is None:
        model = _default_model()
    messages = [{"role": role, "content": prompt}]
    return _dispatch(messages, model, max_tokens, temperature, top_p, **kwargs)


def generate_with_multiple_input(
    messages: List[Dict],
    max_tokens: int = 500,
    temperature: float = None,
    top_p: float = None,
    model: str = None,
    **kwargs,
) -> dict:
    """
    Send a full conversation history to the LLM and receive the next response.

    Args:
        messages    : List of {'role': ..., 'content': ...} dicts.
                      Roles: 'system', 'user', 'assistant'.
        max_tokens  : Maximum response length in tokens. Default 500.
        temperature : Randomness control.
        top_p       : Nucleus sampling.
        model       : Model name override. Defaults to backend default in .env.

    Returns:
        {'role': 'assistant', 'content': '<response text>'}

    Example:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user',   'content': 'What is RAG?'},
        ]
        output = generate_with_multiple_input(messages)
        print(output['content'])
    """
    if model is None:
        model = _default_model()
    return _dispatch(messages, model, max_tokens, temperature, top_p, **kwargs)


# ── Legacy compatibility stubs ────────────────────────────────────────────────
# These exist so any notebook that imports the original Together-only utils.py
# still works without modification.

def get_together_key() -> str:
    """Return the Together API key from environment (legacy helper)."""
    return os.environ.get("TOGETHER_API_KEY", "")

def get_proxy_url() -> str:
    """Return the Together base URL (legacy helper)."""
    return os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/")

def get_proxy_headers() -> dict:
    """Return Together auth headers (legacy helper)."""
    return {"Authorization": os.environ.get("TOGETHER_API_KEY", "")}
