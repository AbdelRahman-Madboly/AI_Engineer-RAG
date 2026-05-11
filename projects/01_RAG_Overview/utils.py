"""
utils.py — LLM helper functions for the RAG course
====================================================
Supports three backends:
    - together  → Together.ai API (original)
    - ollama    → Local Ollama server (http://localhost:11434)
    - gemini    → Google Gemini API

Set LLM_BACKEND in your .env file (defaults to "ollama"):

    # .env
    LLM_BACKEND=ollama          # use local Ollama
    LLM_BACKEND=gemini          # use Google Gemini
    LLM_BACKEND=together        # use Together.ai (needs TOGETHER_API_KEY)

    GEMINI_API_KEY=your_key     # required only for gemini backend
    TOGETHER_API_KEY=your_key   # required only for together backend

    # Optional: override default models per backend
    OLLAMA_MODEL=qwen2.5:7b
    GEMINI_MODEL=gemini-2.0-flash
    TOGETHER_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(override=True)


def _backend() -> str:
    """Read LLM_BACKEND fresh so .env changes take effect without restarting."""
    return os.environ.get("LLM_BACKEND", "ollama").lower()


def _default_model() -> str:
    """Read model name fresh every call."""
    b = _backend()
    defaults = {
        "ollama":   os.environ.get("OLLAMA_MODEL",   "qwen3.5:4B"),
        "gemini":   os.environ.get("GEMINI_MODEL",   "gemini-2.0-flash"),
        "together": os.environ.get("TOGETHER_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    }
    return defaults.get(b, "qwen3.5:4B")


# ── Backend implementations ───────────────────────────────────────────────────

def _call_ollama(messages: list, model: str, max_tokens: int,
                 temperature: float, top_p: float, **kwargs) -> dict:
    """Call local Ollama server via its OpenAI-compatible endpoint."""
    import requests

    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,          # disable Qwen3 thinking tokens
        "options": {"num_predict": max_tokens},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature
    if top_p is not None:
        payload["options"]["top_p"] = top_p

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    content = data["message"].get("content", "")

    # Qwen3 thinking models: if content is empty, answer may be in 'thinking' field
    # or wrapped in <think>...</think> tags — strip them out
    if not content.strip():
        import re
        thinking_raw = data["message"].get("thinking", "") or content
        content = re.sub(r"<think>.*?</think>", "", thinking_raw, flags=re.DOTALL).strip()

    return {
        "role":    data["message"]["role"],
        "content": content,
    }


def _call_gemini(messages: list, model: str, max_tokens: int,
                 temperature: float, top_p: float, **kwargs) -> dict:
    """Call Google Gemini via the generativeai SDK."""
    import google.generativeai as genai # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "\n\n❌ GEMINI_API_KEY not found.\n"
            "Fix: add  GEMINI_API_KEY=your_key  to your .env file.\n"
            "Get a free key at https://aistudio.google.com/apikey\n"
        )
    genai.configure(api_key=api_key)

    # Convert messages to Gemini format
    # Gemini uses 'user' / 'model' roles; system messages become the first user turn
    system_text = ""
    gemini_history = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_text = content  # handled separately
        elif role == "user":
            gemini_history.append({"role": "user", "parts": [content]})
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

    # Last message must be user — start chat with history minus last turn
    if gemini_history and gemini_history[-1]["role"] == "user":
        last_user = gemini_history[-1]["parts"][0]
        history = gemini_history[:-1]
    else:
        last_user = ""
        history = gemini_history

    chat = gemini_model.start_chat(history=history)
    response = chat.send_message(last_user)

    return {
        "role":    "assistant",
        "content": response.text,
    }


def _call_together(messages: list, model: str, max_tokens: int,
                   temperature: float, top_p: float, **kwargs) -> dict:
    """Call Together.ai (original backend)."""
    from together import Together

    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "\n\n❌ TOGETHER_API_KEY not found.\n"
            "Fix: add  TOGETHER_API_KEY=your_key  to your .env file.\n"
        )

    client = Together(api_key=api_key)
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

    response = client.chat.completions.create(**payload).model_dump()
    role = response["choices"][-1]["message"]["role"]
    role = role.name.lower() if hasattr(role, "name") else str(role).lower()
    return {
        "role":    role,
        "content": response["choices"][-1]["message"]["content"],
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _dispatch(messages: list, model: str, max_tokens: int,
              temperature: float, top_p: float, **kwargs) -> dict:
    """Route the call to the correct backend."""
    backend = _backend()
    if backend == "ollama":
        return _call_ollama(messages, model, max_tokens, temperature, top_p, **kwargs)
    elif backend == "gemini":
        return _call_gemini(messages, model, max_tokens, temperature, top_p, **kwargs)
    elif backend == "together":
        return _call_together(messages, model, max_tokens, temperature, top_p, **kwargs)
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND='{backend}'. "
            "Choose one of: ollama, gemini, together"
        )


# ── Public API (same interface as original utils.py) ──────────────────────────

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
    Send a single prompt to an LLM and get one response.

    Args:
        prompt      : The text to send to the model.
        role        : 'user', 'system', or 'assistant'. Defaults to 'user'.
        max_tokens  : Maximum tokens in the response. Default 500.
        temperature : 0.0 = deterministic, 1.0+ = creative. None = model default.
        top_p       : Nucleus sampling. None = model default.
        model       : Model name. Defaults to the backend's default model.

    Returns:
        {'role': 'assistant', 'content': '<response text>'}

    Example:
        output = generate_with_single_input("What is the capital of France?")
        print(output['content'])  # → Paris
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
    Send a full conversation history to an LLM and get the next response.

    Args:
        messages    : List of {'role': ..., 'content': ...} dicts.
                      Roles: 'system', 'user', 'assistant'.
        max_tokens  : Maximum tokens in the response. Default 500.
        temperature : Randomness control.
        top_p       : Nucleus sampling.
        model       : Model name. Defaults to the backend's default model.

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


# ── Compatibility helpers ─────────────────────────────────────────────────────

def get_together_key() -> str:
    return os.environ.get("TOGETHER_API_KEY", "")

def get_proxy_url() -> str:
    return os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/")

def get_proxy_headers() -> dict:
    return {"Authorization": os.environ.get("TOGETHER_API_KEY", "")}