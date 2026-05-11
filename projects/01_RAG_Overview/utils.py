"""
utils.py — LLM helper functions for the RAG course
====================================================
Wraps Together.ai API calls into two clean functions used across all notebooks:

    generate_with_single_input()   — one prompt → one response
    generate_with_multiple_input() — full conversation history → one response

Local setup:
    - Requires TOGETHER_API_KEY in your .env file at the repo root
    - Load it before importing: from dotenv import load_dotenv; load_dotenv()
"""

import os
import json
import requests
from typing import List, Dict
from dotenv import load_dotenv

# ── Auto-load .env when this file is imported ─────────────────────────────────
# Looks for .env in the current folder and all parent folders
load_dotenv()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """
    Reads TOGETHER_API_KEY from environment.
    Raises a clear error if the key is missing so you know exactly what to fix.
    """
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "\n\n❌ TOGETHER_API_KEY not found.\n"
            "Fix: create a .env file at the root of your repo with:\n"
            "     TOGETHER_API_KEY=your_key_here\n"
            "Then run: from dotenv import load_dotenv; load_dotenv()\n"
        )
    return key


def _call_api(payload: dict) -> dict:
    """
    Sends the payload to Together.ai and returns the parsed JSON response.
    Raises descriptive exceptions on HTTP errors or bad JSON.
    """
    from together import Together

    api_key = _get_api_key()
    client = Together(api_key=api_key)

    # together SDK returns a Pydantic model — convert to dict
    response = client.chat.completions.create(**payload).model_dump()

    # Normalize the role field (SDK returns an enum, we want a plain string)
    response["choices"][-1]["message"]["role"] = (
        response["choices"][-1]["message"]["role"].name.lower()
        if hasattr(response["choices"][-1]["message"]["role"], "name")
        else str(response["choices"][-1]["message"]["role"]).lower()
    )
    return response


def _extract_output(response: dict) -> dict:
    """
    Pulls role and content out of the raw API response dict.
    Returns: {'role': 'assistant', 'content': '...'}
    """
    try:
        return {
            "role":    response["choices"][-1]["message"]["role"],
            "content": response["choices"][-1]["message"]["content"],
        }
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Unexpected API response shape. Could not extract output.\n"
            f"Error: {e}\nFull response: {response}"
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_with_single_input(
    prompt: str,
    role: str = "user",
    max_tokens: int = 500,
    temperature: float = None,
    top_p: float = None,
    model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
    **kwargs,
) -> dict:
    """
    Send a single prompt to an LLM and get one response.

    Args:
        prompt      : The text to send to the model.
        role        : Who is sending the message — 'user', 'system', or 'assistant'.
                      Defaults to 'user' (the most common case).
        max_tokens  : Maximum tokens in the response. Default 500.
        temperature : Controls randomness. 0.0 = deterministic, 1.0+ = creative.
                      Leave as None to use the model's default.
        top_p       : Nucleus sampling threshold. Leave as None for model default.
        model       : Together.ai model name.
        **kwargs    : Any extra parameters passed directly to the API.

    Returns:
        dict with two keys:
            'role'    → always 'assistant'
            'content' → the model's response text

    Example:
        output = generate_with_single_input("What is the capital of France?")
        print(output['content'])  # → Paris
    """
    # Build the messages list from the single prompt
    messages = [{"role": role, "content": prompt}]

    payload = {
        "model":    model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},  # disable chain-of-thought reasoning tokens
        **kwargs,
    }

    # Only add optional params if explicitly set — avoids API errors for None values
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    response = _call_api(payload)
    return _extract_output(response)


def generate_with_multiple_input(
    messages: List[Dict],
    max_tokens: int = 500,
    temperature: float = None,
    top_p: float = None,
    model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
    **kwargs,
) -> dict:
    """
    Send a full conversation history to an LLM and get the next response.
    Use this for multi-turn chat or when you need a system message.

    Args:
        messages    : List of message dicts, each with 'role' and 'content'.
                      Roles: 'system', 'user', 'assistant'.
                      The list represents the full conversation so far.
        max_tokens  : Maximum tokens in the response. Default 500.
        temperature : Controls randomness. 0.0 = deterministic, 1.0+ = creative.
        top_p       : Nucleus sampling. Leave as None for model default.
        model       : Together.ai model name.
        **kwargs    : Any extra parameters passed directly to the API.

    Returns:
        dict with two keys:
            'role'    → always 'assistant'
            'content' → the model's response text

    Example:
        messages = [
            {'role': 'system',    'content': 'You are a helpful assistant.'},
            {'role': 'user',      'content': 'What is RAG?'},
            {'role': 'assistant', 'content': 'RAG is Retrieval Augmented Generation.'},
            {'role': 'user',      'content': 'What problem does it solve?'},
        ]
        output = generate_with_multiple_input(messages)
        print(output['content'])
    """
    payload = {
        "model":    model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        **kwargs,
    }

    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    response = _call_api(payload)
    return _extract_output(response)


# ── Compatibility helpers (match the course's original utils.py interface) ─────

def get_together_key() -> str:
    """Returns the Together.ai API key from environment. Used by some notebooks."""
    return _get_api_key()


def get_proxy_url() -> str:
    """
    Returns the API base URL.
    In the Coursera environment this pointed to a proxy — locally it's Together.ai directly.
    """
    return os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/")


def get_proxy_headers() -> dict:
    """Returns auth headers. Kept for compatibility with OpenAI-client notebooks."""
    return {"Authorization": os.environ.get("TOGETHER_API_KEY", "")}