"""Unified LLM provider abstraction with response caching and budget guardrails.

Extracted from the DataCollector research platform (src/providers/llm.py) and made
standalone: the only external dependency is `openai` and/or `requests`, both optional
depending on which provider you use.

Usage:
    from csl_decomp import llm
    llm.set_config(provider="openai", model="gpt-4o")
    response = llm.call("Summarize this issue...", max_tokens=400, caller="extract")
    print(response.content, response.tokens)

Providers: openai, azure_openai, anthropic, gemini, groq, openrouter, nim,
cerebras, sambanova, ollama, mock. Select via set_config() or the LLM_PROVIDER /
LLM_MODEL environment variables.

Caching: every call is cached to disk keyed by SHA-256 of the full request payload
(provider, model, prompt, max_tokens, temperature, caller). Re-running an experiment
is free and byte-identical. Cache dir defaults to .llm_cache/ (override with
LLM_CACHE_DIR or set_config(cache_dir=...)).

Budgets: set_config(max_calls=..., max_tokens=..., budget_usd=...) makes call()
raise "LLM budget exceeded" once the cap is hit; pair with BudgetTracker for
graceful fallbacks.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover - soft dependency
    requests = None  # type: ignore

TRUE_VALUES = {"1", "true", "yes", "on"}


def _truthy(value: Optional[str], default: str = "0") -> bool:
    raw = default if value is None else value
    return raw.strip().lower() in TRUE_VALUES


_FORCE_JSON_SYSTEM = (
    "Your response MUST be a single valid JSON object — nothing else. "
    "Start your response with `{` and end with `}`. "
    "Do not write any prose, markdown, analysis, or explanation outside the JSON. "
    "Never use code fences. The JSON payload is consumed directly by a machine harness."
)


@dataclass
class LLMRuntimeConfig:
    provider: str = os.getenv("LLM_PROVIDER", "mock")
    model: str = os.getenv("LLM_MODEL", "mock-model")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.25"))
    max_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    budget_usd: Optional[float] = None
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("LLM_CACHE_DIR", ".llm_cache")))
    track_cost: bool = False
    timeout: int = 90
    force_json: bool = _truthy(os.getenv("LLM_FORCE_JSON"))


CONFIG = LLMRuntimeConfig()
CALLS_USED = 0
TOKENS_USED = 0
BUDGET_USED = 0.0

# Per-caller usage buckets ({caller: {tokens, time, calls}}), reported by get_usage().
stats: Dict[str, Dict[str, float]] = {}


@dataclass
class LLMUsage:
    tokens_input: int = 0
    tokens_output: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def to_meta(self) -> Dict[str, Any]:
        return {
            "tokens_input": int(self.tokens_input),
            "tokens_output": int(self.tokens_output),
            "total_tokens": int(self.total_tokens),
            "cost_usd": float(self.cost_usd),
        }


@dataclass
class LLMResponse:
    content: str
    tokens: int
    elapsed: float
    budget_exceeded: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "content": self.content,
                "tokens": self.tokens,
                "elapsed": self.elapsed,
                "budget_exceeded": self.budget_exceeded,
                "meta": self.meta,
            }
        )

    @staticmethod
    def from_json(payload: str) -> "LLMResponse":
        data = json.loads(payload)
        return LLMResponse(
            content=data.get("content", ""),
            tokens=data.get("tokens", 0),
            elapsed=data.get("elapsed", 0.0),
            budget_exceeded=data.get("budget_exceeded", False),
            meta=data.get("meta", {}),
        )


def set_config(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    cache_enabled: Optional[bool] = None,
    cache_dir: Optional[Path] = None,
    max_calls: Optional[int] = None,
    max_tokens: Optional[int] = None,
    budget_usd: Optional[float] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
) -> None:
    """Update global LLM configuration."""

    global CONFIG
    if provider:
        CONFIG.provider = provider
    if model:
        CONFIG.model = model
    if cache_enabled is not None:
        CONFIG.cache_enabled = cache_enabled
    if cache_dir is not None:
        CONFIG.cache_dir = Path(cache_dir)
    if max_calls is not None:
        CONFIG.max_calls = max_calls
    if max_tokens is not None:
        CONFIG.max_tokens = max_tokens
    if budget_usd is not None:
        CONFIG.budget_usd = budget_usd
    if temperature is not None:
        CONFIG.temperature = temperature
    if timeout is not None:
        CONFIG.timeout = timeout


def _cache_path(key: str) -> Path:
    target_dir = CONFIG.cache_dir / CONFIG.provider / CONFIG.model
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{key}.json"


def _budget_available(tokens: int) -> bool:
    if CONFIG.max_calls is not None and CALLS_USED >= CONFIG.max_calls:
        return False
    if CONFIG.max_tokens is not None and TOKENS_USED + tokens > CONFIG.max_tokens:
        return False
    if CONFIG.budget_usd is not None and BUDGET_USED >= CONFIG.budget_usd:
        return False
    return True


def _record_usage(tokens: int, elapsed: float, caller: str, cost: float = 0.0) -> None:
    global CALLS_USED, TOKENS_USED, BUDGET_USED
    CALLS_USED += 1
    TOKENS_USED += tokens
    BUDGET_USED += cost
    caller_bucket = stats.setdefault(caller, {"tokens": 0.0, "time": 0.0, "calls": 0.0})
    caller_bucket["tokens"] += float(tokens)
    caller_bucket["time"] += float(elapsed)
    caller_bucket["calls"] += 1.0


def _hash_payload(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _openai_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("openai>=1.0 is required for the OpenAI provider.") from exc

    base_url = os.getenv("OPENAI_BASE_URL")
    configured_model = (CONFIG.model or "").strip()
    env_model = os.getenv("OPENAI_MODEL", "").strip()
    default_model = "gpt-5"
    if configured_model and configured_model != "mock-model":
        model = configured_model
    elif env_model:
        model = env_model
    else:
        model = default_model

    client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": CONFIG.timeout}
    if base_url:
        client_kwargs["base_url"] = base_url.rstrip("/")
    client = OpenAI(**client_kwargs)

    requested_max = max_tokens if max_tokens is not None else 512
    if requested_max < 16:
        requested_max = 16
    model_lower = model.lower()
    reasoning_prefixes = ("o1", "o3", "o4")
    supports_temperature = not any(model_lower.startswith(prefix) for prefix in reasoning_prefixes)
    request_payload = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": requested_max,
    }
    if supports_temperature:
        request_payload["temperature"] = temperature

    def _extract_text(response: Any) -> str:
        text_blob = getattr(response, "output_text", None)
        if text_blob:
            return text_blob
        collected: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for block in getattr(item, "content", []) or []:
                if getattr(block, "type", None) != "output_text":
                    continue
                text_value = getattr(block, "text", "")
                if isinstance(text_value, list):
                    collected.append("".join(str(part) for part in text_value))
                elif text_value:
                    collected.append(str(text_value))
        return "".join(collected).strip()

    try:
        response = client.responses.create(**request_payload)
    except Exception as exc:  # pragma: no cover - depends on SDK/network
        status = getattr(exc, "status_code", None)
        error_response = getattr(exc, "response", None)
        response_text = ""
        if error_response is not None:
            try:
                response_text = error_response.text  # type: ignore[attr-defined]
            except Exception:
                try:
                    response_text = json.dumps(error_response.json())  # type: ignore[call-arg]
                except Exception:
                    response_text = repr(error_response)
        details = str(exc)
        if status or response_text:
            details = f"{details} (status={status or 'unknown'}, response={response_text})"
        raise RuntimeError(details) from exc

    content = _extract_text(response)
    usage = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens))
    else:
        prompt_tokens = max(1, len(prompt.split()))
        total_tokens = prompt_tokens
    llm_usage = LLMUsage(tokens_input=prompt_tokens, tokens_output=completion_tokens, total_tokens=total_tokens)
    return content, llm_usage


def _azure_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not all([api_key, endpoint, deployment]):
        raise RuntimeError("Azure OpenAI configuration missing (AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT).")
    if requests is None:  # pragma: no cover - optional dependency
        raise RuntimeError("requests is required for Azure OpenAI provider but is not installed.")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens or 512,
    }
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=CONFIG.timeout)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    llm_usage = LLMUsage(tokens_input=prompt_tokens, tokens_output=completion_tokens, total_tokens=total_tokens)
    return content, llm_usage


def _anthropic_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    if requests is None:  # pragma: no cover - optional dependency
        raise RuntimeError("requests is required for the Anthropic provider but is not installed.")
    model = CONFIG.model or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-5"
    url = "https://api.anthropic.com/v1/messages"
    # claude-opus-4-x and newer extended-thinking models deprecate temperature
    _no_temp_models = ("claude-opus-4",)
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens or 512,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not any(model.startswith(m) for m in _no_temp_models):
        payload["temperature"] = temperature
    if CONFIG.force_json:
        payload["system"] = _FORCE_JSON_SYSTEM
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    response = None
    for attempt in range(4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=CONFIG.timeout)
        except requests.exceptions.ConnectionError:
            if attempt < 3:
                time.sleep(20 * (attempt + 1))
                continue
            raise
        if response.status_code in (429, 500, 529) and attempt < 3:
            time.sleep(15 * (attempt + 1))
            continue
        break
    response.raise_for_status()
    data = response.json()
    content = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = prompt_tokens + completion_tokens
    llm_usage = LLMUsage(tokens_input=prompt_tokens, tokens_output=completion_tokens, total_tokens=total_tokens)
    return content, llm_usage


def _mock_response(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    preview = prompt[:120].replace("\n", " ")
    content = f"[MOCK] {digest} :: {preview}"
    tokens = max(1, len(prompt.split()))
    llm_usage = LLMUsage(tokens_input=tokens, tokens_output=0, total_tokens=tokens)
    return content, llm_usage


def _ollama_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    if requests is None:  # pragma: no cover - optional dependency
        raise RuntimeError("requests is required for the Ollama provider but is not installed.")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = CONFIG.model or os.getenv("OLLAMA_MODEL") or "llama3"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    if max_tokens is not None:
        payload["options"]["num_predict"] = max_tokens
    response = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=CONFIG.timeout)
    response.raise_for_status()
    data = response.json()
    content = data.get("response", "")
    prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
    completion_tokens = int(data.get("eval_count", 0) or 0)
    total_tokens = prompt_tokens + completion_tokens
    llm_usage = LLMUsage(tokens_input=prompt_tokens, tokens_output=completion_tokens, total_tokens=total_tokens)
    return content, llm_usage


def _chat_completions_request(
    prompt: str,
    max_tokens: Optional[int],
    temperature: float,
    api_key: str,
    base_url: str,
    key_name: str,
) -> Tuple[str, LLMUsage]:
    """Generic OpenAI-compatible chat/completions endpoint (Groq, OpenRouter, NIM, etc.)."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai>=1.0 is required for chat-completions-compatible providers.") from exc
    model = (CONFIG.model or "").strip() or "llama-4-scout-17b-16e-instruct"
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=CONFIG.timeout)
    requested_max = max(16, max_tokens if max_tokens is not None else 512)
    last_exc: Optional[Exception] = None
    for _attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=requested_max,
                temperature=temperature,
            )
            last_exc = None
            break
        except Exception as exc:  # pragma: no cover
            status = getattr(exc, "status_code", None)
            # Retry on 429 rate-limit or 503 overload with exponential backoff
            if status in (429, 503) and _attempt < 2:
                time.sleep(10 * (2 ** _attempt))
                last_exc = exc
                continue
            err_resp = getattr(exc, "response", None)
            body = ""
            if err_resp is not None:
                try:
                    body = err_resp.text  # type: ignore[attr-defined]
                except Exception:
                    body = repr(err_resp)
            details = str(exc)
            if status or body:
                details = f"{details} (status={status or 'unknown'}, response={body})"
            raise RuntimeError(details) from exc
    if last_exc is not None:  # pragma: no cover
        raise RuntimeError(str(last_exc)) from last_exc
    raw_content = response.choices[0].message.content or ""
    # Strip chain-of-thought <think>...</think> blocks emitted by reasoning models
    # (qwen3, DeepSeek-R1, etc.) — we only want the final answer.
    # Also handles truncated responses where </think> is missing (max_tokens too low).
    import re as _re
    content = _re.sub(r"<think>.*?</think>", "", raw_content, flags=_re.DOTALL)
    content = _re.sub(r"<think>.*$", "", content, flags=_re.DOTALL)  # unclosed tag
    content = content.strip()
    if not content:
        content = raw_content  # fallback: use full output if nothing left after strip
    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens))
    llm_usage = LLMUsage(tokens_input=prompt_tokens, tokens_output=completion_tokens, total_tokens=total_tokens)
    return content, llm_usage


def _groq_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    # Groq supports /no_think suffix to suppress <think> blocks for qwen3 models.
    effective_prompt = prompt
    if "qwen3" in (CONFIG.model or "").lower():
        effective_prompt = prompt + "\n/no_think"
    return _chat_completions_request(
        effective_prompt, max_tokens, temperature,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        key_name="GROQ_API_KEY",
    )


def _openrouter_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return _chat_completions_request(
        prompt, max_tokens, temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        key_name="OPENROUTER_API_KEY",
    )


def _nim_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("NVIDIA_NIM") or os.getenv("NIM_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_NIM (or NIM_API_KEY) is not set.")
    return _chat_completions_request(
        prompt, max_tokens, temperature,
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        key_name="NVIDIA_NIM",
    )


def _cerebras_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set.")
    return _chat_completions_request(
        prompt, max_tokens, temperature,
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
        key_name="CEREBRAS_API_KEY",
    )


def _sambanova_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        raise RuntimeError("SAMBANOVA_API_KEY is not set.")
    return _chat_completions_request(
        prompt, max_tokens, temperature,
        api_key=api_key,
        base_url="https://api.sambanova.ai/v1",
        key_name="SAMBANOVA_API_KEY",
    )


_GEMINI_THINKING_MODELS = frozenset({"gemini-2.5-pro", "gemini-2.0-pro"})


def _gemini_request(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if requests is None:  # pragma: no cover
        raise RuntimeError("requests is required for the Gemini provider but is not installed.")
    model = CONFIG.model or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    # Thinking models (gemini-2.5-pro) share maxOutputTokens between internal thinking
    # and final output. A small cap (e.g. 768) is fully consumed by thinking, leaving
    # zero tokens for the response. Do not set maxOutputTokens for thinking models.
    is_thinking_model = model in _GEMINI_THINKING_MODELS
    gen_config: Dict[str, Any] = {
        "temperature": temperature,
        **({"maxOutputTokens": max_tokens} if (max_tokens and not is_thinking_model) else {}),
    }
    if CONFIG.force_json:
        gen_config["responseMimeType"] = "application/json"
    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    # Thinking models can take 3-5 minutes per call; use a longer timeout.
    effective_timeout = 360 if is_thinking_model else CONFIG.timeout
    response = None
    for attempt in range(4):
        try:
            response = requests.post(url, json=payload, timeout=effective_timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            if attempt < 3:
                time.sleep(30 * (attempt + 1))
                continue
            raise
        if response.status_code in (429, 500, 503) and attempt < 3:
            time.sleep(10 * (attempt + 1))
            continue
        break
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    content = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in parts)
    usage = data.get("usageMetadata", {})
    tokens_in = int(usage.get("promptTokenCount", 0) or 0)
    tokens_out = int(usage.get("candidatesTokenCount", 0) or 0)
    llm_usage = LLMUsage(tokens_input=tokens_in, tokens_output=tokens_out, total_tokens=tokens_in + tokens_out)
    return content, llm_usage


def _dispatch(prompt: str, max_tokens: Optional[int], temperature: float) -> Tuple[str, LLMUsage]:
    provider = CONFIG.provider.lower()
    if provider == "openai":
        return _openai_request(prompt, max_tokens, temperature)
    if provider == "azure_openai":
        return _azure_request(prompt, max_tokens, temperature)
    if provider == "anthropic":
        return _anthropic_request(prompt, max_tokens, temperature)
    if provider == "gemini":
        return _gemini_request(prompt, max_tokens, temperature)
    if provider == "groq":
        return _groq_request(prompt, max_tokens, temperature)
    if provider in ("openrouter", "open_router"):
        return _openrouter_request(prompt, max_tokens, temperature)
    if provider in ("nim", "nvidia_nim", "nvidia"):
        return _nim_request(prompt, max_tokens, temperature)
    if provider == "cerebras":
        return _cerebras_request(prompt, max_tokens, temperature)
    if provider in ("sambanova", "samba_nova"):
        return _sambanova_request(prompt, max_tokens, temperature)
    if provider == "ollama":
        return _ollama_request(prompt, max_tokens, temperature)
    if provider == "mock":
        return _mock_response(prompt, max_tokens, temperature)
    raise RuntimeError(f"Unsupported LLM provider '{CONFIG.provider}'.")


def call(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    caller: Optional[str] = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> LLMResponse:
    """Call the configured LLM provider (real or stub)."""

    provider = CONFIG.provider.lower()
    selected_model = model or CONFIG.model
    effective_temp = temperature if temperature is not None else CONFIG.temperature
    payload = {
        "provider": provider,
        "model": selected_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": effective_temp,
        "caller": caller or "default",
        "extra": kwargs,
        "force_json": CONFIG.force_json,
    }
    cache_key = _hash_payload(payload)
    if CONFIG.cache_enabled and use_cache:
        cached = _read_cache(cache_key)
        if cached:
            cached.meta = dict(cached.meta or {})
            cached.meta.setdefault("cache_hit", True)
            # Strip reasoning-model think blocks from cached content too
            import re as _re
            stripped = _re.sub(r"<think>.*?</think>", "", cached.content, flags=_re.DOTALL)
            stripped = _re.sub(r"<think>.*$", "", stripped, flags=_re.DOTALL)  # unclosed tag
            stripped = stripped.strip()
            if stripped:
                cached.content = stripped
                return cached
            elif "<think>" in cached.content:
                # Cache contains only a think block with no final answer (truncated response).
                # Evict this entry so a fresh API call can be made.
                _evict_cache(cache_key)
                # Fall through to live API call below
            else:
                return cached
    tokens_estimate = max(1, len(prompt.split()))
    if not _budget_available(tokens_estimate):
        raise RuntimeError("LLM budget exceeded")
    start = time.perf_counter()
    content, usage_stats = _dispatch(prompt, max_tokens, effective_temp)
    elapsed = time.perf_counter() - start
    caller_name = caller or "default"
    _record_usage(usage_stats.total_tokens, elapsed, caller_name, usage_stats.cost_usd)
    meta = {
        "provider": provider,
        "model": selected_model,
        "caller": caller_name,
        "temperature": effective_temp,
        "cache_hit": False,
        **usage_stats.to_meta(),
        "latency_ms": elapsed * 1000.0,
    }
    response = LLMResponse(
        content=content,
        tokens=usage_stats.total_tokens,
        elapsed=elapsed,
        meta=meta,
    )
    if CONFIG.cache_enabled and use_cache:
        _write_cache(cache_key, response)
    return response


def _read_cache(key: str) -> Optional[LLMResponse]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return LLMResponse.from_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(key: str, response: LLMResponse) -> None:
    path = _cache_path(key)
    try:
        path.write_text(response.to_json(), encoding="utf-8")
    except Exception:
        pass


def _evict_cache(key: str) -> None:
    """Remove a cache entry (e.g., contaminated/truncated responses)."""
    path = _cache_path(key)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def validate_connection(test_prompt: str = "Hello!") -> Tuple[bool, str, float, int]:
    """Run a cheap validation call."""

    try:
        response = call(test_prompt, max_tokens=8, temperature=0.01, caller="llm_validate", use_cache=False)
        if response.budget_exceeded:
            return False, "Budget exceeded before validation call.", response.elapsed, response.tokens
        return True, response.content.strip(), response.elapsed, response.tokens
    except Exception as exc:  # pragma: no cover - depends on provider
        return False, str(exc), 0.0, 0


def get_usage() -> Dict[str, Dict[str, float]]:
    """Return aggregate token/time usage."""

    return {
        "totals": {"tokens": TOKENS_USED, "calls": CALLS_USED, "budget": BUDGET_USED},
        "per_caller": stats,
    }


def total_calls() -> int:
    """Return the recorded number of provider invocations."""

    return CALLS_USED


__all__ = ["LLMResponse", "LLMUsage", "call", "get_usage", "set_config", "validate_connection", "total_calls"]
