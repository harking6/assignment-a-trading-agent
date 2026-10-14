from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.assignment import assignment_todo


class LLMClient:
    """Small OpenAI-compatible gateway for TradingAgents-style roles.

    The project remains fully offline-runnable. In `auto` mode this client only
    activates when OPENAI_API_KEY is present; in `llm` mode failures are surfaced
    in traces but deterministic agents still provide a fallback decision.
    """

    def __init__(
        self,
        mode: str = "auto",
        quick_model: str | None = None,
        deep_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.mode = (mode or "auto").lower()
        # Treat the OpenAI placeholder defaults ("gpt-4o-mini"/"gpt-4o") as
        # "not configured": they are just the schema's built-in default and
        # would shadow a real model set via OPENAI_MODEL/OPENAI_DEEP_MODEL in
        # .env (e.g. deepseek-v4-flash). Any other explicit value is honored.
        self.quick_model = self._resolve_model(quick_model, "OPENAI_MODEL", "gpt-4o-mini")
        self.deep_model = self._resolve_model(deep_model, "OPENAI_DEEP_MODEL", self.quick_model)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self.last_error: str | None = None
        self.last_runtime: str = "offline"
        self.last_call: Dict[str, Any] = {}
        self.call_history: list[Dict[str, Any]] = []

    @staticmethod
    def _resolve_model(value: str | None, env_name: str, fallback: str) -> str:
        if value and value not in ("gpt-4o-mini", "gpt-4o"):
            return value
        env_val = os.getenv(env_name)
        if env_val:
            return env_val
        return value or fallback

    @property
    def enabled(self) -> bool:
        if self.mode == "offline":
            return False
        return bool(os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def _http_client(timeout: float):
        """Build a synchronous client without inheriting host network settings.

        Some Windows Python/OpenSSL environments terminate the TLS handshake
        when httpx inherits the process environment, even when no explicit
        proxy variable is visible.  The DashScope OpenAI-compatible endpoint
        is reachable with normal certificate verification when ``trust_env``
        is disabled.  A deployment that deliberately relies on an HTTP proxy
        can opt back in with ``OPENAI_TRUST_ENV=true``.
        """
        import httpx

        trust_env = os.getenv("OPENAI_TRUST_ENV", "false").strip().lower() in {"1", "true", "yes", "on"}
        request_timeout = httpx.Timeout(
            timeout,
            connect=min(10.0, timeout),
            read=timeout,
            write=timeout,
            pool=min(10.0, timeout),
        )
        return httpx.Client(timeout=request_timeout, trust_env=trust_env)

    def reset_observability(self) -> None:
        """Start a new per-rebalance trace without retaining earlier calls."""
        self.last_call = {}
        self.call_history = []

    def _record_call(self, role: str, model: str, deep: bool, started_at: float, success: bool) -> None:
        """Store bounded, secret-free call metadata for the UI and backtest trace."""
        detail = {
            "role": role,
            "model": model,
            "deep": deep,
            "runtime": self.last_runtime,
            "success": success,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "error": self.last_error,
        }
        self.last_call = detail
        self.call_history.append(detail)
        self.call_history = self.call_history[-20:]

    def _extra_body(self) -> Dict[str, Any] | None:
        """Apply provider-specific structured-output settings when requested.

        Qwen3.7 is a hybrid thinking model. For short JSON control messages,
        allowing the default thinking mode can consume the entire output budget
        before any parseable JSON is emitted. DashScope supports disabling it
        through the OpenAI-compatible ``extra_body`` payload. Other providers
        keep their existing behavior unless the setting is explicitly supplied.
        """
        raw_value = os.getenv("OPENAI_ENABLE_THINKING")
        is_dashscope = bool(self.base_url and "dashscope.aliyuncs.com" in self.base_url)
        if raw_value is None and not is_dashscope:
            return None
        enabled = raw_value.strip().lower() in {"1", "true", "yes", "on"} if raw_value is not None else False
        return {"enable_thinking": enabled}

    def complete_json(self, role: str, prompt: str, deep: bool = False) -> Optional[Dict[str, Any]]:
        model_name = self.deep_model if deep else self.quick_model
        started_at = time.perf_counter()
        if not self.enabled:
            self.last_runtime = "offline"
            self.last_error = None
            self._record_call(role, model_name, deep, started_at, success=False)
            return None
        result = self._complete_json_langchain(role, prompt, deep)
        if result is not None:
            self._record_call(role, model_name, deep, started_at, success=True)
            return result
        # A timeout against the primary client is not retried through the same
        # endpoint: doing so doubles cost and makes a rebalance appear stuck.
        if self.last_runtime in {"langchain-timeout", "langchain-truncated"}:
            self._record_call(role, model_name, deep, started_at, success=False)
            return None
        fallback_error = self.last_error
        try:
            from openai import OpenAI  # type: ignore

            timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
            max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
            kwargs: Dict[str, Any] = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_retries": 0,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            with self._http_client(timeout) as http_client:
                kwargs["http_client"] = http_client
                client = OpenAI(**kwargs)
                create_call = lambda: client.chat.completions.create(
                    model=self.deep_model if deep else self.quick_model,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a financial trading agent role. Return strict JSON only. "
                                "Never claim real execution; decisions are for a paper trading simulator."
                            ),
                        },
                        {"role": "user", "content": f"Role: {role}\n\n{prompt}"},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                    extra_body=self._extra_body(),
                    timeout=timeout,
                )
                response = create_call()
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            self.last_error = None
            self.last_runtime = "openai-sdk-fallback"
            self._record_call(role, model_name, deep, started_at, success=True)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            self.last_error = f"{fallback_error}; fallback={exc}" if fallback_error else str(exc)
            self.last_runtime = "openai-sdk-timeout" if _is_timeout_error(exc) else "openai-sdk-failed"
            self._record_call(role, model_name, deep, started_at, success=False)
            return None

    def _complete_json_langchain(self, role: str, prompt: str, deep: bool = False) -> Optional[Dict[str, Any]]:
        """LC-01: LCEL prompt-model-parser chain.

        Compose ``ChatPromptTemplate | ChatOpenAI | JsonOutputParser`` and
        invoke it. Return a dictionary on success. On failure update
        ``last_error`` / ``last_runtime`` and return ``None`` so the OpenAI SDK
        fallback in ``complete_json`` can run.
        """
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            self.last_error = f"langchain_openai import failed: {exc}"
            self.last_runtime = "langchain-import-failed"
            return None

        api_key = os.getenv("OPENAI_API_KEY")
        model_name = self.deep_model if deep else self.quick_model
        timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
        model_kwargs: Dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "temperature": 0.2,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "max_retries": 0,
            "model_kwargs": {"response_format": {"type": "json_object"}},
        }
        if self.base_url:
            model_kwargs["base_url"] = self.base_url
        extra_body = self._extra_body()
        if extra_body:
            model_kwargs["extra_body"] = extra_body

        try:
            with self._http_client(timeout) as http_client:
                model_kwargs["http_client"] = http_client
                model = ChatOpenAI(**model_kwargs)
                parser = JsonOutputParser()
                template = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a financial trading agent role in a paper-trading simulator. "
                            "Return strict JSON only. Never claim real execution; all decisions are simulated.",
                        ),
                        (
                            "human",
                            "Role: {role}\n\n{prompt}\n\n{format_instructions}",
                        ),
                    ]
                )
                chain = template | model | parser
                invoke_input = {
                    "role": role,
                    "prompt": prompt,
                    "format_instructions": parser.get_format_instructions(),
                }
                # httpx owns the timeout, so the request is cancelled at the
                # transport boundary rather than leaving an unkillable worker
                # thread behind after a Python-level timeout.
                result = chain.invoke(invoke_input)
            self.last_runtime = "langchain-openai"
            self.last_error = None
            return result if isinstance(result, dict) else None
        except Exception as exc:
            self.last_error = f"langchain completion failed: {exc}"
            if _is_timeout_error(exc):
                self.last_runtime = "langchain-timeout"
            elif _is_length_limit_error(exc):
                self.last_runtime = "langchain-truncated"
            else:
                self.last_runtime = "langchain-failed"
            return None


def _is_timeout_error(exc: Exception) -> bool:
    """Recognise httpx/OpenAI timeout wrappers without importing either eagerly."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timed out" in text or "readtimeout" in text


def _is_length_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "length limit" in text or "finish_reason" in text and "length" in text


def clamp_float(value: object, low: float = -1.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return max(low, min(high, result))
