from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

    def complete_json(self, role: str, prompt: str, deep: bool = False) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            self.last_runtime = "offline"
            return None
        result = self._complete_json_langchain(role, prompt, deep)
        if result is not None:
            return result
        # If langchain already hit the thread-level timeout bound above, the
        # endpoint is suspect — don't retry via the SDK (same endpoint, same
        # likely hang). Only fall back to the SDK for genuine API errors.
        if self.last_runtime == "langchain-timeout":
            return None
        fallback_error = self.last_error
        try:
            from openai import OpenAI  # type: ignore

            kwargs: Dict[str, Any] = {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
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
                timeout=timeout,
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(create_call)
                try:
                    response = future.result(timeout=timeout)
                except FuturesTimeoutError:
                    self.last_error = f"{fallback_error}; fallback=sdk timed out after {timeout}s" if fallback_error else f"sdk timed out after {timeout}s"
                    self.last_runtime = "openai-sdk-timeout"
                    return None
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            self.last_error = None
            self.last_runtime = "openai-sdk-fallback"
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            self.last_error = f"{fallback_error}; fallback={exc}" if fallback_error else str(exc)
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
        model_kwargs: Dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "temperature": 0.2,
            "timeout": timeout,
            "model_kwargs": {"response_format": {"type": "json_object"}},
        }
        if self.base_url:
            model_kwargs["base_url"] = self.base_url

        try:
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
            # Wrap chain.invoke in a hard thread timeout: langchain's `timeout`
            # kwarg is not always honoured by the underlying httpx client
            # (observed hanging indefinitely against some OpenAI-compatible
            # gateways). A thread-level bound guarantees one stuck call can
            # never freeze the whole backtest — on expiry we degrade to the
            # rule-based fallback, same as any other LLM failure.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(chain.invoke, invoke_input)
                try:
                    result = future.result(timeout=timeout)
                except FuturesTimeoutError:
                    self.last_error = f"langchain timed out after {timeout}s"
                    self.last_runtime = "langchain-timeout"
                    return None
            self.last_runtime = "langchain-openai"
            self.last_error = None
            return result if isinstance(result, dict) else None
        except Exception as exc:
            self.last_error = f"langchain completion failed: {exc}"
            self.last_runtime = "langchain-failed"
            return None


def clamp_float(value: object, low: float = -1.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return max(low, min(high, result))
