"""Streaming chat client for any OpenAI-compatible endpoint (Groq, Ollama, Gemini, ...)."""

import time
from collections.abc import Callable
from dataclasses import dataclass

from openai import BadRequestError, OpenAI

from echosight.config import LLMConfig


@dataclass
class Completion:
    text: str
    ttft_s: float | None          # time to first content token
    total_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig.from_env()
        if not self.cfg.base_url or not self.cfg.model:
            raise ValueError(f"Unknown LLM provider {self.cfg.provider!r}: set ECHO_LLM_BASE_URL and ECHO_LLM_MODEL")
        if self.cfg.provider in ("groq", "gemini") and not self.cfg.api_key:
            raise ValueError(f"No API key for {self.cfg.provider}: add it to .env (see README)")
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key or "unused",
                             max_retries=6, timeout=120)
        self._unsupported: set[str] = set()     # optional parameters this endpoint rejected

    @property
    def name(self) -> str:
        return f"{self.cfg.provider}/{self.cfg.model}"

    def _reasoning_params(self) -> dict:
        """Keep thinking models fast: low effort for gpt-oss, thinking off for local models."""
        model = self.cfg.model.lower()
        if "gpt-oss" in model:
            params = {"reasoning_effort": "low"}
            if self.cfg.provider == "groq":
                params["include_reasoning"] = False
            return params
        if self.cfg.provider == "ollama" or "qwen3" in model:
            return {"reasoning_effort": "none"}
        return {}

    def _create(self, **request):
        """Send the request, dropping optional parameters that the endpoint rejects by name."""
        while True:
            extra = {k: v for k, v in self._reasoning_params().items() if k not in self._unsupported}
            if "response_format" in self._unsupported:
                request.pop("response_format", None)
            try:
                return self.client.chat.completions.create(extra_body=extra, **request)
            except BadRequestError as error:
                rejected = {k for k in [*extra, "response_format"] if k in str(error)} - self._unsupported
                if not rejected:
                    raise
                self._unsupported |= rejected

    def chat(self, messages: list[dict], on_text: Callable[[str], None] | None = None,
             max_tokens: int | None = None, temperature: float | None = None,
             json_mode: bool = False) -> Completion:
        start = time.perf_counter()
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        stream = self._create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        parts, ttft, usage = [], None, None
        for event in stream:
            if event.usage:
                usage = event.usage
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - start
                parts.append(delta)
                if on_text:
                    on_text(delta)
        return Completion(
            text="".join(parts).strip(),
            ttft_s=ttft,
            total_s=time.perf_counter() - start,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
