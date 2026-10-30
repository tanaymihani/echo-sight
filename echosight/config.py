"""Settings for retrieval and the LLM backend. Every LLM setting can be overridden from .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# Quiet model loading in a CLI meant for screen readers.
for _name, _value in (("HF_HUB_DISABLE_PROGRESS_BARS", "1"), ("HF_HUB_VERBOSITY", "error"),
                      ("TRANSFORMERS_VERBOSITY", "error"), ("TOKENIZERS_PARALLELISM", "false")):
    os.environ.setdefault(_name, _value)

PDF_PATH = Path(os.getenv("ECHO_PDF", ROOT / "data" / "code-student-conduct.pdf"))
INDEX_DIR = Path(os.getenv("ECHO_INDEX_DIR", ROOT / "index"))
# CPU by default: the models are small, and Apple-GPU (mps) memory runs out on 8 GB machines.
DEVICE = os.getenv("ECHO_DEVICE", "cpu")


@dataclass(frozen=True)
class RetrievalConfig:
    embed_model: str = "BAAI/bge-base-en-v1.5"
    chunk_size: int = 800              # max characters per chunk; chunks never cross a heading
    section_headers: bool = True       # prefix each chunk with its heading path before indexing
    mode: str = "hybrid"               # "dense", "bm25" or "hybrid" (reciprocal rank fusion)
    reranker: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 20               # pool size taken from each retriever before reranking
    top_k: int = 4                     # passages given to the LLM
    # Optional: refuse without calling the LLM below this rerank score. Off by default because
    # rerank scores of paraphrased questions overlap with out-of-scope ones (see README).
    min_score: float | None = None
    # STAIR-style table-of-contents routing (echosight/toc.py), as "provider:model", e.g.
    # "groq:openai/gpt-oss-20b". toc_fusion: "none" uses routing only for the scope gate below,
    # "candidates" adds it to the pre-rerank fusion, "final" fuses it with the reranked list,
    # "restrict" reranks only chunks of routed sections. Fusion did not help on the dev split.
    toc_router: str | None = None
    toc_k: int = 3
    toc_fusion: str = "none"
    # Scope gate: refuse without calling the LLM when the router finds no section AND the top rerank
    # score is below this. The router is only awaited when the score is low, so confident queries
    # pay no extra latency. Chosen on the dev split (0 of 79 answerable questions gated).
    toc_gate_score: float | None = 0.05

    @property
    def name(self) -> str:
        parts = [self.embed_model.split("/")[-1], f"{self.chunk_size}c", self.mode]
        if not self.section_headers:
            parts.append("no-headers")
        if self.reranker:
            parts.append("+" + self.reranker.split("/")[-1])
        if self.toc_router:
            parts.append(f"+STAIR({self.toc_router.split('/')[-1]}, {self.toc_fusion})")
        return " ".join(parts)


# OpenAI-compatible endpoints. Switch with ECHO_LLM_PROVIDER, or point ECHO_LLM_BASE_URL anywhere else.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "ollama": {
        "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1",
        "key_env": None,
        "model": "qwen3.5:4b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
    },
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "groq"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 800              # includes hidden reasoning tokens for reasoning models

    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None) -> "LLMConfig":
        provider = provider or os.getenv("ECHO_LLM_PROVIDER", "groq")
        preset = PROVIDERS.get(provider, {})
        key_env = preset.get("key_env")
        return cls(
            provider=provider,
            model=model or os.getenv("ECHO_LLM_MODEL") or preset.get("model"),
            base_url=os.getenv("ECHO_LLM_BASE_URL") or preset.get("base_url"),
            api_key=os.getenv("ECHO_LLM_API_KEY") or (os.getenv(key_env) if key_env else None),
        )
