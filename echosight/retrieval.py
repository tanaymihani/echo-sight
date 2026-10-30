"""Hybrid retrieval: BM25 + dense vectors fused with reciprocal rank fusion, then cross-encoder reranking."""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer
import torch
from rank_bm25 import BM25Okapi

from echosight.chunking import Chunk
from echosight.config import DEVICE, RetrievalConfig
from echosight.document import normalize, spoken_title
from echosight.index import VectorIndex, load_chunks

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
""".split())
_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", normalize(text).lower())
    return _stemmer.stemWords([w for w in words if w not in STOPWORDS])


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class BM25Index:
    def __init__(self, chunks: tuple[Chunk, ...], headers: bool = True):
        self.bm25 = BM25Okapi([tokenize(c.indexed_text(headers)) for c in chunks])

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(i, float(scores[i])) for i in best]


@lru_cache(maxsize=2)
def cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=DEVICE)


@dataclass
class Hit:
    chunk: Chunk
    score: float      # reranker probability when a reranker is used, otherwise the fused/raw score

    @property
    def citation(self) -> str:
        return f"{spoken_title(self.chunk.section)}, page {self.chunk.page}"


def make_router(cfg: RetrievalConfig):
    """Build the STAIR-style ToC router named by cfg.toc_router ("provider:model")."""
    from echosight.config import LLMConfig
    from echosight.index import load_sections
    from echosight.llm import LLM
    from echosight.toc import ToCRouter

    provider, model = cfg.toc_router.split(":", 1)
    return ToCRouter(list(load_sections()), LLM(LLMConfig.from_env(provider, model)), k=cfg.toc_k)


class Retriever:
    def __init__(self, cfg: RetrievalConfig = RetrievalConfig(), router=None):
        self.cfg = cfg
        self.chunks = load_chunks(cfg.chunk_size)
        self.dense = VectorIndex(cfg) if cfg.mode in ("dense", "hybrid") else None
        self.bm25 = BM25Index(self.chunks, cfg.section_headers) if cfg.mode in ("bm25", "hybrid") else None
        self.reranker = cross_encoder(cfg.reranker) if cfg.reranker else None
        self.router = router or (make_router(cfg) if cfg.toc_router else None)
        self._pool = ThreadPoolExecutor(max_workers=1) if self.router else None
        self.last_timing: dict[str, float] = {}
        self.last_routed: list[int] = []
        self.last_routing = None

    def _toc_ranking(self, routed: list[int], order: list[int]) -> list[int]:
        """Chunks of each routed section (and its subsections), sections best first, chunks in `order`."""
        position = {chunk: rank for rank, chunk in enumerate(order)}
        ranking: list[int] = []
        for s in routed:
            path = self.router.sections[s].path
            members = [i for i, c in enumerate(self.chunks) if c.path[: len(path)] == path and i not in ranking]
            ranking += sorted(members, key=lambda i: position.get(i, len(order) + i))
        return ranking

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.cfg.top_k
        start = time.perf_counter()
        routing = None
        if self.router:   # the LLM call overlaps with local retrieval
            routing = self._pool.submit(self.router.route, query)
        rankings = []
        if self.dense:
            rankings.append(self.dense.search(query, self.cfg.candidates))
        if self.bm25:
            rankings.append(self.bm25.search(query, self.cfg.candidates))
        orders = [[i for i, _ in r] for r in rankings]
        fused = rankings[0] if len(rankings) == 1 else rrf(orders)
        toc = []
        self.last_routing = routing          # with toc_fusion="none", callers await it only if needed
        if routing and self.cfg.toc_fusion != "none":
            self.last_routed = routing.result()
            toc = self._toc_ranking(self.last_routed, [i for i, _ in fused])
        fusion = self.cfg.toc_fusion if toc else None
        if fusion == "candidates":
            fused = rrf(orders + [toc])
        candidates = fused[: self.cfg.candidates]
        if fusion == "restrict":
            candidates = [(i, 0.0) for i in toc[: self.cfg.candidates]]
        if self.reranker:
            texts = [self.chunks[i].indexed_text(self.cfg.section_headers) for i, _ in candidates]
            scores = self.reranker.predict([(query, t) for t in texts], activation_fn=torch.nn.Sigmoid())
            candidates = sorted(((i, float(s)) for (i, _), s in zip(candidates, scores)),
                                key=lambda pair: pair[1], reverse=True)
        if fusion == "final":
            score = dict(candidates)
            candidates = [(i, score.get(i, 0.0)) for i, _ in rrf([[i for i, _ in candidates], toc])]
        self.last_timing = {"retrieval_s": time.perf_counter() - start}
        return [Hit(self.chunks[i], score) for i, score in candidates[:k]]
