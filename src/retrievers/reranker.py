"""
src/retrievers/reranker.py
──────────────────────────
V3: Naive Retrieval + Cross-Encoder Reranking

Two-stage retrieval architecture
─────────────────────────────────
Stage 1 — Bi-encoder (ChromaDB):
  Encodes query and documents independently → fast approximate search.
  Fetch a large candidate pool (k * fetch_multiplier).

Stage 2 — Cross-encoder (ms-marco-MiniLM):
  Sees the query and each document TOGETHER → accurate relevance scoring.
  Reranks the candidate pool and returns the top-k.

Why this matters
────────────────
Bi-encoders are trained for speed: they compress documents into fixed
vectors independently of any query. This is efficient but loses
interaction signals between query tokens and document tokens.

Cross-encoders attend to both query and document simultaneously
(full transformer attention), capturing subtle relevance signals that
bi-encoders miss. The tradeoff: too slow for full-corpus search,
but fast enough to rerank 15-50 candidates in <200ms on CPU.

This two-stage architecture is standard in production search systems
(Google, Bing, OpenAI's retrieval API) and is well-documented in the
IR literature (Nogueira & Cho, 2020).

Reference
─────────
Nogueira & Cho (2020). Passage Re-ranking with BERT.
https://arxiv.org/abs/1901.04085

Reimers & Gurevych (2019). Sentence-BERT. EMNLP 2019.
https://arxiv.org/abs/1908.10084
"""

import logging
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

from .base import BaseRetriever

logger = logging.getLogger(__name__)

# Strong cross-encoder fine-tuned on MS MARCO passage ranking.
# Generalises well to biomedical text; ~12M params, fast on CPU.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# How many candidates to fetch before reranking (trade-off: recall vs latency)
FETCH_MULTIPLIER = 4


class RerankerRetriever(BaseRetriever):
    """
    Two-stage retriever: bi-encoder candidate fetch → cross-encoder rerank.

    Parameters
    ----------
    vectorstore       : ChromaDB Chroma instance
    cross_encoder_model : HuggingFace model identifier for cross-encoder
    fetch_multiplier  : candidate pool = k × fetch_multiplier
    """

    variant_name = "V3_Reranker"

    def __init__(
        self,
        vectorstore: Chroma,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        fetch_multiplier: int = FETCH_MULTIPLIER,
    ):
        self.vectorstore      = vectorstore
        self.fetch_multiplier = fetch_multiplier

        logger.info("Loading cross-encoder: %s", cross_encoder_model)
        # Downloads ~65 MB on first call, cached to HuggingFace cache dir
        self.cross_encoder = CrossEncoder(
            cross_encoder_model,
            max_length=512,
        )
        logger.info("Cross-encoder ready.")

    def _retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> tuple[list[Document], dict]:
        """
        Stage 1: Fetch k * multiplier candidates via dense retrieval.
        Stage 2: Rerank with cross-encoder, return top-k.
        """
        fetch_k = k * self.fetch_multiplier

        # Stage 1 — retrieve large candidate pool
        candidates = self.vectorstore.similarity_search(
            query  = query,
            k      = fetch_k,
            filter = metadata_filter,
        )

        if not candidates:
            return [], {}

        # Stage 2 — cross-encoder scoring
        # Input format: list of (query, document_text) pairs
        pairs  = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Sort by score descending, take top-k
        ranked = sorted(
            zip(scores, candidates),
            key    = lambda x: x[0],
            reverse=True,
        )
        top_docs = [doc for _, doc in ranked[:k]]

        # Attach reranker score to metadata for transparency.
        # Copy metadata dict — never mutate the original Document in-place,
        # as the same object may be reused across queries from the cache.
        for score, doc in ranked[:k]:
            doc.metadata = {**doc.metadata, "reranker_score": round(float(score), 4)}

        intermediate = {
            "candidate_scores": [round(float(s), 3) for s in scores[:10]],
            "top_score"       : round(float(ranked[0][0]), 4) if ranked else 0.0,
        }

        return top_docs, intermediate
