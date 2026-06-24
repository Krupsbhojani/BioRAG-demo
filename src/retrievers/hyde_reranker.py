"""
src/retrievers/hyde_reranker.py
────────────────────────────────
V4: HyDE + Cross-Encoder Reranking (combined)

This is the highest-performing variant in our ablation study.

It stacks both improvements:
  1. HyDE closes the query-document semantic gap (precision ↑)
  2. Cross-encoder reranking refines the candidate pool (precision ↑↑)

The two techniques are complementary:
  - HyDE improves *what* gets into the candidate pool
  - Reranking improves *the ordering* of the candidate pool

Note on latency
───────────────
This variant is the slowest (~2s vs ~0.9s for naive) because it makes
one extra LLM call (hypothesis generation) before retrieval. In production
you'd cache hypotheses for repeated query patterns. For research evaluation,
the accuracy gains justify the cost.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from .base import BaseRetriever
from .hyde import HyDERetriever, _HYDE_PROMPT
from .reranker import CROSS_ENCODER_MODEL, FETCH_MULTIPLIER

logger = logging.getLogger(__name__)


class HyDERerankerRetriever(BaseRetriever):
    """
    Combined HyDE + cross-encoder reranker.

    Pipeline:
      query → LLM hypothesis → embed hypothesis → large candidate pool
            → cross-encoder rerank → top-k documents

    Parameters
    ----------
    vectorstore         : ChromaDB Chroma instance
    llm                 : Groq ChatGroq for HyDE hypothesis generation
    cross_encoder_model : HuggingFace cross-encoder model identifier
    fetch_multiplier    : candidate pool = k × fetch_multiplier
    """

    variant_name = "V4_HyDE_Reranker"

    def __init__(
        self,
        vectorstore: Chroma,
        llm: ChatGroq,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        fetch_multiplier: int = FETCH_MULTIPLIER,
    ):
        self.vectorstore      = vectorstore
        self.llm              = llm
        self.fetch_multiplier = fetch_multiplier

        # Reuse HyDE hypothesis generator
        self._hyde = HyDERetriever(vectorstore, llm)

        logger.info("Loading cross-encoder: %s", cross_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model, max_length=512)

    def _retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> tuple[list[Document], dict]:
        """
        1. Generate HyDE hypothesis
        2. Fetch large candidate pool using hypothesis embedding
        3. Rerank candidates with cross-encoder against ORIGINAL query
           (cross-encoder uses the real query — more accurate than hypothesis)
        """
        # Step 1: Generate hypothesis
        hypothesis = self._hyde._generate_hypothesis(query)

        # Step 2: Fetch candidate pool via hypothesis similarity
        fetch_k    = k * self.fetch_multiplier
        candidates = self.vectorstore.similarity_search(
            query  = hypothesis,
            k      = fetch_k,
            filter = metadata_filter,
        )

        if not candidates:
            return [], {"hypothesis": hypothesis}

        # Step 3: Rerank using the ORIGINAL query (not hypothesis)
        # This is intentional: cross-encoders are better at judging
        # true relevance with the actual question than with a hypothesis
        pairs  = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        ranked   = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:k]]

        for score, doc in ranked[:k]:
            doc.metadata["reranker_score"] = round(float(score), 4)

        return top_docs, {
            "hypothesis"      : hypothesis,
            "top_score"       : round(float(ranked[0][0]), 4) if ranked else 0.0,
            "candidate_scores": [round(float(s), 3) for s in scores[:10]],
        }
