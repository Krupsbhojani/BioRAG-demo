"""
src/retrievers/multiquery.py
─────────────────────────────
V5: Multi-Query Decomposition + Reranking

Motivation
──────────
Complex biomedical questions often contain multiple sub-intents. Example:

  "What are the side effects, efficacy, and contraindications of
   metformin in elderly patients with type 2 diabetes and renal impairment?"

A single query embedding averages across all sub-intents, diluting the
signal for each. A retriever that fetches good docs for "metformin side effects"
may miss docs about "renal impairment dosing", and vice versa.

Solution: decompose → retrieve independently → deduplicate → rerank

Algorithm
─────────
1. LLM decomposes the question into 3 focused sub-questions
2. Retrieve k candidates for EACH sub-question
3. Deduplicate by PMID (same article, different chunk = keep best)
4. Cross-encoder rerank the unified pool against the original question
5. Return top-k

This is especially effective for:
  - Multi-faceted clinical questions
  - Comparative questions ("X vs Y")
  - Questions with multiple entities (drug + disease + population)

Reference
─────────
Ma et al. (2023). Query Rewriting for Retrieval-Augmented Large Language Models.
https://arxiv.org/abs/2305.14283
"""

import logging
import json
import re
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from .base import BaseRetriever
from .reranker import CROSS_ENCODER_MODEL, FETCH_MULTIPLIER

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """You are a biomedical information retrieval expert.
Break down the following complex question into exactly 3 focused sub-questions.
Each sub-question should target a distinct aspect of the original question.
Return ONLY a JSON array of 3 strings. No explanation, no markdown.

Original question: {question}

JSON array of 3 sub-questions:"""


class MultiQueryRetriever(BaseRetriever):
    """
    Multi-query decomposition with cross-encoder reranking.

    Parameters
    ----------
    vectorstore         : ChromaDB Chroma instance
    llm                 : Groq ChatGroq for query decomposition
    cross_encoder_model : cross-encoder for final reranking
    n_subqueries        : number of sub-questions to generate (default 3)
    """

    variant_name = "V5_MultiQuery"

    def __init__(
        self,
        vectorstore: Chroma,
        llm: ChatGroq,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        n_subqueries: int = 3,
    ):
        self.vectorstore  = vectorstore
        self.llm          = llm
        self.n_subqueries = n_subqueries

        logger.info("Loading cross-encoder: %s", cross_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model, max_length=512)

    def _decompose_query(self, question: str) -> list[str]:
        """
        Use LLM to decompose the question into sub-questions.
        Falls back to [question] on any parsing error.
        """
        try:
            prompt   = _DECOMPOSE_PROMPT.format(question=question)
            response = self.llm.invoke(prompt)
            raw      = response.content.strip()

            # Strip accidental markdown fences
            raw = re.sub(r"```json|```", "", raw).strip()
            subqueries = json.loads(raw)

            if isinstance(subqueries, list) and len(subqueries) > 0:
                logger.debug("Decomposed into %d sub-queries.", len(subqueries))
                return [str(q).strip() for q in subqueries[: self.n_subqueries]]

        except Exception as exc:
            logger.warning("Query decomposition failed (%s). Using original query.", exc)

        return [question]

    def _retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> tuple[list[Document], dict]:
        """
        1. Decompose query into sub-questions.
        2. Retrieve candidates for each sub-question.
        3. Deduplicate by PMID.
        4. Cross-encoder rerank pool against original query.
        5. Return top-k.
        """
        subqueries = self._decompose_query(query)

        # Per-subquery fetch = k × FETCH_MULTIPLIER → total pool = n_subqueries × k × 4
        # For n=3, k=5: 3 × 5 × 4 = 60 candidates = 12k (matches paper Table 1)
        fetch_per_query = k * FETCH_MULTIPLIER

        # Retrieve for each sub-question, collect all candidates
        all_candidates: list[Document] = []
        seen_ids: set[str] = set()

        for subq in subqueries:
            candidates = self.vectorstore.similarity_search(
                query  = subq,
                k      = fetch_per_query,
                filter = metadata_filter,
            )
            for doc in candidates:
                # Deduplicate by PMID + section — different questions may
                # retrieve the same chunk; we keep the first occurrence
                # Use a tuple key: pmid + section + first 100 chars of content.
                # 100 chars (vs 50) reduces false collisions between different
                # chunks from the same pmid/section split by the chunker.
                uid = (doc.metadata.get("pmid", ""),
                       doc.metadata.get("section", ""),
                       doc.page_content[:100])
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    all_candidates.append(doc)

        if not all_candidates:
            return [], {"subqueries": subqueries}

        # Cross-encoder rerank the unified pool against the ORIGINAL query
        pairs  = [(query, doc.page_content) for doc in all_candidates]
        scores = self.cross_encoder.predict(pairs)

        ranked   = sorted(zip(scores, all_candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:k]]

        for score, doc in ranked[:k]:
            doc.metadata["reranker_score"] = round(float(score), 4)

        return top_docs, {
            "subqueries"      : subqueries,
            "total_candidates": len(all_candidates),
            "top_score"       : round(float(ranked[0][0]), 4) if ranked else 0.0,
        }
