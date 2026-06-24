"""
src/retrievers/hybrid.py
────────────────────────
V7: Hybrid BM25 + Dense Retrieval with RRF fusion

Why this works for biomedical text:
- BM25 excels at exact medical terminology matching
  (e.g., "BRCA1", "tau phosphorylation", "dopaminergic neurons")
- Dense retrieval excels at semantic matching
  (e.g., "brain cell death" matches "neuronal apoptosis")
- Neither alone is optimal — combining via RRF captures both signals

Reference:
Cormack et al. (2009). Reciprocal Rank Fusion outperforms Condorcet
and individual rank learning methods. SIGIR 2009.
"""

import logging
from typing import Optional
from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from .base import BaseRetriever
from .reranker import CROSS_ENCODER_MODEL

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant


class HybridRetriever(BaseRetriever):
    variant_name = "V7_Hybrid"

    def __init__(
        self,
        vectorstore: Chroma,
        corpus_dir: str = "data/corpus",
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        fetch_multiplier: int = 4,
    ):
        self.vectorstore = vectorstore
        self.fetch_multiplier = fetch_multiplier

        # Load cross-encoder
        logger.info("Loading cross-encoder: %s", cross_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model, max_length=512)

        # Build BM25 index from corpus
        self.corpus_docs = []  # list of Document objects
        self.bm25 = None
        self._build_bm25_index(corpus_dir)

    def _build_bm25_index(self, corpus_dir: str):
        """Load all articles and build BM25 index."""
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists():
            logger.warning("Corpus dir not found: %s", corpus_dir)
            return

        logger.info("Building BM25 index from %s...", corpus_dir)
        tokenized_corpus = []

        for fpath in sorted(corpus_path.glob("*.json")):
            try:
                article = json.loads(fpath.read_text(encoding="utf-8"))
                text = f"{article.get('title', '')} {article.get('abstract', '')}"
                doc = Document(
                    page_content=text,
                    metadata={
                        "pmid"   : article.get("pmid", ""),
                        "title"  : article.get("title", "")[:200],
                        "journal": article.get("journal", "")[:100],
                        "year"   : str(article.get("year", "")),
                        "source" : "pubmed",
                        "section": "ABSTRACT",
                    }
                )
                self.corpus_docs.append(doc)
                tokenized_corpus.append(text.lower().split())
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath.name, exc)

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built: %d documents", len(self.corpus_docs))

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        """BM25 keyword search."""
        if self.bm25 is None or not self.corpus_docs:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        results = []
        for idx in top_indices:
            doc = self.corpus_docs[idx]
            doc.metadata["bm25_score"] = round(float(scores[idx]), 4)
            results.append(doc)
        return results

    def _rrf_fusion(
        self,
        dense_docs: list[Document],
        bm25_docs: list[Document],
        k: int = RRF_K,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion.
        score(doc) = sum(1 / (k + rank_i)) across all lists.
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, 1):
            uid = f"{doc.metadata.get('pmid', '')}_{doc.page_content[:50]}"
            scores[uid] = scores.get(uid, 0) + 1 / (k + rank)
            doc_map[uid] = doc

        for rank, doc in enumerate(bm25_docs, 1):
            uid = f"{doc.metadata.get('pmid', '')}_{doc.page_content[:50]}"
            scores[uid] = scores.get(uid, 0) + 1 / (k + rank)
            if uid not in doc_map:
                doc_map[uid] = doc

        sorted_uids = sorted(scores, key=lambda u: scores[u], reverse=True)
        return [doc_map[uid] for uid in sorted_uids]

    def _retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> tuple[list[Document], dict]:
        fetch_k = k * self.fetch_multiplier

        # Stage 1a: Dense retrieval
        dense_docs = self.vectorstore.similarity_search(
            query=query, k=fetch_k, filter=metadata_filter
        )

        # Stage 1b: BM25 retrieval
        bm25_docs = self._bm25_search(query, k=fetch_k)

        # Stage 2: RRF fusion
        fused_docs = self._rrf_fusion(dense_docs, bm25_docs)

        # Stage 3: Cross-encoder rerank
        if not fused_docs:
            return [], {}

        pairs = [(query, doc.page_content) for doc in fused_docs[:fetch_k]]
        rerank_scores = self.cross_encoder.predict(pairs)

        ranked = sorted(
            zip(rerank_scores, fused_docs[:fetch_k]),
            key=lambda x: x[0],
            reverse=True,
        )
        top_docs = [doc for _, doc in ranked[:k]]
        for score, doc in ranked[:k]:
            doc.metadata["reranker_score"] = round(float(score), 4)

        return top_docs, {
            "dense_count": len(dense_docs),
            "bm25_count" : len(bm25_docs),
            "fused_count": len(fused_docs),
            "top_score"  : round(float(ranked[0][0]), 4) if ranked else 0.0,
        }
