"""
src/retrievers/naive.py
───────────────────────
V1: Naive Dense Retrieval — the baseline every other variant is measured against.

How it works
────────────
1. Embed the user query using BioMedBERT
2. Compute cosine similarity against all stored chunk embeddings
3. Return top-k by similarity score

This is the standard "vanilla RAG" approach from Lewis et al. (2020).
It works well when the query naturally resembles the target document text,
but degrades on complex biomedical questions where the query phrasing
diverges significantly from abstract language.

Reference
─────────
Lewis et al. (2020). Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. NeurIPS 2020.
https://arxiv.org/abs/2005.11401
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from .base import BaseRetriever


class NaiveRetriever(BaseRetriever):
    """
    Baseline dense retrieval — cosine similarity in embedding space.

    Parameters
    ----------
    vectorstore : ChromaDB Chroma instance (already indexed)
    """

    variant_name = "V1_Naive"

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def _retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> tuple[list[Document], dict]:
        """Standard similarity search with optional metadata filter."""
        docs = self.vectorstore.similarity_search(
            query  = query,
            k      = k,
            filter = metadata_filter,
        )
        return docs, {}
