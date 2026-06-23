"""
src/retrievers/base.py
──────────────────────
Abstract base class for all retrieval variants.

Every retriever in this project implements the same interface so the
evaluation harness can swap them transparently:

    retriever = HyDERetriever(vectorstore, llm)
    results   = retriever.retrieve("What causes Alzheimer's disease?", k=5)

This design pattern is called "Strategy" — the algorithm (retrieval method)
is decoupled from the context (evaluation pipeline) that uses it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time

from langchain.schema import Document


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Standardised output from any retriever variant.

    Attributes
    ----------
    query           : original user query
    documents       : ranked list of retrieved Documents
    latency_ms      : wall-clock retrieval time in milliseconds
    variant         : name of the retriever that produced this result
    intermediate    : optional dict for debugging (e.g., HyDE hypothesis)
    """
    query        : str
    documents    : list[Document]
    latency_ms   : float
    variant      : str
    intermediate : dict = field(default_factory=dict)

    @property
    def contexts(self) -> list[str]:
        """Plain-text content of all retrieved documents."""
        return [doc.page_content for doc in self.documents]

    @property
    def top_context(self) -> str:
        """Concatenated top-k contexts as a single string."""
        return "\n\n---\n\n".join(self.contexts)

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(variant={self.variant!r}, "
            f"k={len(self.documents)}, latency={self.latency_ms:.0f}ms)"
        )


# ── Base class ────────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Abstract base for all retrieval strategies.

    Subclasses must implement `_retrieve()`.
    The public `retrieve()` method wraps it with timing and result packaging.
    """

    #: Override in subclass to set the variant name used in eval tables
    variant_name: str = "base"

    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[dict] = None,
    ) -> RetrievalResult:
        """
        Public retrieval entry point.

        Parameters
        ----------
        query           : user's natural language question
        k               : number of documents to return
        metadata_filter : optional ChromaDB metadata filter dict

        Returns
        -------
        RetrievalResult with documents, latency, and any intermediate outputs
        """
        t0   = time.perf_counter()
        docs, intermediate = self._retrieve(query, k, metadata_filter)
        ms   = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query        = query,
            documents    = docs,
            latency_ms   = ms,
            variant      = self.variant_name,
            intermediate = intermediate or {},
        )

    @abstractmethod
    def _retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> tuple[list[Document], dict]:
        """
        Core retrieval logic.

        Returns
        -------
        (documents, intermediate_dict)
        """
        raise NotImplementedError
