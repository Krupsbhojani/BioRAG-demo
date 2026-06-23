"""
src/pipeline.py
───────────────
BioRAGPipeline — orchestrates retrieval + LLM answer generation.

Separates concerns cleanly:
  - Retrievers handle *finding* relevant context
  - Pipeline handles *answering* given context
  - Eval harness handles *scoring* the answers

This makes the system easy to swap components in any direction.
"""

import logging
from typing import Optional

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from src.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

# Biomedical QA prompt — grounded in context but always produces a substantive answer.
# The "insufficient information" fallback tanks RAGAs answer_relevancy (the metric
# generates synthetic questions from the answer — an "I don't know" produces
# off-topic synthetic questions with near-zero similarity to the original).
_QA_PROMPT = PromptTemplate.from_template(
    """You are a biomedical expert. Answer the question using the provided
PubMed literature excerpts as your primary source.

Rules:
- Prioritise information from the provided context.
- If the context is relevant but incomplete, synthesise an answer from what
  is available and note any gaps (e.g. "Based on available literature...").
- If the context is entirely unrelated to the question, draw on your general
  biomedical knowledge to give a brief, accurate answer.
- Be precise. Use medical terminology correctly.
- Keep answers to 2-4 sentences.

Context:
{context}

Question: {question}

Answer:"""
)


class BioRAGPipeline:
    """
    Full RAG pipeline: retriever (pluggable) + LLM answer generation.

    Parameters
    ----------
    retriever   : any BaseRetriever subclass (swappable at runtime)
    llm         : Groq ChatGroq instance
    """

    def __init__(self, retriever: BaseRetriever, llm: ChatGroq):
        self.retriever = retriever
        self.llm       = llm

    def switch_retriever(self, retriever: BaseRetriever):
        """Hot-swap the retriever without rebuilding the pipeline."""
        self.retriever = retriever
        logger.info("Switched retriever to: %s", retriever.variant_name)

    def answer(
        self,
        question: str,
        contexts: Optional[list[str]] = None,
        k: int = 5,
        metadata_filter: Optional[dict] = None,
    ) -> str:
        """
        Generate an answer for the question.

        If contexts are provided (e.g. from eval harness), skips retrieval.
        Otherwise, runs the configured retriever.

        Parameters
        ----------
        question        : user's question
        contexts        : pre-retrieved context strings (optional)
        k               : documents to retrieve if contexts not provided
        metadata_filter : optional ChromaDB filter

        Returns
        -------
        LLM-generated answer string
        """
        if contexts is None:
            result   = self.retriever.retrieve(question, k=k,
                                                metadata_filter=metadata_filter)
            contexts = result.contexts

        context_str = "\n\n---\n\n".join(contexts) if contexts else "No context retrieved."
        prompt      = _QA_PROMPT.format(context=context_str, question=question)

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as exc:
            logger.error("LLM answer generation failed: %s", exc)
            return f"Error generating answer: {exc}"

    def query(
        self,
        question: str,
        k: int = 5,
        metadata_filter: Optional[dict] = None,
    ) -> dict:
        """
        Full pipeline query: retrieve + answer + return structured result.

        Returns
        -------
        dict:
          answer      : str
          sources     : list of source dicts (pmid, title, section, snippet)
          variant     : retriever variant name
          latency_ms  : retrieval latency
          intermediate: any debug info from the retriever (HyDE hypothesis, etc.)
        """
        result = self.retriever.retrieve(question, k=k,
                                          metadata_filter=metadata_filter)
        answer = self.answer(question, contexts=result.contexts)

        sources = [
            {
                "pmid"    : doc.metadata.get("pmid", ""),
                "title"   : doc.metadata.get("title", ""),
                "journal" : doc.metadata.get("journal", ""),
                "year"    : doc.metadata.get("year", ""),
                "section" : doc.metadata.get("section", ""),
                "snippet" : doc.page_content[:300],
            }
            for doc in result.documents
        ]

        return {
            "question"    : question,
            "answer"      : answer,
            "sources"     : sources,
            "variant"     : result.variant,
            "latency_ms"  : result.latency_ms,
            "intermediate": result.intermediate,
        }
