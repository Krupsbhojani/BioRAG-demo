"""
src/retrievers/hyde.py
──────────────────────
V2: HyDE — Hypothetical Document Embeddings

The core insight
────────────────
In biomedical RAG, a user query like:
    "What are the mechanisms of tau protein aggregation in Alzheimer's?"

...does NOT look like a PubMed abstract, which might read:
    "Tau hyperphosphorylation leads to neurofibrillary tangle formation
     via microtubule dissociation and subsequent oligomer nucleation..."

Embedding the query and the abstract in the same space creates a
semantic gap — the query lives in "question-land", the docs live in
"answer-land". Cosine similarity between them is imprecise.

HyDE solves this by:
1. Using the LLM to generate a *hypothetical* PubMed-style answer
2. Embedding THAT (which lives in "answer-land")
3. Finding real documents similar to the fake answer

This narrows the query-document domain gap and measurably improves
recall on biomedical benchmarks.

Reference
─────────
Gao et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels.
ACL 2023. https://arxiv.org/abs/2212.10496
"""

import logging
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from .base import BaseRetriever

logger = logging.getLogger(__name__)

# Prompt engineered to produce PubMed-style abstract language
_HYDE_PROMPT = """You are a biomedical researcher writing a PubMed abstract.
Write a concise, factual passage (3-5 sentences) that directly answers
the following question in the style of a scientific abstract.
Use precise medical terminology. Do not add disclaimers or say "I".

Question: {question}

Hypothetical abstract passage:"""


class HyDERetriever(BaseRetriever):
    """
    HyDE retriever: generates a hypothetical answer, embeds it,
    retrieves real documents by similarity to that embedding.

    Parameters
    ----------
    vectorstore : ChromaDB Chroma instance
    llm         : Groq ChatGroq instance for hypothesis generation
    """

    variant_name = "V2_HyDE"

    def __init__(self, vectorstore: Chroma, llm: ChatGroq):
        self.vectorstore = vectorstore
        self.llm         = llm

    def _generate_hypothesis(self, question: str) -> str:
        """
        Generate a hypothetical PubMed-style passage for the question.

        Returns the hypothesis text, or the original question on failure.
        """
        try:
            prompt   = _HYDE_PROMPT.format(question=question)
            response = self.llm.invoke(prompt)
            hypothesis = response.content.strip()
            logger.debug("HyDE hypothesis: %s", hypothesis[:120])
            return hypothesis
        except Exception as exc:
            logger.warning("HyDE hypothesis generation failed: %s. Falling back to query.", exc)
            return question

    def _retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> tuple[list[Document], dict]:
        """
        Generate hypothesis → embed → retrieve real documents.
        Returns intermediate dict containing the hypothesis for inspection.
        """
        hypothesis = self._generate_hypothesis(query)

        # Use the hypothesis as the search string — its embedding is
        # closer to real abstract embeddings than the raw question
        docs = self.vectorstore.similarity_search(
            query  = hypothesis,
            k      = k,
            filter = metadata_filter,
        )

        return docs, {"hypothesis": hypothesis}
