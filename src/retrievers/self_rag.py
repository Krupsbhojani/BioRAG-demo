"""
src/retrievers/self_rag.py
──────────────────────────
V6: Self-RAG — Iterative Retrieval with Confidence Critique

The core insight
────────────────
Standard RAG retrieves once and trusts the result. Self-RAG adds a
critique loop: after generating an answer, the LLM evaluates whether
the answer is fully grounded in the retrieved context and directly
addresses the question. If confidence is LOW or MEDIUM, it generates a
targeted follow-up query and retrieves again, accumulating context
across rounds until confidence is HIGH or max_rounds is reached.

This explicitly combats hallucination — the model only accepts an answer
when it can verify grounding, making faithfulness the primary optimisation
target rather than an afterthought.

Reference
─────────
Asai et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection. NeurIPS 2023. https://arxiv.org/abs/2310.11511
"""

import json
import logging
import time
from typing import Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from .base import BaseRetriever

logger = logging.getLogger(__name__)

# ── Critique prompt ────────────────────────────────────────────────────────────

_CRITIQUE_PROMPT = """You are evaluating a biomedical answer for quality.
Question: {question}
Retrieved context: {context}
Generated answer: {answer}
Evaluate the answer on two criteria:

Is the answer fully grounded in the provided context? (no hallucination)
Does the answer directly and completely address the question?

Respond in this exact JSON format:
{{
"confidence": "HIGH" or "MEDIUM" or "LOW",
"missing": "what information is still needed (empty string if HIGH)",
"followup_query": "a specific search query to find missing info (empty if HIGH)"
}}
Only respond with JSON, no other text."""


class SelfRAGRetriever(BaseRetriever):
    """
    Self-RAG retriever: iterative retrieval with LLM-based confidence critique.

    Each round:
    1. Retrieve top-k chunks
    2. Generate a candidate answer from accumulated context
    3. Critique the answer — HIGH / MEDIUM / LOW confidence
    4. If HIGH → accept and return
    5. If MEDIUM/LOW → generate follow-up query, retrieve again, repeat

    Parameters
    ----------
    vectorstore          : ChromaDB Chroma instance
    llm                  : Groq ChatGroq for critique + follow-up generation
    max_rounds           : maximum retrieval iterations (default 3)
    confidence_threshold : accept answer at this confidence or above (default HIGH)
    """

    variant_name = "V6_SelfRAG"

    def __init__(
        self,
        vectorstore: Chroma,
        llm: ChatGroq,
        max_rounds: int = 3,
        confidence_threshold: str = "HIGH",
    ):
        self.vectorstore          = vectorstore
        self.llm                  = llm
        self.max_rounds           = max_rounds
        self.confidence_threshold = confidence_threshold

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _retrieve_docs(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> list[Document]:
        """Single similarity search — returns k documents."""
        return self.vectorstore.similarity_search(
            query  = query,
            k      = k,
            filter = metadata_filter,
        )

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate a candidate answer from the accumulated context."""
        prompt = (
            f"You are a biomedical expert. Answer the question using only the "
            f"provided context. Be concise and factual.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as exc:
            logger.warning("Answer generation failed: %s", exc)
            return ""

    def _critique(self, question: str, context: str, answer: str) -> dict:
        """
        Run the critique prompt. Returns dict with keys:
        confidence, missing, followup_query.

        Falls back to LOW confidence + original question on any failure.
        """
        prompt = _CRITIQUE_PROMPT.format(
            question=question,
            context=context[:3000],   # truncate to avoid token overflow
            answer=answer,
        )
        try:
            response = self.llm.invoke(prompt)
            raw = response.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            # Normalise confidence to uppercase
            parsed["confidence"] = parsed.get("confidence", "LOW").upper()
            if parsed["confidence"] not in ("HIGH", "MEDIUM", "LOW"):
                parsed["confidence"] = "LOW"

            return parsed

        except Exception as exc:
            logger.warning("Critique parsing failed: %s — treating as LOW", exc)
            return {
                "confidence"    : "LOW",
                "missing"       : "parse error",
                "followup_query": question,
            }

    @staticmethod
    def _dedup_docs(docs: list[Document]) -> list[Document]:
        """
        Deduplicate documents by (pmid, section) metadata key.
        Preserves insertion order — earlier rounds take precedence.
        """
        seen  = set()
        deduped = []
        for doc in docs:
            key = (
                doc.metadata.get("pmid", ""),
                doc.metadata.get("section", doc.page_content[:80]),
            )
            if key not in seen:
                seen.add(key)
                deduped.append(doc)
        return deduped

    # ── Core retrieval logic ───────────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict],
    ) -> tuple[list[Document], dict]:
        """
        Iterative Self-RAG loop.

        Returns
        -------
        (accumulated_docs, intermediate_dict)
        """
        all_docs           : list[Document] = []
        confidence_history : list[str]      = []
        followup_queries   : list[str]      = []
        best_answer        : str            = ""
        current_query      : str            = query

        for round_num in range(1, self.max_rounds + 1):
            logger.debug("Self-RAG round %d | query: %s", round_num, current_query[:80])

            # Step 1: Retrieve
            round_docs = self._retrieve_docs(current_query, k, metadata_filter)
            all_docs.extend(round_docs)
            all_docs = self._dedup_docs(all_docs)

            # Step 2: Generate candidate answer from ALL accumulated context
            context = "\n\n---\n\n".join(d.page_content for d in all_docs)
            answer  = self._generate_answer(query, context)
            if answer:
                best_answer = answer

            # Step 3: Critique
            try:
                critique = self._critique(query, context, answer)
            except Exception as exc:
                # Groq rate limit or network error — stop here, return best so far
                logger.warning("Critique call failed (round %d): %s — stopping early", round_num, exc)
                confidence_history.append("ERROR")
                break

            confidence = critique.get("confidence", "LOW")
            confidence_history.append(confidence)
            logger.debug("Round %d confidence: %s", round_num, confidence)

            # Step 4: Accept or iterate
            if confidence == self.confidence_threshold:
                logger.debug("Self-RAG accepted at round %d (HIGH confidence)", round_num)
                break

            if round_num < self.max_rounds:
                followup = critique.get("followup_query", "").strip()
                if not followup:
                    followup = query   # fallback to original
                followup_queries.append(followup)
                current_query = followup

            # Small sleep to respect Groq RPM between rounds
            time.sleep(1.0)

        intermediate = {
            "rounds_completed"  : len(confidence_history),
            "confidence_history": confidence_history,
            "followup_queries"  : followup_queries,
            "total_unique_docs" : len(all_docs),
            "best_answer"       : best_answer[:300] if best_answer else "",
        }

        return all_docs, intermediate
