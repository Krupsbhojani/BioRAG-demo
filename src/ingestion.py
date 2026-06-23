"""
src/ingestion.py
────────────────
Converts raw PubMed article dicts into ChromaDB vector store entries.

Key design decisions vs. naive RAG
────────────────────────────────────
1. BioMedBERT embeddings  — domain-adaptive model pre-trained on 29M PubMed
   abstracts (Gu et al., 2021). Significantly outperforms general-purpose
   sentence-transformers on biomedical retrieval benchmarks.

2. Structured chunking — abstracts are split on section boundaries
   (BACKGROUND, METHODS, RESULTS, CONCLUSIONS) before character chunking.
   This preserves semantic coherence better than blind character splits.

3. Rich metadata — each chunk stores PMID, title, journal, year, MeSH terms,
   and section label. Enables metadata-filtered retrieval (e.g. post-2020 only).

4. Persistent collection with collision detection — re-running ingestion
   skips already-indexed PMIDs, making incremental updates cheap.

Reference
─────────
Gu et al. (2021). Domain-Specific Language Model Pretraining for
Biomedical Natural Language Processing. ACM CHIL 2021.
https://arxiv.org/abs/2007.15779
"""

import logging
import re
from pathlib import Path
from typing import Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# BioMedBERT — pre-trained on PubMed abstracts + full text (Gu et al., 2021)
# Paper §3.1 refers to this as "BiomedNLP-BiomedBERT-base-uncased" (shortened name);
# the actual HuggingFace ID is the abstract-fulltext variant below.
BIOMEDBERT_MODEL  = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Chunking — abstracts are shorter than papers; tighter chunks improve precision
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64
COLLECTION_NAME   = "biorag_pubmed"
CHROMA_PERSIST    = "./chroma_db"

# Abstract section headers to split on first (then fall back to character split)
SECTION_HEADERS   = re.compile(
    r"\b(BACKGROUND|OBJECTIVE|METHODS|RESULTS|CONCLUSIONS?|INTRODUCTION"
    r"|PURPOSE|SIGNIFICANCE|DISCUSSION|ABSTRACT)\s*[:.]?\s*",
    re.IGNORECASE,
)


# ── Chunking helpers ──────────────────────────────────────────────────────────

def split_abstract_into_sections(abstract: str) -> list[tuple[str, str]]:
    """
    Split a structured abstract into (section_label, text) tuples.

    If the abstract has no section headers, returns a single
    ("ABSTRACT", full_text) tuple.

    Examples
    --------
    "BACKGROUND: X. METHODS: Y. RESULTS: Z."
    → [("BACKGROUND", "X."), ("METHODS", "Y."), ("RESULTS", "Z.")]
    """
    parts   = SECTION_HEADERS.split(abstract)
    headers = SECTION_HEADERS.findall(abstract)

    if not headers:
        return [("ABSTRACT", abstract.strip())]

    sections = []
    # parts[0] is any text before the first header (usually empty)
    for i, header in enumerate(headers):
        text = parts[i * 2 + 2].strip() if (i * 2 + 2) < len(parts) else ""
        if text:
            sections.append((header.upper().rstrip(":. "), text))

    return sections if sections else [("ABSTRACT", abstract.strip())]


def article_to_documents(article: dict) -> list[Document]:
    """
    Convert a single PubMed article dict into a list of LangChain Documents.

    Each Document contains one abstract section (or the full abstract if
    unstructured), with complete metadata for filtered retrieval.

    Parameters
    ----------
    article : dict with keys: pmid, title, abstract, journal, year,
              authors, mesh_terms

    Returns
    -------
    List of Documents (typically 1-5 per article)
    """
    abstract = article.get("abstract", "").strip()
    if not abstract:
        return []

    base_metadata = {
        "pmid"      : article.get("pmid", ""),
        "title"     : article.get("title", "")[:200],   # ChromaDB metadata limit
        "journal"   : article.get("journal", "")[:100],
        "year"      : str(article.get("year", "")),
        "authors"   : ", ".join(article.get("authors", [])[:3]),  # first 3
        "mesh_terms": "|".join(article.get("mesh_terms", [])[:10]),
        "source"    : "pubmed",
    }

    sections = split_abstract_into_sections(abstract)
    documents = []

    for section_label, section_text in sections:
        # Prefix the chunk with title + section for richer embedding signal
        enriched_text = (
            f"Title: {article.get('title', '')}\n"
            f"Section: {section_label}\n\n"
            f"{section_text}"
        )
        documents.append(
            Document(
                page_content=enriched_text,
                metadata={**base_metadata, "section": section_label},
            )
        )

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Further split Documents that exceed CHUNK_SIZE characters.
    Most abstract sections will be under 512 chars and won't be split.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ── Embedding model ───────────────────────────────────────────────────────────

def load_embeddings(model_name: str = BIOMEDBERT_MODEL) -> HuggingFaceEmbeddings:
    """
    Load BioMedBERT embeddings. Downloads ~440 MB on first call,
    cached to ~/.cache/huggingface thereafter.
    """
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name    = model_name,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True, "batch_size": 32},
    )


# ── Vector store ──────────────────────────────────────────────────────────────

class BioRAGVectorStore:
    """
    Manages the ChromaDB collection for biomedical RAG.

    Supports incremental indexing (skip already-indexed PMIDs),
    metadata-filtered retrieval, and clean resets.
    """

    def __init__(
        self,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
        persist_dir: str = CHROMA_PERSIST,
        collection_name: str = COLLECTION_NAME,
    ):
        self.embeddings      = embeddings or load_embeddings()
        self.persist_dir     = persist_dir
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None

    def _get_or_create_store(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name  = self.collection_name,
                embedding_function = self.embeddings,
                persist_directory  = self.persist_dir,
            )
        return self._vectorstore

    def get_indexed_pmids(self) -> set[str]:
        """Return the set of PMIDs already in the collection."""
        store = self._get_or_create_store()
        try:
            results = store.get(include=["metadatas"])
            return {
                m.get("pmid", "") for m in results["metadatas"] if m.get("pmid")
            }
        except Exception:
            return set()

    def index_articles(
        self,
        articles: list[dict],
        batch_size: int = 500,
        skip_existing: bool = True,
    ) -> dict:
        """
        Embed and index a list of PubMed article dicts.

        Parameters
        ----------
        articles       : from fetch_pubmed.load_corpus()
        batch_size     : documents per ChromaDB upsert call
        skip_existing  : if True, skip PMIDs already indexed

        Returns
        -------
        Summary dict: {total_articles, indexed, skipped, total_chunks}
        """
        store = self._get_or_create_store()

        existing_pmids = self.get_indexed_pmids() if skip_existing else set()
        logger.info("Already indexed: %d PMIDs", len(existing_pmids))

        # Build document list, skipping existing PMIDs
        all_docs: list[Document] = []
        skipped = 0

        for article in articles:
            pmid = article.get("pmid", "")
            if pmid in existing_pmids:
                skipped += 1
                continue
            docs = article_to_documents(article)
            all_docs.extend(chunk_documents(docs))

        if not all_docs:
            logger.info("Nothing new to index.")
            return {
                "total_articles": len(articles),
                "indexed"       : 0,
                "skipped"       : skipped,
                "total_chunks"  : 0,
            }

        logger.info("Indexing %d chunks from %d new articles…",
                    len(all_docs), len(articles) - skipped)

        # Batch upsert to avoid memory spikes
        indexed_chunks = 0
        for i in tqdm(
            range(0, len(all_docs), batch_size),
            desc="Indexing batches",
            unit="batch",
        ):
            batch = all_docs[i : i + batch_size]
            store.add_documents(batch)
            indexed_chunks += len(batch)

        store.persist()
        logger.info("Indexing complete. %d chunks persisted.", indexed_chunks)

        return {
            "total_articles": len(articles),
            "indexed"       : len(articles) - skipped,
            "skipped"       : skipped,
            "total_chunks"  : indexed_chunks,
        }

    def as_retriever(self, search_type: str = "mmr", k: int = 5, **kwargs):
        """Return a LangChain retriever from the vector store."""
        store = self._get_or_create_store()
        search_kwargs = {"k": k, **kwargs}
        if search_type == "mmr":
            search_kwargs.update({"fetch_k": k * 3, "lambda_mult": 0.7})
        return store.as_retriever(
            search_type   = search_type,
            search_kwargs = search_kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Direct similarity search with optional metadata filter.

        Example filter: {"year": "2023"}  or  {"journal": "Lancet"}
        """
        store = self._get_or_create_store()
        return store.similarity_search(query, k=k, filter=filter)

    def collection_stats(self) -> dict:
        """Return basic stats about the indexed collection."""
        store  = self._get_or_create_store()
        result = store.get(include=["metadatas"])
        metas  = result.get("metadatas", [])

        from collections import Counter
        years    = Counter(m.get("year", "?")  for m in metas)
        journals = Counter(m.get("journal", "?") for m in metas)

        return {
            "total_chunks"  : len(metas),
            "unique_pmids"  : len({m.get("pmid") for m in metas if m.get("pmid")}),
            "year_dist"     : dict(years.most_common(5)),
            "top_journals"  : dict(journals.most_common(5)),
        }

    def reset(self):
        """Delete the entire collection (irreversible)."""
        store = self._get_or_create_store()
        store.delete_collection()
        self._vectorstore = None
        logger.info("Collection deleted.")


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from data.fetch_pubmed import load_corpus

    parser = argparse.ArgumentParser(description="Index PubMed corpus into ChromaDB.")
    parser.add_argument("--corpus_dir", default="data/corpus")
    parser.add_argument("--persist_dir", default="./chroma_db")
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()

    articles = load_corpus(args.corpus_dir)

    embeddings = load_embeddings()
    vs         = BioRAGVectorStore(embeddings=embeddings, persist_dir=args.persist_dir)
    summary    = vs.index_articles(articles, batch_size=args.batch_size)

    print("\n✅ Indexing complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    stats = vs.collection_stats()
    print("\n📊 Collection stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")