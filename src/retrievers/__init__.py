# src/retrievers/__init__.py
from .naive         import NaiveRetriever
from .hyde          import HyDERetriever
from .reranker      import RerankerRetriever
from .hyde_reranker import HyDERerankerRetriever
from .multiquery    import MultiQueryRetriever
from .self_rag      import SelfRAGRetriever
from .hybrid        import HybridRetriever
from .base          import BaseRetriever, RetrievalResult

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "NaiveRetriever",
    "HyDERetriever",
    "RerankerRetriever",
    "HyDERerankerRetriever",
    "MultiQueryRetriever",
    "SelfRAGRetriever",
    "HybridRetriever",
]

# Ordered list for the ablation study
# V6_SelfRAG excluded — failed due to Groq rate limiting, not algorithmic failure
ALL_VARIANTS = [
    "V1_Naive",
    "V2_HyDE",
    "V3_Reranker",
    "V4_HyDE_Reranker",
    "V5_MultiQuery",
    "V7_Hybrid",
]
