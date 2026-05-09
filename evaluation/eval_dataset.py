"""
Evaluation Dataset — curated Q&A pairs for measuring RAG quality.

Each EvalQuestion has:
  question  — what a user would ask
  reference — the ideal ground-truth answer (written by you)
  tags      — topic labels for grouping results

Replace DEFAULT_EVAL_QUESTIONS with questions about YOUR documents.
The ones below work if your knowledge base contains content about RAG /
retrieval systems (e.g. the notes you ingested during weeks 1–5).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalQuestion:
    question:  str
    reference: Optional[str] = None   # ideal answer — needed for ContextRecall
    tags:      list[str]     = field(default_factory=list)


DEFAULT_EVAL_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        question="What is Retrieval Augmented Generation and why is it useful?",
        reference=(
            "Retrieval Augmented Generation (RAG) is a technique that combines "
            "information retrieval with language model generation. It retrieves "
            "relevant documents from a knowledge base and passes them as context "
            "to an LLM, reducing hallucinations and enabling up-to-date answers "
            "without retraining the model."
        ),
        tags=["rag", "overview"],
    ),
    EvalQuestion(
        question="How does BM25 differ from vector search?",
        reference=(
            "BM25 is a keyword-based ranking algorithm that scores documents by "
            "term frequency and inverse document frequency. Vector search uses "
            "dense embeddings to find semantically similar documents even without "
            "exact keyword matches. BM25 excels at exact terms; vector search "
            "excels at paraphrased or conceptual queries."
        ),
        tags=["retrieval", "bm25"],
    ),
    EvalQuestion(
        question="What is Reciprocal Rank Fusion and how does it combine search results?",
        reference=(
            "Reciprocal Rank Fusion (RRF) merges multiple ranked lists by assigning "
            "each document a score of 1/(k + rank) across all lists, where k=60. "
            "Documents appearing in multiple lists accumulate higher scores. It is "
            "robust because it uses only rank positions, not raw scores."
        ),
        tags=["retrieval", "rrf"],
    ),
    EvalQuestion(
        question="What embedding models does the Second Brain pipeline support?",
        reference=(
            "The pipeline supports two backends: local sentence-transformers "
            "(all-MiniLM-L6-v2, 384 dimensions, free) and OpenAI "
            "(text-embedding-3-small, 1536 dimensions)."
        ),
        tags=["embeddings"],
    ),
    EvalQuestion(
        question="How does the chunker handle documents that are too large for one chunk?",
        reference=(
            "The SemanticChunker splits at paragraph and heading boundaries first. "
            "Oversized paragraphs are split at sentence boundaries. Single sentences "
            "that are still too long get a hard token-level split. Overlap tokens "
            "from the previous chunk are prepended to preserve cross-boundary context."
        ),
        tags=["chunking"],
    ),
    EvalQuestion(
        question="How are citations extracted from the generated answer?",
        reference=(
            "The generator prompts the LLM to include [source_N] markers inline. "
            "After generation, a regex extracts all source numbers and maps each "
            "back to the corresponding retrieved chunk, producing CitedSource "
            "objects with chunk_id, title, source_uri, and text."
        ),
        tags=["generation", "citations"],
    ),
]