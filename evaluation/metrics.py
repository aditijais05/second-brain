"""
RAGAS Metrics — wrappers around the four core RAG evaluation metrics.

  Faithfulness       — are all answer claims supported by retrieved chunks?
                       Catches hallucinations. Score 0–1.
  Answer Relevancy   — does the answer actually address the question?
                       Penalises off-topic or incomplete answers.
  Context Precision  — are the top-ranked chunks actually relevant?
                       Measures retrieval signal-to-noise.
  Context Recall     — do the chunks cover the ground-truth answer?
                       Needs a reference answer. Measures coverage.

Faithfulness + Answer Relevancy  →  generation quality  (no reference needed)
Context Precision + Context Recall →  retrieval quality  (recall needs reference)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MetricConfig:
    faithfulness:      bool = True
    answer_relevancy:  bool = True
    context_precision: bool = True
    context_recall:    bool = True   # requires reference answers in eval set

    judge_model: str = "gemini-2.0-flash"
    batch_size:  int = 5             # lower = fewer Gemini rate-limit errors


def build_llm_and_embeddings(api_key: str, model: str = "gemini-2.0-flash"):
    """
    Returns (ragas_llm, ragas_embeddings) backed by Gemini.
    Requires: pip install langchain-google-genai
    """
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
    )
    ragas_emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
    )
    return ragas_llm, ragas_emb


def build_ragas_metrics(config: MetricConfig, llm, embeddings) -> list:
    """Build and wire RAGAS metric objects to the given LLM + embeddings."""
    from ragas.metrics.collections import (
        Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
    )
    metrics = []
    if config.faithfulness:
        m = Faithfulness(); m.llm = llm; metrics.append(m)
    if config.answer_relevancy:
        m = AnswerRelevancy(); m.llm = llm; m.embeddings = embeddings; metrics.append(m)
    if config.context_precision:
        m = ContextPrecision(); m.llm = llm; metrics.append(m)
    if config.context_recall:
        m = ContextRecall(); m.llm = llm; metrics.append(m)
    return metrics