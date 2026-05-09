"""
RAG Evaluator — runs RAGAS metrics against your Week5Pipeline.

Quick start:
    from ingestion.pipeline import IngestionPipeline
    from week5_pipeline import Week5Pipeline
    from evaluation.evaluator import RAGEvaluator
    from evaluation.eval_dataset import DEFAULT_EVAL_QUESTIONS

    docs     = IngestionPipeline().run(["notes/rag.md", "papers/attention.pdf"])
    pipeline = Week5Pipeline(embedding_backend="local")
    pipeline.index(docs)

    evaluator = RAGEvaluator(pipeline)          # reads GEMINI_API_KEY from env
    report    = evaluator.run(DEFAULT_EVAL_QUESTIONS)
    report.print_summary()
    report.save("evaluation/results/run_001.json")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from week5_pipeline import Week5Pipeline

from evaluation.eval_dataset import EvalQuestion
from evaluation.metrics import MetricConfig, build_ragas_metrics, build_llm_and_embeddings


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class SingleResult:
    question:         str
    answer:           str
    retrieved_chunks: list[str]
    reference:        Optional[str]

    faithfulness:     Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision:Optional[float] = None
    context_recall:   Optional[float] = None

    latency_ms: float = 0.0
    error:      Optional[str] = None

    def scores(self) -> dict[str, float]:
        return {k: v for k, v in {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }.items() if v is not None}


@dataclass
class EvalReport:
    results:    list[SingleResult]
    run_at:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    num_docs:   int = 0
    num_chunks: int = 0
    model:      str = "gemini-2.0-flash"

    def mean(self, metric: str) -> Optional[float]:
        vals = [getattr(r, metric) for r in self.results if getattr(r, metric) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def summary(self) -> dict:
        return {
            "faithfulness":      self.mean("faithfulness"),
            "answer_relevancy":  self.mean("answer_relevancy"),
            "context_precision": self.mean("context_precision"),
            "context_recall":    self.mean("context_recall"),
            "num_questions":     len(self.results),
            "num_errors":        sum(1 for r in self.results if r.error),
            "avg_latency_ms":    round(
                sum(r.latency_ms for r in self.results) / max(len(self.results), 1), 0
            ),
        }

    def print_summary(self) -> None:
        s = self.summary()
        w = 54
        print(f"\n{'RAGAS Evaluation Report':^{w}}")
        print("─" * w)
        print(f"  Questions : {s['num_questions']}   Errors: {s['num_errors']}   "
              f"Avg latency: {s['avg_latency_ms']:.0f}ms")
        print("─" * w)
        rows = [
            ("Faithfulness",      s["faithfulness"],      "claims grounded in context"),
            ("Answer Relevancy",  s["answer_relevancy"],  "answer addresses question"),
            ("Context Precision", s["context_precision"], "retrieved chunks relevant"),
            ("Context Recall",    s["context_recall"],    "ground truth covered"),
        ]
        for name, score, meaning in rows:
            if score is not None:
                filled = "█" * int(score * 20)
                empty  = "░" * (20 - int(score * 20))
                print(f"  {name:<22} {score:.3f}  {filled}{empty}  {meaning}")
            else:
                print(f"  {name:<22}   n/a")
        print("─" * w)

        print("\nPer-question breakdown:")
        print(f"  {'#':>2}  {'Question':<43} {'Fth':>5} {'Rel':>5} {'Pre':>5} {'Rec':>5}")
        print("  " + "─" * 67)
        for i, r in enumerate(self.results, 1):
            q  = (r.question[:41] + "..") if len(r.question) > 43 else r.question
            fa = f"{r.faithfulness:.2f}"      if r.faithfulness      is not None else "  -"
            re = f"{r.answer_relevancy:.2f}"  if r.answer_relevancy  is not None else "  -"
            pr = f"{r.context_precision:.2f}" if r.context_precision is not None else "  -"
            rc = f"{r.context_recall:.2f}"    if r.context_recall    is not None else "  -"
            flag = " ⚠" if r.error else ""
            print(f"  {i:>2}  {q:<43} {fa:>5} {re:>5} {pr:>5} {rc:>5}{flag}")
        print()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_at": self.run_at, "num_docs": self.num_docs,
            "num_chunks": self.num_chunks, "model": self.model,
            "summary": self.summary(),
            "results": [
                {
                    "question": r.question, "answer": r.answer,
                    "reference": r.reference,
                    "retrieved_contexts": r.retrieved_chunks,
                    "scores": r.scores(),
                    "latency_ms": r.latency_ms, "error": r.error,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Report saved → {path}")

    @classmethod
    def load(cls, path: str) -> "EvalReport":
        with open(path) as f:
            data = json.load(f)
        results = []
        for r in data["results"]:
            s = r.get("scores", {})
            results.append(SingleResult(
                question=r["question"], answer=r["answer"],
                retrieved_chunks=r.get("retrieved_contexts", []),
                reference=r.get("reference"),
                faithfulness=s.get("faithfulness"),
                answer_relevancy=s.get("answer_relevancy"),
                context_precision=s.get("context_precision"),
                context_recall=s.get("context_recall"),
                latency_ms=r.get("latency_ms", 0), error=r.get("error"),
            ))
        return cls(results=results, run_at=data.get("run_at", ""),
                   num_docs=data.get("num_docs", 0),
                   num_chunks=data.get("num_chunks", 0),
                   model=data.get("model", ""))


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Runs RAGAS evaluation over a built Week5Pipeline.

    Args:
        pipeline:       An indexed Week5Pipeline instance.
        gemini_api_key: Falls back to GEMINI_API_KEY env var.
        config:         MetricConfig — which metrics to run.
        top_k:          Chunks to retrieve per question.
    """

    def __init__(
        self,
        pipeline:       "Week5Pipeline",
        gemini_api_key: Optional[str] = None,
        config:         Optional[MetricConfig] = None,
        top_k:          Optional[int] = None,
    ):
        self.pipeline = pipeline
        self.api_key  = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self.config   = config or MetricConfig()
        self.top_k    = top_k or pipeline.top_k
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var.")

    def run(self, questions: list[EvalQuestion], verbose: bool = True) -> EvalReport:
        if self.pipeline.retriever is None:
            raise RuntimeError("Pipeline not indexed. Call pipeline.index(docs) first.")

        if verbose:
            enabled = [k for k, v in {
                "faithfulness": self.config.faithfulness,
                "answer_relevancy": self.config.answer_relevancy,
                "context_precision": self.config.context_precision,
                "context_recall": self.config.context_recall,
            }.items() if v]
            print(f"\nRAGAS evaluation — {len(questions)} questions, "
                  f"metrics: {', '.join(enabled)}\n")

        raw     = self._collect_answers(questions, verbose)
        scored  = self._score_with_ragas(raw, verbose)

        return EvalReport(
            results=scored,
            num_docs=len(set(c.doc_id for c in self.pipeline._chunks)),
            num_chunks=self.pipeline.vector_store.count(),
            model=self.config.judge_model,
        )

    # ── Step 1: retrieve + generate ────────────────────────────────────

    def _collect_answers(
        self, questions: list[EvalQuestion], verbose: bool
    ) -> list[SingleResult]:
        results = []
        for i, eq in enumerate(questions, 1):
            if verbose:
                print(f"  [{i}/{len(questions)}] {eq.question[:65]}")
            t0 = time.time()
            try:
                cited   = self.pipeline.ask(eq.question, top_k=self.top_k)
                latency = (time.time() - t0) * 1000
                results.append(SingleResult(
                    question=eq.question,
                    answer=cited.answer,
                    retrieved_chunks=[c.get("text", "") for c in cited.all_sources],
                    reference=eq.reference,
                    latency_ms=latency,
                ))
                if verbose:
                    print(f"         {len(cited.all_sources)} chunks · "
                          f"{len(cited.cited_sources)} citations · {latency:.0f}ms")
            except Exception as e:
                results.append(SingleResult(
                    question=eq.question, answer="",
                    retrieved_chunks=[], reference=eq.reference, error=str(e),
                ))
                if verbose:
                    print(f"         ⚠ {e}")
        return results

    # ── Step 2: RAGAS scoring ──────────────────────────────────────────

    def _score_with_ragas(
        self, results: list[SingleResult], verbose: bool
    ) -> list[SingleResult]:
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas import evaluate

        if verbose:
            print(f"\nScoring with RAGAS (judge: {self.config.judge_model})…")

        ragas_llm, ragas_emb = build_llm_and_embeddings(
            self.api_key, self.config.judge_model
        )
        metrics = build_ragas_metrics(self.config, ragas_llm, ragas_emb)
        if not metrics:
            return results

        valid = [r for r in results if not r.error and r.answer]
        if not valid:
            print("  No valid results to score.")
            return results

        samples = []
        for r in valid:
            kw = dict(
                user_input=r.question,
                response=r.answer,
                retrieved_contexts=r.retrieved_chunks,
            )
            if r.reference and self.config.context_recall:
                kw["reference"] = r.reference
            samples.append(SingleTurnSample(**kw))

        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=metrics,
                raise_exceptions=False,
                show_progress=verbose,
                batch_size=self.config.batch_size,
            )
            df = result.to_pandas()
        except Exception as e:
            print(f"  ⚠ RAGAS error: {e}")
            return results

        col_map = {
            "faithfulness":      "faithfulness",
            "answer_relevancy":  "answer_relevancy",
            "context_precision": "context_precision",
            "context_recall":    "context_recall",
        }
        for i, r in enumerate(valid):
            if i >= len(df):
                break
            row = df.iloc[i]
            for attr, col in col_map.items():
                if col in df.columns:
                    v = row[col]
                    if v is not None and str(v) != "nan":
                        setattr(r, attr, float(v))

        if verbose:
            print("  Done.\n")
        return results