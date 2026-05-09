"""
Week 7 — RAGAS Evaluation Script

Run:
    python week7_eval.py --docs notes/ --key AIza...
    python week7_eval.py --docs notes/ --questions-only   # skip RAGAS scoring
    python week7_eval.py --load evaluation/results/run_001.json  # view saved report

Requirements:
    pip install ragas langchain-google-genai
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run RAGAS evaluation on Second Brain")
    p.add_argument("--docs",    nargs="+", help="Paths to files or folders to ingest")
    p.add_argument("--key",     default=os.environ.get("GEMINI_API_KEY", ""),
                   help="Gemini API key (or set GEMINI_API_KEY env var)")
    p.add_argument("--top-k",  type=int, default=5,  help="Chunks retrieved per question")
    p.add_argument("--output",  default="evaluation/results/run_001.json",
                   help="Where to save the JSON report")
    p.add_argument("--load",    help="Load + display a saved report instead of running")
    p.add_argument("--questions-only", action="store_true",
                   help="Print answers without running RAGAS scoring (no API needed)")
    p.add_argument("--no-recall", action="store_true",
                   help="Skip ContextRecall (useful if you have no reference answers)")
    p.add_argument("--embedding", default="local", choices=["local", "openai"],
                   help="Embedding backend")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load-only mode ────────────────────────────────────────────────
    if args.load:
        from evaluation.evaluator import EvalReport
        report = EvalReport.load(args.load)
        report.print_summary()
        return

    # ── Validate inputs ───────────────────────────────────────────────
    if not args.docs:
        print("Error: provide --docs <path> [path ...]")
        sys.exit(1)

    if not args.questions_only and not args.key:
        print("Error: --key required for RAGAS scoring (or set GEMINI_API_KEY)")
        sys.exit(1)

    # ── Ingest ────────────────────────────────────────────────────────
    from ingestion.pipeline import IngestionPipeline

    sources = []
    for d in args.docs:
        p = Path(d.rstrip("/\\"))   # strip trailing slashes (Windows issue)
        if p.is_dir():
            found = list(p.rglob("*.md")) + list(p.rglob("*.pdf")) + list(p.rglob("*.txt"))
            if not found:
                print(f"Warning: no .md/.pdf/.txt files found in '{p}'")
            sources.extend(found)
        elif p.is_file():
            sources.append(p)
        else:
            print(f"Warning: '{p}' is not a file or directory — skipping")

    if not sources:
        print("Error: no files found to ingest. Check your --docs paths.")
        sys.exit(1)

    print(f"Found {len(sources)} file(s) to ingest:")
    docs = IngestionPipeline().run(sources)
    if not docs:
        print("No documents ingested. Check your --docs paths.")
        sys.exit(1)

    # ── Build pipeline ────────────────────────────────────────────────
    from week5_pipeline import Week5Pipeline

    # In questions-only mode we still need a key placeholder to satisfy
    # CitationGenerator.__init__ — the generator will be called normally
    # with the real key when pipeline.ask() runs.
    effective_key = args.key or "placeholder-not-used-in-questions-only-mode"

    pipeline = Week5Pipeline(
        embedding_backend=args.embedding,
        gemini_api_key=effective_key,
        top_k=args.top_k,
    )
    pipeline.index(docs)

    # ── Questions-only mode ───────────────────────────────────────────
    from evaluation.eval_dataset import DEFAULT_EVAL_QUESTIONS

    if args.questions_only:
        print("\n── Answers (no RAGAS scoring) ──────────────────────\n")
        for i, eq in enumerate(DEFAULT_EVAL_QUESTIONS, 1):
            print(f"Q{i}: {eq.question}")
            try:
                answer = pipeline.ask(eq.question, top_k=args.top_k)
                print(f"A:  {answer.answer}")
                print(f"    Sources: {[s.title for s in answer.cited_sources]}")
            except Exception as e:
                print(f"    ⚠ Error: {e}")
            print()
        return

    # ── Full RAGAS evaluation ─────────────────────────────────────────
    from evaluation.evaluator import RAGEvaluator
    from evaluation.metrics import MetricConfig

    config = MetricConfig(
        faithfulness=True,
        answer_relevancy=True,
        context_precision=True,
        context_recall=not args.no_recall,
    )

    evaluator = RAGEvaluator(pipeline, gemini_api_key=args.key, config=config)
    report    = evaluator.run(DEFAULT_EVAL_QUESTIONS)
    report.print_summary()
    report.save(args.output)


if __name__ == "__main__":
    main()