"""
Tests for Week 7: RAGAS evaluation framework.

These tests mock both the pipeline and the Gemini/RAGAS APIs so no
API key or indexed documents are needed.

Run: pytest tests/test_week7.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from evaluation.eval_dataset import EvalQuestion, DEFAULT_EVAL_QUESTIONS
from evaluation.evaluator import RAGEvaluator, EvalReport, SingleResult
from evaluation.metrics import MetricConfig


# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    EvalQuestion(
        question="What is RAG?",
        reference="RAG combines retrieval with language model generation.",
        tags=["rag"],
    ),
    EvalQuestion(
        question="How does BM25 work?",
        reference="BM25 ranks documents by term frequency and inverse document frequency.",
        tags=["retrieval"],
    ),
]


def make_mock_pipeline(answer: str = "RAG is great [source_1].") -> MagicMock:
    """Return a mock Week5Pipeline with a pre-built retriever."""
    from generation.generator import CitedAnswer, CitedSource

    cited_source = CitedSource(
        source_num=1, chunk_id="c1", doc_id="d1",
        title="RAG Notes", source_uri="/rag.md",
        text="RAG combines retrieval with language model generation.",
    )
    mock_answer = CitedAnswer(
        answer=answer,
        cited_sources=[cited_source],
        all_sources=[{
            "chunk_id": "c1", "doc_id": "d1",
            "title": "RAG Notes", "source_uri": "/rag.md",
            "text": "RAG combines retrieval with language model generation.",
            "rrf_score": 0.032, "rrf_rank": 1,
        }],
        query="What is RAG?",
    )

    mock_chunk = MagicMock()
    mock_chunk.doc_id = "d1"

    pipeline = MagicMock()
    pipeline.retriever = MagicMock()          # not None = indexed
    pipeline.top_k = 5
    pipeline._chunks = [mock_chunk]
    pipeline.vector_store.count.return_value = 10
    pipeline.ask.return_value = mock_answer
    return pipeline


def make_mock_evaluator(pipeline=None, **config_kwargs) -> RAGEvaluator:
    """Create a RAGEvaluator with a fake API key."""
    p = pipeline or make_mock_pipeline()
    return RAGEvaluator(
        pipeline=p,
        gemini_api_key="fake-key",
        config=MetricConfig(**config_kwargs),
    )


# ── EvalQuestion tests ─────────────────────────────────────────────────────

def test_eval_question_has_required_fields():
    eq = EvalQuestion(question="What is RAG?", reference="It is great.", tags=["rag"])
    assert eq.question == "What is RAG?"
    assert eq.reference == "It is great."
    assert "rag" in eq.tags


def test_eval_question_reference_is_optional():
    eq = EvalQuestion(question="What is RAG?")
    assert eq.reference is None
    assert eq.tags == []


def test_default_eval_questions_not_empty():
    assert len(DEFAULT_EVAL_QUESTIONS) >= 4


def test_default_eval_questions_have_references():
    for eq in DEFAULT_EVAL_QUESTIONS:
        assert eq.reference is not None, f"Missing reference for: {eq.question}"


def test_default_eval_questions_have_tags():
    for eq in DEFAULT_EVAL_QUESTIONS:
        assert len(eq.tags) >= 1


# ── MetricConfig tests ─────────────────────────────────────────────────────

def test_metric_config_defaults():
    cfg = MetricConfig()
    assert cfg.faithfulness is True
    assert cfg.answer_relevancy is True
    assert cfg.context_precision is True
    assert cfg.context_recall is True


def test_metric_config_can_disable_recall():
    cfg = MetricConfig(context_recall=False)
    assert cfg.context_recall is False


# ── RAGEvaluator init tests ────────────────────────────────────────────────

def test_evaluator_requires_api_key():
    pipeline = make_mock_pipeline()
    with patch.dict("os.environ", {}, clear=True):
        # Remove GEMINI_API_KEY if present
        import os; os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(ValueError, match="API key"):
            RAGEvaluator(pipeline=pipeline, gemini_api_key="")


def test_evaluator_raises_if_not_indexed():
    pipeline = make_mock_pipeline()
    pipeline.retriever = None               # simulate un-indexed pipeline
    evaluator = make_mock_evaluator(pipeline=pipeline)
    with pytest.raises(RuntimeError, match="not indexed"):
        evaluator.run(SAMPLE_QUESTIONS)


# ── _collect_answers tests ─────────────────────────────────────────────────

def test_collect_answers_returns_one_result_per_question():
    ev = make_mock_evaluator()
    results = ev._collect_answers(SAMPLE_QUESTIONS, verbose=False)
    assert len(results) == len(SAMPLE_QUESTIONS)


def test_collect_answers_preserves_question_text():
    ev = make_mock_evaluator()
    results = ev._collect_answers(SAMPLE_QUESTIONS, verbose=False)
    assert results[0].question == SAMPLE_QUESTIONS[0].question


def test_collect_answers_populates_retrieved_chunks():
    ev = make_mock_evaluator()
    results = ev._collect_answers(SAMPLE_QUESTIONS, verbose=False)
    assert len(results[0].retrieved_chunks) >= 1
    assert isinstance(results[0].retrieved_chunks[0], str)


def test_collect_answers_preserves_reference():
    ev = make_mock_evaluator()
    results = ev._collect_answers(SAMPLE_QUESTIONS, verbose=False)
    assert results[0].reference == SAMPLE_QUESTIONS[0].reference


def test_collect_answers_records_latency():
    ev = make_mock_evaluator()
    results = ev._collect_answers(SAMPLE_QUESTIONS, verbose=False)
    assert results[0].latency_ms > 0


def test_collect_answers_handles_pipeline_error():
    ev = make_mock_evaluator()
    ev.pipeline.ask.side_effect = RuntimeError("LLM timeout")
    results = ev._collect_answers(SAMPLE_QUESTIONS[:1], verbose=False)
    assert results[0].error is not None
    assert "timeout" in results[0].error.lower()


# ── EvalReport tests ───────────────────────────────────────────────────────

def make_report(n: int = 3) -> EvalReport:
    results = [
        SingleResult(
            question=f"Q{i}", answer=f"A{i}",
            retrieved_chunks=["chunk text"],
            reference=f"Ref {i}",
            faithfulness=0.8 + i * 0.05,
            answer_relevancy=0.75 + i * 0.05,
            context_precision=0.7 + i * 0.05,
            context_recall=0.65 + i * 0.05,
            latency_ms=300.0,
        )
        for i in range(n)
    ]
    return EvalReport(results=results, num_docs=2, num_chunks=20)


def test_report_mean_faithfulness():
    report = make_report(2)
    expected = (0.80 + 0.85) / 2
    assert abs(report.mean("faithfulness") - expected) < 0.001


def test_report_summary_keys():
    report = make_report()
    s = report.summary()
    for key in ("faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "num_questions", "num_errors", "avg_latency_ms"):
        assert key in s


def test_report_num_questions():
    report = make_report(4)
    assert report.summary()["num_questions"] == 4


def test_report_counts_errors():
    results = [SingleResult(
        question="Q", answer="", retrieved_chunks=[], reference=None,
        error="something went wrong",
    )]
    report = EvalReport(results=results)
    assert report.summary()["num_errors"] == 1


def test_report_mean_none_when_no_scores():
    results = [SingleResult(
        question="Q", answer="A", retrieved_chunks=[], reference=None,
    )]
    report = EvalReport(results=results)
    assert report.mean("faithfulness") is None


def test_report_save_and_load(tmp_path):
    report = make_report(2)
    path = str(tmp_path / "report.json")
    report.save(path)

    loaded = EvalReport.load(path)
    assert len(loaded.results) == 2
    assert abs(loaded.mean("faithfulness") - report.mean("faithfulness")) < 0.001


def test_report_save_creates_parent_dirs(tmp_path):
    report = make_report(1)
    path = str(tmp_path / "nested" / "dir" / "report.json")
    report.save(path)
    import os; assert os.path.exists(path)


def test_report_print_summary_runs(capsys):
    report = make_report(2)
    report.print_summary()
    out = capsys.readouterr().out
    assert "Faithfulness" in out
    assert "Context Precision" in out


def test_single_result_scores_excludes_none():
    r = SingleResult(
        question="Q", answer="A", retrieved_chunks=[], reference=None,
        faithfulness=0.9,
        # answer_relevancy left as None
    )
    scores = r.scores()
    assert "faithfulness" in scores
    assert "answer_relevancy" not in scores