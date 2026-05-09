---
title: "Second Brain Project Summary"
author: "Developer"
tags: [rag, retrieval, embeddings, chunking, bm25, vector-search, citations, streamlit]
---

# Second Brain — Full Project Summary (Weeks 1–6)

## What Was Built

A production-quality RAG system from scratch: ingest notes/PDFs/URLs → chunk → embed → hybrid search → cited answers → Streamlit UI.

---

## Week 1–2 — Ingestion Pipeline

### Files Built
- core/document.py
- ingestion/base.py
- ingestion/pdf_ingestor.py
- ingestion/markdown_ingestor.py
- ingestion/url_ingestor.py
- ingestion/pipeline.py

### What It Does

Routes any source (PDF, markdown file, URL) to the right ingestor and produces a unified Document object with a stable sha256 ID. Deduplicates on re-ingestion so the same file can be run through the pipeline multiple times without creating duplicates.

The Document object carries the raw extracted text, a human-readable title, provenance fields (source_type, source_uri), timestamps, optional metadata (author, tags, language), and a computed doc_id which is a sha256 hash of source_uri and title truncated to 16 characters.

### Issues and Fixes

- datetime.utcnow() was deprecated in Python 3.12 and was replaced with datetime.now(timezone.utc) everywhere.
- from pathlib import Path was accidentally stripped when removing sys.path lines and had to be added back manually to pipeline.py, base.py, pdf_ingestor.py, and markdown_ingestor.py.
- python-frontmatter raised a codecs deprecation warning which was suppressed in pytest.ini with filterwarnings.
- A PDF bug occurred where len(doc) was called after doc.close(), so page_count was saved before closing.

---

## Week 3 — Chunking, Embeddings, Vector Store

### Files Built
- core/chunk.py
- chunking/chunker.py
- embeddings/embedder.py
- embeddings/vector_store.py

### What It Does

Implements 3-tier semantic chunking: paragraph boundaries first, then sentence boundaries for oversized paragraphs, then hard token-level splitting as a last resort. Overlap tokens from the previous chunk are prepended to each new chunk to preserve cross-boundary context.

Supports two embedding backends:
- Local: sentence-transformers all-MiniLM-L6-v2, 384 dimensions, free, no API key needed, runs on CPU
- OpenAI: text-embedding-3-small, 1536 dimensions, approximately $0.02 per million tokens

Uses Qdrant as the vector store with three modes: in-memory for development, local disk persistence, and server mode for Docker or cloud deployments.

### Issues and Fixes

- qdrant-client v1.10 and above removed the .search() method, which was replaced with .query_points().points.
- A cosine similarity test was flaky because vectors [3,0,0,0] and [4,0,0,0] point in the same direction and have identical cosine similarity. The assertion was fixed to check that the top result is one of the close ones rather than requiring an exact rank.

---

## Week 4 — Hybrid Search with BM25 and RRF

### Files Built
- retrieval/bm25_index.py
- retrieval/rrf.py
- retrieval/hybrid_retriever.py

### What It Does

BM25 (Best Match 25) is a keyword ranking algorithm used by Elasticsearch and Solr. It beats plain TF-IDF because it normalises for document length and applies term saturation. The BM25 index supports save and load so it does not need to be rebuilt on every run.

Reciprocal Rank Fusion (RRF) merges the vector search results and BM25 results into a single ranked list. For each document across each ranked list, the RRF score is the sum of 1 divided by (k plus rank), where k equals 60. Documents appearing in both lists accumulate higher scores. RRF is robust because it uses only rank positions and not raw scores, so it works regardless of how different retrievers scale their outputs.

The HybridRetriever combines both approaches and includes an explain_rrf() debugger that shows the score breakdown per document.

### Issues and Fixes

- No issues. All 20 tests passed on the first run after rank-bm25 was installed.

---

## Week 5 — Citation-Grounded Generation

### Files Built
- generation/generator.py
- week5_pipeline.py

### What It Does

Formats retrieved chunks as numbered sources labelled [source_1], [source_2], and so on. Prompts Gemini to answer with inline citations using these markers. Parses the markers back to exact chunk objects after generation. Returns a CitedAnswer object containing the answer text, a list of CitedSource objects (each with chunk_id, title, source_uri, and text), all retrieved chunks, the original query, and the model name.

The format_answer() method renders the answer with a References section below it showing the title, URI, and a text preview for each cited source.

The system prompt instructs the model to cite every factual claim with [source_N] immediately after the claim, to cite multiple sources when a claim is supported by more than one chunk, to say so clearly if the sources lack enough information, and never to make up information not found in the sources.

### Issues and Fixes

- Unicode ellipsis characters in format_answer() caused a SyntaxError on Windows and were replaced with plain ASCII.
- The original google.generativeai SDK was deprecated. Migrated to google.genai (the new SDK) using client.models.generate_content().
- Tests were patching the wrong object after the SDK migration. Fixed to patch _client.models.generate_content.
- LLM provider history: started with Anthropic, switched to Groq, then switched to Gemini which offers a free tier of 1500 requests per day.

---

## Week 6 — Streamlit UI

### Files Built
- app.py
- ui/chat_page.py
- ui/timeline_page.py
- ui/ingest_page.py

### What It Does

A three-page Streamlit application. The Chat page lets users ask questions and receive cited answers with expandable source cards. The Timeline page shows all documents sorted by date with type badges, word counts, tags, and a filter bar. The Ingest page has tabs for URL, file upload, and paste-text with a queue system that lets users stage multiple sources before building the index.

### Issues and Fixes

- The ingest page used a local sources_to_add list that was wiped on every Streamlit rerun. Moved to st.session_state.queued_sources so the queue persists across reruns.
- A "No sources to index" error appeared even after uploading because the user clicked Build without clicking Add files first. The UI now shows a queue preview and disables the Build button until sources are queued.
- torchvision warnings from the transformers library were harmless and fixed by running pip install torchvision.
- The API key entered in the sidebar was set into os.environ but wiped on each Streamlit rerun. Now saved to st.session_state["gemini_key"] and reloaded as the default value on each rerun.
- The CitationGenerator was created at index-build time with no key. chat_page.py now reinitialises CitationGenerator fresh on every question with the current key.

---

## Import Path Issues (Recurring Across All Weeks)

The most persistent problem throughout the project. Python only adds the script's own directory to sys.path when running directly. All intra-project imports such as from retrieval.hybrid_retriever import failed unless the project root was on the path.

Attempted fixes that partially worked: sys.path.insert(0, ...) in every file worked for tests but caused conflicts. conftest.py plus pytest.ini pythonpath=. fixed pytest but not scripts.

The final fix that works permanently is running pip install -e . which installs the project as an editable package. With setup.py present, this registers the project root as the package root so all imports resolve correctly everywhere including scripts, tests, and Streamlit, with no sys.path hacks needed.

---

## Test Suite — 58 Tests, All Passing

- test_ingestion.py: 12 tests covering document IDs, markdown parsing, deduplication, and error handling
- test_week3.py: 12 tests covering chunk IDs, chunker size limits, vector store upsert and search
- test_week4.py: 20 tests covering BM25 tokeniser, index build, search, save and load, and RRF formula
- test_week5.py: 16 tests covering prompt construction, citation extraction, and mocked Gemini API

---

## Chunking Strategy — Semantic Boundary Chunking

The SemanticChunker works in four steps. First it splits text into paragraphs at blank lines and markdown headings. Second it merges small paragraphs into chunks up to chunk_size tokens (default 400). Third it splits oversized paragraphs at sentence boundaries. Fourth it applies chunk_overlap tokens (default 80) of the previous chunk as a prefix to each new chunk so retrieval does not miss context that straddles a chunk boundary.

This beats fixed-size chunking because it keeps related sentences together and avoids cutting mid-sentence, which degrades retrieval quality.

---

## Vector Store — Qdrant

Qdrant is an open-source vector database. The VectorStore wrapper supports three modes. Memory mode runs in-process and data is lost on exit, making it ideal for development. Local mode persists to disk at a specified path and requires no Docker. Server mode connects to a running Qdrant instance for Docker or production deployments.

Chunks are stored as Qdrant points with an integer ID derived from the hex chunk_id, a vector embedding, and a payload containing all metadata fields. Search uses cosine distance and returns the top_k most similar chunks along with their scores and metadata.

---

## Weeks Remaining

- Week 7: RAGAS evaluation framework to measure retrieval quality with real metrics
- Week 8: Deploy to Railway with a live URL, polished README, and demo GIF

..
What we started with: A complete 6-week RAG project on your local machine, but you hadn't touched it in a while and needed to re-orient.

Re-orientation (start of session)
Walked through the full architecture with an interactive diagram — 5 layers from ingestion to Streamlit UI, what each file does, and how data flows through the pipeline.

Testing the existing codebase (Weeks 1–6)
You uploaded all your source files. We:

Reconstructed the full project structure in the sandbox
Discovered core/chunk.py was missing — inferred and created it, then replaced it with your real version
Created the three missing ingestor files (pdf_ingestor.py, markdown_ingestor.py, url_ingestor.py) since they weren't uploaded
Installed all dependencies from requirements.txt
Ran all 58 tests → 57/58 passing, with the 1 failure being a tiktoken CDN block in the sandbox (passes fine on your machine)
Ran a full end-to-end smoke test through all 7 pipeline stages with fake embeddings to confirm everything connects correctly


Week 7 — RAGAS Evaluation (built this session)
Created four new files:
FileWhat it doesevaluation/eval_dataset.py6 curated Q&A pairs about your project to test againstevaluation/metrics.pyWires RAGAS metrics to Gemini as the judge LLMevaluation/evaluator.pyRAGEvaluator class — runs retrieval+generation then scores with RAGASweek7_eval.pyCLI script to run everything from the command linetests/test_week7.py24 tests for the evaluation module, all mocked (no API key needed)
The four metrics it measures:
MetricWhat a low score tells youFaithfulnessGenerator is hallucinating — claims not grounded in retrieved chunksAnswer RelevancyGenerator is going off-topic or giving incomplete answersContext PrecisionRetriever is returning noisy/irrelevant chunksContext RecallRetriever is missing chunks that contain the answer
Bugs fixed along the way:

scikit-network C++ build error on Windows → pip install --only-binary=:all:
notes/ folder didn't exist → created it and generated second_brain_summary.md as the knowledge base
Windows trailing slash bug on Path.is_dir() → strip with rstrip("/\\")
--questions-only mode still hit Gemini because Week5Pipeline.__init__ always creates the generator → fixed by passing a placeholder key, but ultimately you still need a real key since generation always calls Gemini

Current status: The pipeline runs end-to-end. Ingestion → chunking (6 chunks) → embedding → Qdrant → BM25 → hybrid retrieval all working. Just needs your real Gemini key to complete generation and RAGAS scoring.

What's left:

Run python week7_eval.py --docs notes --key AIza... with your real key to get your first RAGAS scores
Week 8: Deploy to Railway with a live URL, polished README, and demo GIF