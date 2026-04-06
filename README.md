# Second Brain — Personal RAG System

A portfolio-grade RAG system that turns your notes, PDFs, and bookmarks
into a queryable knowledge base with cited, source-grounded answers.

---

## Project structure

```
second_brain/
├── core/
│   └── document.py          # Unified Document dataclass
├── ingestion/
│   ├── base.py              # BaseIngestor interface
│   ├── pdf_ingestor.py      # PDF → Document  (PyMuPDF)
│   ├── markdown_ingestor.py # .md → Document  (python-frontmatter)
│   ├── url_ingestor.py      # URL → Document  (Trafilatura)
│   └── pipeline.py          # Unified router + deduplication
├── tests/
│   └── test_ingestion.py    # pytest suite
└── requirements.txt
```

---

## Week 1–2: Ingestion pipeline ✅

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Quick demo

```python
from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()
docs = pipeline.run([
    "notes/meeting.md",
    "papers/attention_is_all_you_need.pdf",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
])

for doc in docs:
    print(doc)
# Document(id='a3f1...', title='Attention Is All You Need', source=pdf, words=6842)
# Document(id='b92c...', title='LLM Powered Autonomous Agents', source=url, words=9103)
```

### Run tests

```bash
pytest tests/ -v --cov=ingestion --cov=core
```

---

## Roadmap

| Week | Topic                              | Status |
|------|------------------------------------|--------|
| 1–2  | Ingestion pipeline                 | ✅ Done |
| 3    | Chunking + embeddings              | 🔜     |
| 4    | Hybrid search (BM25 + vector RRF)  | 🔜     |
| 5    | Citation-grounded generation       | 🔜     |
| 6    | UI + knowledge timeline            | 🔜     |
| 7    | Eval framework (RAGAS)             | 🔜     |
| 8    | Deploy to Railway                  | 🔜     |