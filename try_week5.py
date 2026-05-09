from ingestion.pipeline import IngestionPipeline
from week5_pipeline import Week5Pipeline

docs = IngestionPipeline().run([
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
])

pipeline = Week5Pipeline(embedding_backend="local")
pipeline.index(docs)

answer = pipeline.ask("How does RAG reduce hallucinations?")
print(answer.format_answer())