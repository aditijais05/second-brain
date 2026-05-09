from ingestion.pipeline import IngestionPipeline
from week4_pipeline import Week4Pipeline

docs = IngestionPipeline().run([
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
])

pipeline = Week4Pipeline(embedding_backend="local")
pipeline.index(docs)

# Semantic query — vector search wins here
results = pipeline.search("how does RAG improve accuracy?", top_k=3)
pipeline.print_results(results, query="how does RAG improve accuracy?")

# Exact keyword query — BM25 wins here, RRF combines both
results = pipeline.search("FAISS vector index", top_k=3, explain=True)
pipeline.print_results(results, query="FAISS vector index")