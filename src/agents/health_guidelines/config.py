"""Configuration for this RAG module."""
MODEL_NAME = "openai/gpt-oss-120b"

PDF_PATHS = [
    "src/agents/health_guidelines/data/healthguidelines.pdf"
]

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

VECTOR_DB_PATH = "src/agents/health_guidelines/data/vector_store"
