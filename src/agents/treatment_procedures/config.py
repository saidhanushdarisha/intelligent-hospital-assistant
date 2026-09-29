"""Configuration for this RAG module."""
MODEL_NAME = "openai/gpt-oss-120b"

PDF_PATHS = [
    "src/agents/treatment_procedures/data/TreatmentProcedure.pdf"
]

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

VECTOR_DB_PATH = "src/agents/treatment_procedures/data/vector_store"
