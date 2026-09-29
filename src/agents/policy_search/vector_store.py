from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from agents.policy_search.data.data import load_documents


def create_db():
    """Load the existing persistent policy DB without re-embedding documents."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        # The project already contains a persistent Chroma database in db/.
        # Opening it is much faster than Chroma.from_documents() on every import.
        db = Chroma(
            persist_directory="db",
            embedding_function=embeddings,
            collection_name="langchain",
        )
        if db._collection.count() > 0:
            return db
    except Exception:
        pass

    # First-run fallback: build the DB once, then future runs use the persistent copy.
    documents = load_documents()
    return Chroma.from_documents(
        documents,
        embeddings,
        persist_directory="db",
        collection_name="langchain",
    )
