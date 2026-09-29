from config import *

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

###############################################################
# Load All PDFs
###############################################################

documents = []

for pdf_path in PDF_PATHS:

    print(f"Loading: {pdf_path}")

    loader = PyMuPDFLoader(pdf_path)

    docs = loader.load()

    # Store PDF name in metadata
    for doc in docs:
        doc.metadata["source"] = pdf_path

    documents.extend(docs)

###############################################################
# Split Documents
###############################################################

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

###############################################################
# Embeddings
###############################################################

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

###############################################################
# Create Vector Store
###############################################################

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

vector_db.save_local(VECTOR_DB_PATH)

print(f"\nVector Database Created Successfully")
print(f"Total PDFs Loaded : {len(PDF_PATHS)}")
print(f"Total Chunks      : {len(chunks)}")