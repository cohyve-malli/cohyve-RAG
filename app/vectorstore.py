import os
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from embedder import load_embedder

DB_DIR = "faiss_db"

def build_vectorstore(chunks):
    """
    chunks: List of dicts with keys:
      - content: str
      - metadata: dict (must include 'chunk_id', 'source')
    """
    documents = [
        Document(
            page_content=chunk["content"],
            metadata=chunk["metadata"]  # will have chunk_id, source, maybe page
        )
        for chunk in chunks
    ]

    embedder = load_embedder()
    vectorstore = FAISS.from_documents(documents, embedder)

    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    vectorstore.save_local(DB_DIR)
    print(f"✅ FAISS vectorstore saved to: {DB_DIR}")

def load_vectorstore():
    embedder = load_embedder()
    return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
