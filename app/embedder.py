# embedder.py
from langchain.embeddings import HuggingFaceEmbeddings

def load_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(chunks):
    embedder = load_embedder()
    texts = [chunk["content"] for chunk in chunks]
    vectors = embedder.embed_documents(texts)
    return vectors
