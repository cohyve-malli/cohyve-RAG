# embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model globally for reuse
_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_embedder():
    """
    Return the SentenceTransformer model (for backward compatibility)
    """
    return _model

def embed_texts(chunks):
    """
    Takes a list of dicts with `content` key and returns a list of vectors
    """
    texts = [chunk["content"] for chunk in chunks]
    vectors = _model.encode(texts)
    return vectors

def embed_query(text):
    return _model.encode([text])[0]  # returns a 1D numpy array

def cosine_score(query_vec, chunk_vecs):
    return cosine_similarity([query_vec], chunk_vecs)[0]  # returns 1D array

def get_top_k_chunks(query, chunks, k=5, similarity_threshold=0.6):
    """
    Filter the top-k most relevant chunks using cosine similarity.
    """
    query_vec = embed_query(query)
    chunk_vecs = embed_texts(chunks)
    similarities = cosine_score(query_vec, chunk_vecs)

    top_indices = np.argsort(similarities)[::-1][:k]

    top_chunks = []
    for i in top_indices:
        if similarities[i] >= similarity_threshold:
            top_chunks.append({
                "content": chunks[i]["content"],
                "metadata": chunks[i]["metadata"],
                "similarity": float(similarities[i])
            })

    return top_chunks
