import os
from typing import List
from app.pipeline.vectorstore import load_vectorstore
from app.pipeline.generator import generate_answer
from langchain_core.documents import Document


def generate_query_variants(user_query: str, n: int = 4) -> List[str]:
    """
    Generate `n` rephrased variants of the user query using Gemini.
    """
    prompt = f"""
You are a helpful assistant. Given the user question below, generate {n} diverse, semantically different rephrasings of the question. Return only the rephrased versions as a bullet list.

User Question:
{user_query}
"""

    output = generate_answer(prompt) 
    variants = []

    for line in output.splitlines():
        line = line.strip("•*- ").strip()
        if line:
            variants.append(line)

    return variants[:n]


def retrieve_top_k_multiquery(query: str, k: int = 3, n_queries: int = 4) -> List[Document]:
    """
    Perform semantic retrieval using multiple rephrased versions of the user query.
    """
    # Step 1: Load the FAISS vectorstore
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Step 2: Generate rephrased queries (multi-query style)
    all_queries = [query] + generate_query_variants(query, n=n_queries)

    # Step 3: Accumulate all results
    all_results = []
    seen_chunks = set()

    for q in all_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            # Deduplicate using chunk_id
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                all_results.append(doc)

    return all_results[:k]  # Return top-k unique chunks

def retrieve_context(query: str, top_k: int = 5) -> List[Document]:
    return retrieve_top_k_multiquery(query, k=top_k)
