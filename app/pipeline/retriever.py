import os
from typing import List
from collections import defaultdict
from app.pipeline.vectorstore import load_vectorstore
from app.pipeline.generator import generate_answer
from langchain_core.documents import Document

def generate_query_variants(user_query: str, n: int = 4) -> List[str]:
    """
    Generate `n` diverse rephrasings of the user query using Gemini.
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

def reciprocal_rank_fusion(results: List[List[Document]], k_rrf: int = 60) -> List[Document]:
    """
    Combine and rank results from multiple queries using Reciprocal Rank Fusion (RRF).
    Documents are deduplicated by their 'chunk_id' metadata and ranked by RRF score.
    """
    score_dict = defaultdict(float)
    doc_dict = {}

    for result_list in results:
        for rank, doc in enumerate(result_list):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue  # Skip docs without a unique identifier
            # RRF scoring: higher score for higher ranks (lower numbers)
            score_dict[chunk_id] += 1.0 / (k_rrf + rank + 1)
            if chunk_id not in doc_dict:
                doc_dict[chunk_id] = doc

    # Sort all unique documents by their RRF score, highest first
    sorted_chunk_ids = sorted(score_dict, key=score_dict.get, reverse=True)
    return [doc_dict[cid] for cid in sorted_chunk_ids]

def retrieve_top_k_multiquery(query: str, k: int = 3, n_queries: int = 4) -> List[Document]:
    """
    Perform semantic retrieval using multiple rephrased versions of the user query.
    Results are combined and ranked using Reciprocal Rank Fusion (RRF).
    Returns the top-k ranked, unique Documents.
    """
    # Step 1: Load the FAISS vectorstore and create a retriever
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Step 2: Generate the original and rephrased queries
    all_queries = [query] + generate_query_variants(query, n=n_queries)

    # Step 3: Retrieve results for each query variant
    all_results = []
    for q in all_queries:
        docs = retriever.invoke(q)
        all_results.append(docs)

    # Step 4: Fuse and rank the results using RRF
    ranked_docs = reciprocal_rank_fusion(all_results)

    # Step 5: Return the top-k ranked unique documents
    return ranked_docs[:k]

def retrieve_context(query: str, top_k: int = 5) -> List[Document]:
    """
    User-facing function to retrieve the top-k ranked context documents for a query.
    """
    return retrieve_top_k_multiquery(query, k=top_k)
