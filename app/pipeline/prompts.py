from typing import List
from langchain_core.documents import Document

def format_prompt(retrieved_docs: List[Document], user_question: str) -> str:
    """
    Formats the final prompt for the LLM with fallback:
    - Uses context if available
    - Falls back to LLM's reasoning if context is missing
    """

    if not retrieved_docs:
        return f"""
        You are a highly intelligent and helpful AI assistant. There is no retrieved context available for this question.
        Use your internal knowledge to generate a comprehensive, accurate, and clear answer.

        Question:
        {user_question}

        Answer:
        """.strip()

    # Build context from available documents
    context_blocks = [
        f"[Document {i+1} | Source: {doc.metadata.get('source', 'unknown')} | Chunk ID: {doc.metadata.get('chunk_id', '')}]\n{doc.page_content.strip()}"
        for i, doc in enumerate(retrieved_docs)
    ]

    context = "\n\n".join(context_blocks)

    prompt = f"""
    You are a highly knowledgeable AI assistant. Use the following context to answer the user's question. 
    Prefer answering strictly from the context if it’s clearly relevant. If the context does not directly answer the question, 
    you are allowed to use your own understanding to generate a rich, accurate, and clear explanation.

    Context:
    {context}

    ---

    Question:
    {user_question}

    Answer:
    """.strip()

    return prompt
