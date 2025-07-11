import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents_into_chunks(
    loaded_documents: List[Dict[str, str]],
    chunk_size: int = 200,
    chunk_overlap: int = 1000
) -> List[Dict[str, Any]]:
    """
    Splits loaded documents into chunks using RecursiveCharacterTextSplitter.

    Returns:
        List[Dict]: Each dict contains 'content' and 'metadata'.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    all_processed_chunks: List[Dict[str, Any]] = []

    for doc_data in loaded_documents:
        filename = doc_data.get('filename', 'unknown_file')
        full_text = doc_data.get('text', '')
        doc_type = doc_data.get('type', 'unknown_type')

        if not full_text:
            print(f"Skipping empty content for file: {filename}")
            continue

        doc_to_split = Document(
            page_content=full_text,
            metadata={"source": filename, "type": doc_type}
        )

        chunks = text_splitter.split_documents([doc_to_split])

        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.splitext(filename)[0].replace(' ', '_')}_{i:03d}"
            metadata = {
                'source': filename,
                'chunk_id': chunk_id
            }

            if 'page' in chunk.metadata:
                metadata['page'] = chunk.metadata['page']

            all_processed_chunks.append({
                'content': chunk.page_content,
                'metadata': metadata
            })

    return all_processed_chunks
