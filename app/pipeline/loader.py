import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_pdf_content(filepath: str) -> str:
    try:
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        return full_text
    except Exception as e:
        print(f"Error loading PDF '{filepath}': {e}")
        return ""

def load_txt_content(filepath: str) -> str:
    try:
        loader = TextLoader(filepath)
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        return full_text
    except Exception as e:
        print(f"Error loading TXT '{filepath}': {e}")
        return ""

def load_documents(directory_path: str) -> list[dict]:
    """
    Loads all supported documents from a directory.
    Returns:
        List[Dict]: Each dict has 'filename', 'text', and 'type'.
    """
    all_loaded_documents = []

    for root, _, files in os.walk(directory_path):
        for fname in files:
            filepath = os.path.join(root, fname)
            file_extension = os.path.splitext(fname)[1].lower()
            content = ""
            doc_type = ""

            if file_extension == ".pdf":
                content = load_pdf_content(filepath)
                doc_type = "pdf"
            elif file_extension == ".txt":
                content = load_txt_content(filepath)
                doc_type = "txt"
            else:
                print(f"Unsupported file type: {filepath}")
                continue

            if content:
                all_loaded_documents.append({
                    "filename": fname,
                    "text": content,
                    "type": doc_type
                })

    return all_loaded_documents
