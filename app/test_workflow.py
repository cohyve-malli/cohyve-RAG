from vectorstore import build_vectorstore, load_vectorstore

shivani_chunks = [
    {
        "content": "Retrieval-Augmented Generation (RAG) is a hybrid approach...",
        "metadata": {
            "chunk_id": "rag_intro_0001",
            "source": "doc1.pdf"
        }
    },
    {
        "content": "It combines information retrieval and generative models...",
        "metadata": {
            "chunk_id": "rag_intro_0002",
            "source": "doc1.pdf"
        }
    }
]

# Build and store
build_vectorstore(data)

# Load and query
vs = load_vectorstore()
results = vs.similarity_search("What is RAG?", k=1)

print("Retrieved Document:")
print(results[0].page_content)
print("Metadata:", results[0].metadata)
