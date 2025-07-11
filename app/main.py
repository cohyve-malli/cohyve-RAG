import argparse
from app.config import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, DATA_PATH, VECTORSTORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP) 
from app.utils.logger import logger

from app.pipeline.loader import load_documents
from app.pipeline.splitter import split_documents_into_chunks
from app.pipeline.embedder import embed_texts
from app.pipeline.vectorstore import build_vectorstore, load_vectorstore

from app.pipeline.retriever import retrieve_context
from app.pipeline.prompts import format_prompt
from app.pipeline.generator import generate_answer


def run_ingest():
    logger.info("Starting ingestion pipeline...")

    docs = load_documents(DATA_PATH)
    all_chunks = []
    for doc in docs:
        chunks = split_documents_into_chunks(
            [doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        all_chunks.extend(chunks)

    build_vectorstore(all_chunks)

def run_query(question: str):
    logger.info(f"Querying: {question}")
    docs = retrieve_context(question, top_k=5)

    logger.info(f"Retrieved {len(docs)} chunks.")
    for doc in docs:
        logger.info(f"From: {doc.metadata['source']} | {doc.metadata['chunk_id']}")

    prompt = format_prompt(docs, question)

    answer = generate_answer(
        prompt,
        model_name=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        api_key=GEMINI_API_KEY
    )
    
    print("\nFinal Answer:")
    print(answer)


def main():
    parser = argparse.ArgumentParser(description="RAG CLI (Backend Only)")
    parser.add_argument("mode", choices=["ingest", "query"], help="Choose 'ingest' or 'query'")
    parser.add_argument("--question", type=str, help="Question to ask (only for query mode)")

    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingest()
    elif args.mode == "query":
        if not args.question:
            print("❗ Please provide a --question for query mode")
            return
        run_query(args.question)

if __name__ == "__main__":
    main()
