"""Extract every PDF in the raw data folder and index the results in Pinecone.

Usage:
    python scripts/ingest.py
"""

from collections import Counter

from mmrag.indexing.vectorstore import index_documents
from mmrag.ingestion.pipeline import ingest_directory
from mmrag.retrieval.bm25 import save_chunks


def main() -> None:
    docs = ingest_directory()
    print(f"Extracted {len(docs)} chunks:", dict(Counter(d.metadata["modality"] for d in docs)))

    index_documents(docs)
    print("Indexed in Pinecone.")

    save_chunks(docs)
    print("Saved chunks for BM25 keyword search.")


if __name__ == "__main__":
    main()
