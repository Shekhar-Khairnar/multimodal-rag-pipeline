"""Extract every PDF in the raw data folder and index the results in Pinecone.

Usage:
    python scripts/ingest.py
"""

from collections import Counter

from mmrag.indexing.vectorstore import index_documents
from mmrag.ingestion.pipeline import ingest_directory


def main() -> None:
    docs = ingest_directory()
    print(f"Extracted {len(docs)} chunks:", dict(Counter(d.metadata["modality"] for d in docs)))

    index_documents(docs)
    print("Indexed in Pinecone.")


if __name__ == "__main__":
    main()
