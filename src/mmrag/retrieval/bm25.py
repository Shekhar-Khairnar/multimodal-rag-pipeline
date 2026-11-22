"""Keyword search over the indexed chunks with BM25.

Dense embeddings match meaning but are weak on exact tokens such as acronyms,
codes and specific numbers ("MTBF", "SINGLE-SOURCE", "Q2'25"). BM25 matches
those exact terms, so the two complement each other.

BM25 runs locally: chunks are saved to a JSONL file at ingestion time and
loaded into memory on first search.
"""

import json
import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from mmrag.config import get_settings


def _chunks_path():
    return get_settings().index_dir / "chunks.jsonl"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def save_chunks(docs: list[Document]) -> None:
    """Store the chunks that BM25 searches. Called by the ingest script."""
    path = _chunks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}) + "\n")
    _load_index.cache_clear()


@lru_cache
def _load_index() -> tuple[BM25Okapi, list[Document]]:
    with _chunks_path().open(encoding="utf-8") as f:
        docs = [Document(**json.loads(line)) for line in f]
    return BM25Okapi([tokenize(d.page_content) for d in docs]), docs


def bm25_search(question: str, n: int) -> list[Document]:
    """Return the n chunks with the highest BM25 score, best first."""
    index, docs = _load_index()
    scores = index.get_scores(tokenize(question))
    best = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:n]
    # Copies, so callers can add metadata (e.g. scores) without changing the cached chunks.
    return [Document(page_content=docs[i].page_content, metadata=dict(docs[i].metadata)) for i in best]
