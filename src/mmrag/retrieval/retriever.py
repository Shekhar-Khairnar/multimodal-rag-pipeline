"""Retrieve relevant chunks for a question and format them as numbered, citable context.

Pipeline (each stage can be switched off in settings for comparison):
  1. dense search      top-N chunks by embedding similarity (Pinecone)
  2. hybrid search     + top-N chunks by keyword match (BM25), merged with Reciprocal Rank Fusion
  3. reranking         a cross-encoder reorders the merged candidates; the best k are kept

retrieve_many() runs this pipeline once per search query (see query_rewriter.py).
"""

from langchain_core.documents import Document

from mmrag.config import get_settings
from mmrag.indexing.vectorstore import get_vectorstore
from mmrag.retrieval.bm25 import bm25_search
from mmrag.retrieval.reranker import rerank

RRF_K = 60  # standard damping constant from the original RRF paper


def dense_search(question: str, n: int) -> list[Document]:
    """Top-n chunks by cosine similarity, with the similarity as metadata["score"]."""
    docs = []
    for doc, score in get_vectorstore().similarity_search_with_score(question, k=n):
        doc.metadata["score"] = round(float(score), 4)
        docs.append(doc)
    return docs


def rrf_fuse(rankings: list[list[Document]]) -> list[Document]:
    """Merge ranked lists with Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank).

    Only ranks are used, so BM25 scores and cosine similarities (which are on
    different scales) never need to be compared. Chunks ranked well in both
    lists rise to the top.
    """
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            chunk_id = doc.metadata["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
            docs_by_id.setdefault(chunk_id, doc)

    fused = sorted(scores, key=scores.get, reverse=True)
    for chunk_id in fused:
        docs_by_id[chunk_id].metadata["score"] = round(scores[chunk_id], 4)
    return [docs_by_id[chunk_id] for chunk_id in fused]


def retrieve(question: str, k: int | None = None) -> list[Document]:
    """Return the k most relevant chunks, best first, each with metadata["score"].

    The score comes from the last stage that ran: cross-encoder relevance,
    RRF score, or cosine similarity.
    """
    settings = get_settings()
    k = k or settings.top_k
    n = max(k, settings.retrieval_candidates)

    candidates = dense_search(question, n)
    if settings.hybrid_search:
        candidates = rrf_fuse([candidates, bm25_search(question, n)])
    if settings.rerank:
        return rerank(question, candidates, k)
    return candidates[:k]


def retrieve_many(queries: list[str], k: int | None = None) -> list[Document]:
    """Retrieve k chunks for each query and merge them in query order, without duplicates.

    Each query keeps its own top-k, so one fact's chunks cannot crowd out another's.
    """
    seen: set[str] = set()
    merged = []
    for query in queries:
        for doc in retrieve(query, k):
            if doc.metadata["chunk_id"] not in seen:
                seen.add(doc.metadata["chunk_id"])
                merged.append(doc)
    return merged


def format_context(docs: list[Document]) -> str:
    """Join chunks into one context string, numbered so the LLM can cite them.

    Output format:
        [1] Source: report.pdf | page 3 | table
        <chunk text>

        [2] Source: ...
    """
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        # Pinecone returns numbers as floats (3.0), hence int().
        header = f"[{i}] Source: {meta['source']} | page {int(meta['page'])} | {meta['modality']}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Which campus region has the highest PUE?"
    print(format_context(retrieve(question)))
