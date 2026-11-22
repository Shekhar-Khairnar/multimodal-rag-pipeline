"""Rerank candidate chunks with a cross-encoder.

An embedding model encodes the question and each chunk separately and compares
the two vectors. A cross-encoder reads the question and the chunk together and
outputs a relevance score, which is much more accurate. The cost is one model
call per (question, chunk) pair, so it is only applied to a short candidate list.
"""

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from mmrag.config import get_settings


@lru_cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(get_settings().reranker_model)


def rerank(question: str, docs: list[Document], k: int) -> list[Document]:
    """Return the k most relevant docs, with the cross-encoder score as metadata["score"]."""
    if not docs:
        return []
    scores = get_reranker().predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)[:k]
    for doc, score in ranked:
        doc.metadata["score"] = round(float(score), 4)
    return [doc for doc, _ in ranked]
