"""Retrieve relevant chunks for a question and format them as numbered, citable context."""

from langchain_core.documents import Document

from mmrag.config import get_settings
from mmrag.indexing.vectorstore import get_vectorstore


def retrieve(question: str, k: int | None = None) -> list[Document]:
    """Return the k chunks most similar to the question, best first.

    The similarity score (cosine, higher = more similar) is added to each
    Document's metadata as "score", so it can be shown in the UI and used
    in evaluation.
    """
    k = k or get_settings().top_k
    docs = []
    for doc, score in get_vectorstore().similarity_search_with_score(question, k=k):
        doc.metadata["score"] = round(float(score), 4)
        docs.append(doc)
    return docs


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
