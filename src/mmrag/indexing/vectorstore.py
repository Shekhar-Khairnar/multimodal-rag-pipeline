"""Embeddings and the Pinecone vector store."""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from mmrag.config import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformers model once. It runs locally, with no API calls."""
    return HuggingFaceEmbeddings(
        model_name=get_settings().embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # unit-length vectors suit cosine similarity
    )


@lru_cache
def get_index():
    """Return the Pinecone index, creating it on first use. Cached, so the
    connection and existence check happen once per process, not per query."""
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())

    if not pc.has_index(settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,  # must match the embedding model's output size
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return pc.Index(settings.pinecone_index_name)


def get_vectorstore(index=None) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=index or get_index(),
        embedding=get_embeddings(),
        namespace=get_settings().pinecone_namespace,
    )


def index_documents(docs: list[Document]) -> None:
    """Replace everything in the namespace with `docs`.

    The namespace is cleared first so chunks that disappear after re-ingesting
    (e.g. a changed PDF) do not linger. chunk_id is used as the vector ID, so
    every chunk keeps the same ID across runs.
    """
    index = get_index()
    namespace = get_settings().pinecone_namespace

    if namespace in index.describe_index_stats().namespaces:
        index.delete(delete_all=True, namespace=namespace)

    get_vectorstore(index).add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
