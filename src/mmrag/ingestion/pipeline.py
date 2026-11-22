"""Ingestion pipeline: turn PDFs into text, table and visual Documents ready for indexing."""

from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from mmrag.config import get_settings
from mmrag.ingestion.image_utils import extract_visuals
from mmrag.ingestion.pdf_parser import doc_metadata, parse_page
from mmrag.ingestion.vlm_summarizer import visual_to_document


def extract_multimodal_documents(pdf_path) -> list[Document]:
    """Extract every text, table and visual Document from one PDF.

    Visuals are located first so their areas can be excluded from the page
    text; otherwise chart labels such as "$0 $250 Q1'24" would be indexed
    as narrative text.
    """
    path = Path(pdf_path)
    assets_dir = get_settings().extracted_assets_dir
    docs = []

    with pymupdf.open(path) as pdf:
        meta = doc_metadata(pdf, path)
        for page in pdf:
            visuals = extract_visuals(page, meta["doc_id"], assets_dir)
            chart_areas = [v["rect"] for v in visuals]

            docs.extend(parse_page(page, meta, exclude=chart_areas))
            docs.extend(visual_to_document(v, meta) for v in visuals)

    return docs


def ingest_directory(data_dir: Path | None = None) -> list[Document]:
    """Extract Documents from every PDF in a folder (defaults to settings.raw_data_dir)."""
    data_dir = Path(data_dir or get_settings().raw_data_dir)
    docs = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        docs.extend(extract_multimodal_documents(pdf_path))
    return docs


if __name__ == "__main__":
    from collections import Counter

    docs = ingest_directory()
    print(f"{len(docs)} documents:", dict(Counter(d.metadata["modality"] for d in docs)))
    for source, count in sorted(Counter(d.metadata["source"] for d in docs).items()):
        print(f"  {count:3}  {source}")
