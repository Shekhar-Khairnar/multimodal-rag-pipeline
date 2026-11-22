"""Summarize charts, diagrams and images with a vision-language model (VLM).

Embedding models only read text, so each visual is converted into a short,
factual text summary. The summary is embedded and searched like any other
chunk, while the Document keeps the image path so the original image can be
passed to the VLM again when answering a question.

Summaries are cached by image content hash, so re-running ingestion does not
call the VLM again for images it has already seen.
"""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from mmrag.config import PROJECT_ROOT, get_settings
from mmrag.ingestion.image_utils import image_to_data_uri
from mmrag.llm import vision_llm

# The most important facts go first: the embedding model only reads the
# first 256 tokens of a chunk.
SUMMARY_PROMPT = """You are describing a visual extracted from a business document so that it can be found by search.

Write a plain-text summary of at most 150 words:
1. One sentence stating what kind of visual it is and what it shows, using its title or caption.
2. The key facts:
   - Charts: the exact labeled values, the highest and lowest values, and the overall trend.
   - Diagrams: the main components and how they connect (A -> B), including any labels on arrows.

Only describe what is visible. If a value has to be read from gridlines, say "approximately".
Do not use Markdown."""


def summarize_image(image_path: str) -> str:
    """Return a text summary of an image, using the cache when possible."""
    image_bytes = Path(image_path).read_bytes()
    cache_dir = get_settings().extracted_assets_dir / "vlm_cache"
    cache_file = cache_dir / f"{hashlib.sha256(image_bytes).hexdigest()[:16]}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    # A multimodal message: text instructions plus the image as a base64 data URI.
    message = HumanMessage(content=[
        {"type": "text", "text": SUMMARY_PROMPT},
        {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
    ])
    summary = vision_llm().invoke([message]).content.strip()
    if len(summary.split()) < 20:  # empty or truncated responses must not be cached
        raise ValueError(f"VLM returned an unusable summary for {image_path}: {summary!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(summary, encoding="utf-8")
    return summary


def visual_to_document(visual: dict, meta: dict) -> Document:
    """Turn one visual from image_utils.extract_visuals() into a searchable Document.

    meta: dict from pdf_parser.doc_metadata()
    """
    page_no = visual["page"]
    summary = summarize_image(visual["image_path"])
    # Stored relative to the project root so the index works on any machine.
    image_path = Path(visual["image_path"]).resolve().relative_to(PROJECT_ROOT).as_posix()

    return Document(
        page_content=f'Chart or figure from "{meta["doc_title"]}", page {page_no}:\n{summary}',
        metadata={
            **meta,
            "page": page_no,
            "modality": "visual",
            "image_path": image_path,
            "chunk_id": f"{meta['doc_id']}::p{page_no}::img{visual['index']}",
        },
    )


if __name__ == "__main__":
    # Summarize every visual in the PDF given on the command line.
    import sys

    import pymupdf

    from mmrag.ingestion.image_utils import extract_visuals
    from mmrag.ingestion.pdf_parser import doc_metadata

    path = Path(sys.argv[1])
    with pymupdf.open(path) as pdf:
        meta = doc_metadata(pdf, path)
        for page in pdf:
            for visual in extract_visuals(page, meta["doc_id"], get_settings().extracted_assets_dir):
                doc = visual_to_document(visual, meta)
                print("=" * 80)
                print(doc.metadata)
                print("-" * 80)
                print(doc.page_content)
