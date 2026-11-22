"""Extract narrative text and tables from PDF pages as LangChain Documents.

Each page produces:
  * text Documents  - the page's paragraphs, split into ~700-character chunks (modality="text")
  * table Documents - each table, split into small row groups as Markdown (modality="table")

Every chunk starts with a line naming its source document and page, so a chunk
such as "Gross Profit | $508.4" stays tied to the right document when it is
retrieved on its own.

Example table chunk:
    Table from "Annual Report 2025", page 4 (rows 1-3 of 10):
    | Metric ($M) | FY2024 | FY2025 |
    | --- | --- | --- |
    | Revenue | $1,780.0 | $2,460.0 |
    ...

Charts and images are handled in image_utils.py.

Positions on a page are bounding boxes: (x0, y0, x1, y1), the top-left and
bottom-right corners of a rectangle.
"""

import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chunk sizes are chosen to fit the embedding model's input limit
# (all-MiniLM-L6-v2 truncates after 256 tokens).
# Numeric table cells use many tokens each, so table chunks hold only a few rows.
ROWS_PER_CHUNK = 3

# Splits at paragraph breaks first, then line breaks, then spaces. The overlap
# repeats the end of each chunk at the start of the next, so a sentence cut at
# a boundary still appears whole in one of the two chunks.
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)


def _inside(bbox, rect) -> bool:
    """True if the center of bbox lies inside rect.

    Using the center tolerates words that slightly overflow a cell border.
    """
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return rect.x0 <= center_x <= rect.x1 and rect.y0 <= center_y <= rect.y1


def _cell_text(words, cell) -> str:
    """Return the text of one table cell, built from the words positioned inside it.

    PyMuPDF's built-in table text extraction can reorder characters in
    formatted numbers (e.g. "$1,240.0" -> "$12400\\n, .") and drop spaces in
    headers, so cells are rebuilt from word positions instead.

    words: tuples from page.get_text("words"):
           (x0, y0, x1, y1, word, block_no, line_no, word_no)
    cell:  the cell's bbox, or None for an empty/merged cell
    """
    if cell is None:
        return ""
    cell_rect = pymupdf.Rect(cell)
    return " ".join(w[4] for w in words if _inside(w[:4], cell_rect))


def _markdown(header, rows) -> str:
    """Format a header row and data rows as a Markdown table."""
    all_rows = [header, ["---"] * len(header)] + rows
    return "\n".join("| " + " | ".join(row) + " |" for row in all_rows)


def doc_metadata(pdf: pymupdf.Document, path: Path) -> dict:
    """Metadata shared by every chunk from one PDF.

    source:    file name, used for citations
    doc_id:    file name as a lowercase slug, used for vector IDs and evaluation
    doc_title: title from the PDF's metadata, falling back to the file name
    """
    doc_id = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    title = pdf.metadata.get("title") or path.stem.replace("_", " ")
    return {"source": path.name, "doc_id": doc_id, "doc_title": title}


def parse_page(page: pymupdf.Page, meta: dict, exclude: list | None = None) -> list[Document]:
    """Return the text and table Documents for one page.

    meta:    dict from doc_metadata()
    exclude: rectangles (e.g. charts from image_utils.extract_visuals) whose
             text should not be included in the narrative text. Table areas
             are excluded automatically.
    """
    page_no = page.number + 1  # PyMuPDF pages are 0-indexed
    words = page.get_text("words")
    skip_regions = list(exclude or [])
    docs = []

    # 1. Tables: find_tables() locates tables and cells; _cell_text() reads each cell.
    for table_no, table in enumerate(page.find_tables().tables):
        skip_regions.append(pymupdf.Rect(table.bbox))  # keep table text out of the narrative text

        header = [_cell_text(words, cell) for cell in table.header.cells]
        rows = [[_cell_text(words, cell) for cell in row.cells] for row in table.rows]
        if not table.header.external:
            rows = rows[1:]  # the header is also the first row of the table

        # Split into row groups, repeating the header in each.
        for start in range(0, len(rows), ROWS_PER_CHUNK):
            chunk_rows = rows[start : start + ROWS_PER_CHUNK]
            end = start + len(chunk_rows)
            label = f'Table from "{meta["doc_title"]}", page {page_no} (rows {start + 1}-{end} of {len(rows)}):'

            docs.append(Document(
                page_content=label + "\n" + _markdown(header, chunk_rows),
                metadata={
                    **meta,
                    "page": page_no,
                    "modality": "table",
                    "chunk_id": f"{meta['doc_id']}::p{page_no}::t{table_no}::r{start + 1}",
                },
            ))

    # 2. Narrative text: text blocks (roughly paragraphs) outside tables and excluded regions.
    #    Block tuple: (x0, y0, x1, y1, text, block_no, block_type); block_type 0 = text, 1 = image.
    paragraphs = []
    for block in page.get_text("blocks", sort=True):  # sort=True gives reading order
        is_text = block[6] == 0
        in_skipped_area = any(_inside(block[:4], region) for region in skip_regions)
        if is_text and not in_skipped_area:
            paragraphs.append(block[4].strip())

    text_docs = []
    for i, chunk in enumerate(TEXT_SPLITTER.split_text("\n".join(paragraphs))):
        text_docs.append(Document(
            page_content=f'From "{meta["doc_title"]}", page {page_no}:\n{chunk}',
            metadata={
                **meta,
                "page": page_no,
                "modality": "text",
                "chunk_id": f"{meta['doc_id']}::p{page_no}::text{i}",
            },
        ))

    return text_docs + docs


def parse_pdf(pdf_path) -> list[Document]:
    """Extract text and table Documents from every page of a PDF."""
    path = Path(pdf_path)
    with pymupdf.open(path) as pdf:
        meta = doc_metadata(pdf, path)
        docs = []
        for page in pdf:
            docs.extend(parse_page(page, meta))
        return docs


if __name__ == "__main__":
    # Print every chunk extracted from the PDF given on the command line.
    import sys

    for doc in parse_pdf(sys.argv[1]):
        print("=" * 80)
        print(doc.metadata)
        print("-" * 80)
        print(doc.page_content)
