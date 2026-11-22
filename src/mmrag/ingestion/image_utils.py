"""Detect charts, diagrams and images on PDF pages, save them as PNGs, and encode them for a VLM.

PDFs contain visuals in two forms:

1. Raster images: embedded picture files (PNG/JPEG), listed by page.get_images().
2. Vector graphics: charts and diagrams drawn from paths, rectangles and text.
   These are not images, so page.get_images() does not return them. They are
   located with page.cluster_drawings(), which groups nearby shapes into
   bounding rectangles.

Both forms are saved the same way: once a visual's rectangle is known, that
area of the page is rendered to a PNG with page.get_pixmap(clip=rect). The
rendered image also includes the axis labels, legends and captions inside
that area.
"""

import base64
import io
from pathlib import Path

import pymupdf
from PIL import Image

RENDER_DPI = 200  # high enough for a VLM to read small chart labels
MIN_WIDTH, MIN_HEIGHT = 100, 60  # in PDF points; skips logos, icons and divider lines
MIN_CHART_SHAPES = 5  # charts contain many shapes; shaded text boxes contain only one or two


def find_visual_regions(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    """Return [(kind, rect), ...] for every visual on the page.

    kind is "raster" (embedded image) or "vector" (drawn chart/diagram).
    """
    regions = []

    # Raster images: get_images() returns each image's xref (internal ID),
    # get_image_rects(xref) returns where it is drawn on the page.
    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            if rect.width >= MIN_WIDTH and rect.height >= MIN_HEIGHT:
                regions.append(("raster", rect))

    # Vector graphics: table borders and shaded text boxes are also drawings,
    # so a cluster only counts as a chart if it does not overlap a table,
    # is not tiny, and contains enough individual shapes.
    table_rects = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
    drawings = page.get_drawings()  # every individual shape on the page

    for rect in page.cluster_drawings():
        overlaps_table = any(rect.intersects(t) for t in table_rects)
        too_small = rect.width < MIN_WIDTH or rect.height < MIN_HEIGHT
        shape_count = sum(1 for d in drawings if rect.contains(d["rect"]))
        if not overlaps_table and not too_small and shape_count >= MIN_CHART_SHAPES:
            regions.append(("vector", rect))

    return regions


def extract_visuals(page: pymupdf.Page, doc_id: str, out_dir: Path) -> list[dict]:
    """Render every visual on the page to a PNG.

    Returns one dict per visual:
        {"kind": "vector", "rect": Rect(...), "page": 2, "index": 0,
         "image_path": ".../<doc_id>_p2_img0.png"}

    "rect" can be passed to pdf_parser.parse_page(exclude=...) so chart labels
    are not indexed as narrative text; "image_path" is what gets sent to the VLM.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    page_no = page.number + 1
    visuals = []

    for i, (kind, rect) in enumerate(find_visual_regions(page)):
        image_path = out_dir / f"{doc_id}_p{page_no}_img{i}.png"
        page.get_pixmap(clip=rect, dpi=RENDER_DPI).save(image_path)
        visuals.append({
            "kind": kind,
            "rect": rect,
            "page": page_no,
            "index": i,
            "image_path": str(image_path),
        })

    return visuals


def image_to_data_uri(image_path: str) -> str:
    """Encode an image as a base64 JPEG data URI ("data:image/jpeg;base64,...").

    This is the inline image format accepted by OpenAI-compatible vision APIs.
    JPEG keeps the payload much smaller than PNG.
    """
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


if __name__ == "__main__":
    # Extract visuals from every PDF in the raw data folder.
    from mmrag.config import get_settings
    from mmrag.ingestion.pdf_parser import doc_metadata

    settings = get_settings()
    for pdf_path in sorted(settings.raw_data_dir.glob("*.pdf")):
        with pymupdf.open(pdf_path) as pdf:
            doc_id = doc_metadata(pdf, pdf_path)["doc_id"]
            for page in pdf:
                for v in extract_visuals(page, doc_id, settings.extracted_assets_dir):
                    print(f"{v['kind']:7} page {page.number + 1}  {Path(v['image_path']).name}")
