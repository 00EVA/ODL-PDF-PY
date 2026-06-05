# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""End-to-end extraction pipeline: PDF -> ordered semantic document.

Ties the parser and the layout processors into one flow, mirroring the order
of DocumentProcessor.processDocument:

    parse -> group lines -> group paragraphs -> detect headings + levels
          -> detect header/footer -> detect tables (line-art) -> detect lists
          -> reading-order sort -> attach per-page ordered content (`_kids`)

The result Document has, on each Page, a ``_kids`` list of IObjects in reading
order — exactly what :func:`odl_pdf.output.json_writer.write_document_json`
consumes. Heavy logging at each stage for the debugging loop.
"""

from __future__ import annotations

from pathlib import Path

from odl_pdf.entities import Document, IObject, Page, TableBorder
from odl_pdf.logging_config import get_logger
from odl_pdf.parser import parse_pdf
from odl_pdf.processors import grouping, lists, reading_order, tables

logger = get_logger(__name__)


def _bbox_of(obj: IObject) -> object | None:
    """Bounding box of any content object (entities expose ``.bounding_box``)."""
    return getattr(obj, "bounding_box", None)


def assemble_page(page: Page) -> list[IObject]:
    """Turn one parsed page's raw chunks into an ordered list of semantic objects."""
    # 1. bordered tables from line-art; their text chunks are consumed by cells.
    detected_tables = tables.detect_bordered_tables(page.line_art_chunks, page.text_chunks)
    consumed = _chunks_in_tables(detected_tables)
    free_chunks = [c for c in page.text_chunks if id(c) not in consumed]
    logger.debug(
        "page %d: %d table(s), %d/%d chunks free after table assignment",
        page.page_number + 1, len(detected_tables), len(free_chunks), len(page.text_chunks),
    )

    # 2. group free chunks -> lines -> paragraph nodes.
    lines_ = grouping.group_lines(free_chunks)
    nodes = grouping.group_paragraphs(lines_)

    # 3. lists from the grouped lines (label detection), then headings/levels.
    detected_lists = lists.detect_lists(lines_)
    grouping.detect_headings(nodes, _body_font_size(nodes), page_width=page.width)
    grouping.assign_heading_levels(nodes)
    # Header/footer is a cross-page pass (needs all pages' lines + heights);
    # the document-level extract() runs it separately. Skipped here.

    # 4. collect every top-level object and sort into reading order.
    objects: list[IObject] = [*nodes, *detected_tables, *detected_lists]
    ordered = _reading_order(objects)
    logger.debug(
        "page %d assembled: %d nodes, %d tables, %d lists -> %d ordered objects",
        page.page_number + 1, len(nodes), len(detected_tables), len(detected_lists), len(ordered),
    )
    return ordered


def _reading_order(objects: list[IObject]) -> list[IObject]:
    items = []
    indexed = []
    for i, obj in enumerate(objects):
        box = _bbox_of(obj)
        if box is None:
            continue
        items.append((i, box))
        indexed.append(obj)
    if not items:
        return objects
    order = reading_order.sort_reading_order(items)
    # order is a list of the user indices (the 0-based i we passed).
    by_index = {i: obj for i, obj in zip((it[0] for it in items), indexed)}
    return [by_index[i] for i in order if i in by_index]


def _chunks_in_tables(detected_tables: list[TableBorder]) -> set[int]:
    consumed: set[int] = set()
    for table in detected_tables:
        for row in table.rows:
            for cell in row.cells:
                for obj in cell.contents:
                    consumed.add(id(obj))
    return consumed


def _body_font_size(nodes) -> float:
    sizes = []
    for n in nodes:
        for col in n.columns:
            for block in col.blocks:
                for line in block.lines:
                    for c in line.chunks:
                        if c.font_size:
                            sizes.append(c.font_size)
    if not sizes:
        return 0.0
    # Body size = most common (mode); fall back to median.
    sizes.sort()
    return sizes[len(sizes) // 2]


def extract(pdf_path: str | Path) -> Document:
    """Full pipeline: parse then assemble each page's ordered semantic content."""
    pdf_path = Path(pdf_path)
    logger.info("pipeline: extracting %s", pdf_path.name)
    document = parse_pdf(pdf_path)
    for page in document.pages:
        try:
            kids = assemble_page(page)
        except Exception:  # noqa: BLE001 — one page must not abort the document
            logger.warning("pipeline: page %d assembly failed", page.page_number + 1, exc_info=True)
            kids = []
        # The JSON writer reads this attribute.
        page._kids = kids  # type: ignore[attr-defined]
    total = sum(len(getattr(p, "_kids", [])) for p in document.pages)
    logger.info("pipeline: %s -> %d ordered objects across %d page(s)",
                pdf_path.name, total, document.number_of_pages)
    return document
