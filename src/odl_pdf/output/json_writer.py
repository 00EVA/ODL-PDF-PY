# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of the Java JSON writer (``json/JsonWriter.java``,
# ``json/serializers/*Serializer.java``, ``json/JsonName.java``).
# Field names, field ordering, and the ``DoubleSerializer`` rounding rule are
# reconstructed from the upstream opendataloader-pdf Java sources
# (Apache-2.0). No veraPDF source was copied.
#
# Conformance rules (§6.1 of 07-output-writers.md):
#   * Field ordering matches the exact ``writeXxxField`` call order in each
#     Java serializer (Jackson writes in insertion order).
#   * ``Double`` rounding: ``BigDecimal(Double.toString(v)).setScale(3, HALF_UP)``
#     — implemented via Python's ``Decimal(str(v)).quantize('0.001', ROUND_HALF_UP)``.
#   * Optional fields (``pdfua_tag``, ``id``, ``level``, ``text color``,
#     ``hidden text``, ``alt``, ...) are ABSENT when their condition is false —
#     never written as ``null``.
#   * Document-level metadata fields (``author``, ``title``, ``creation date``,
#     ``modification date``) are ALWAYS present, even when ``null``.
#   * ``LineArtChunk`` objects are silently skipped everywhere.
"""JSON output writer — produces the same JSON envelope as the Java JAR oracle."""

from __future__ import annotations

import json
from collections import OrderedDict
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from odl_pdf.entities.chunk import ImageChunk, LineArtChunk, TextChunk
from odl_pdf.entities.document import Document
from odl_pdf.entities.list_entity import ListItem, PDFList
from odl_pdf.entities.object import IObject
from odl_pdf.entities.semantic import SemanticTextNode, SemanticType
from odl_pdf.entities.table import TableBorder, TableBorderCell, TableBorderRow
from odl_pdf.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Double rounding — replicate Java's BigDecimal(Double.toString(v)).setScale(3, HALF_UP)
# ---------------------------------------------------------------------------

def round_double(value: float) -> float:
    """Round a float to 3 decimal places using HALF_UP on the shortest string repr.

    Replicates Java's ``DoubleSerializer``:
      ``BigDecimal(Double.toString(value)).setScale(3, RoundingMode.HALF_UP).doubleValue()``

    Using ``str(value)`` first applies Python's shortest-round-trip repr (same
    semantics as Java's ``Double.toString``), then ``Decimal.quantize`` with
    ``ROUND_HALF_UP`` gives the exact tie-breaking the Java oracle expects.
    """
    d = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return float(d)


# ---------------------------------------------------------------------------
# pdfua_tag mapping (mirrors SerializerUtil.pdfuaTagFor)
# ---------------------------------------------------------------------------

def _pdfua_tag(type_str: str, heading_level: int | None = None) -> str | None:
    """Return the PDF/UA structure tag for the given JSON type string, or None."""
    if type_str == "heading":
        if heading_level is not None and 1 <= heading_level <= 6:
            return f"H{heading_level}"
        return "H"
    return {
        "paragraph": "P",
        "image": "Figure",
        "formula": "Formula",
        "list": "L",
        "list item": "LI",
        "table": "Table",
        "table cell": "TD",
    }.get(type_str)


# ---------------------------------------------------------------------------
# text_color serialization  (Arrays.toString(double[]) format)
# ---------------------------------------------------------------------------

def _text_color_string(color: list[float]) -> str:
    """Serialize a color array to Java's Arrays.toString format: ``[v]`` or ``[r, g, b]``.

    Java's ``Arrays.toString(double[])`` produces ``[0.0]`` for a one-element
    array and ``[0.1, 0.2, 0.3]`` for multi-element arrays (single space after
    each comma, no trailing space).
    """
    parts = ", ".join(str(float(v)) for v in color)
    return f"[{parts}]"


# ---------------------------------------------------------------------------
# Font/color extraction helpers (SemanticTextNode has no direct font fields)
# ---------------------------------------------------------------------------

def _node_font_name(node: SemanticTextNode) -> str:
    """Font name from the first text chunk of the node, or ''."""
    for col in node.columns:
        for block in col.blocks:
            for line in block.lines:
                if line.chunks:
                    return line.chunks[0].font_name
    return ""


def _node_font_size(node: SemanticTextNode) -> float:
    """Font size from the first text chunk, or 0.0."""
    for col in node.columns:
        for block in col.blocks:
            for line in block.lines:
                if line.chunks:
                    return line.chunks[0].font_size
    return 0.0


def _node_text_color(node: SemanticTextNode) -> list[float] | None:
    """Text color list from the first text chunk, or None when absent/empty."""
    for col in node.columns:
        for block in col.blocks:
            for line in block.lines:
                if line.chunks:
                    color = line.chunks[0].text_color
                    if color:  # non-empty list is present
                        return list(color)
    return None


def _node_is_hidden_text(node: SemanticTextNode) -> bool:
    """Whether all chunks of the node are hidden text."""
    has_chunk = False
    for col in node.columns:
        for block in col.blocks:
            for line in block.lines:
                for chunk in line.chunks:
                    has_chunk = True
                    if not chunk.hidden_text:
                        return False
    return has_chunk


# ---------------------------------------------------------------------------
# Base element dict builder (SerializerUtil.writeEssentialInfo)
# ---------------------------------------------------------------------------

def _essential_info(
    type_str: str,
    obj: IObject,
    *,
    heading_level: int | None = None,
    recognized_structure_id: int | None = None,
    level: str | None = None,
    page_number: int = 0,
) -> OrderedDict:
    """Emit the base fields in the exact Java field order:
      type → pdfua_tag? → id? → level? → page number → bounding box
    """
    d: OrderedDict = OrderedDict()
    d["type"] = type_str
    tag = _pdfua_tag(type_str, heading_level)
    if tag is not None:
        d["pdfua_tag"] = tag
    if recognized_structure_id is not None and recognized_structure_id != 0:
        d["id"] = recognized_structure_id
    if level is not None:
        d["level"] = level
    d["page number"] = page_number + 1  # 1-based
    bbox = obj.bounding_box
    if bbox is not None:
        d["bounding box"] = [
            round_double(bbox.left),
            round_double(bbox.bottom),
            round_double(bbox.right),
            round_double(bbox.top),
        ]
    else:
        d["bounding box"] = [0.0, 0.0, 0.0, 0.0]
    return d


def _text_info_into(d: OrderedDict, node: SemanticTextNode) -> None:
    """Append text fields (SerializerUtil.writeTextInfo) in order:
      font → font size → text color? → content → hidden text?
    """
    d["font"] = _node_font_name(node)
    d["font size"] = round_double(_node_font_size(node))
    color = _node_text_color(node)
    if color is not None:
        d["text color"] = _text_color_string(color)
    d["content"] = node.value
    if _node_is_hidden_text(node):
        d["hidden text"] = True


# ---------------------------------------------------------------------------
# Per-type serializers
# ---------------------------------------------------------------------------

def _serialize_semantic_node(node: SemanticTextNode) -> OrderedDict:
    """SemanticTextNodeSerializer / HeadingSerializer."""
    stype = node.semantic_type
    type_str = stype.value.lower()

    # Headings use the specialized HeadingSerializer path
    is_heading = (stype is SemanticType.HEADING)

    # Page number from the first chunk bounding box
    page_num = 0
    bbox = node.bounding_box
    if bbox is not None:
        page_num = bbox.page_number

    d = _essential_info(
        type_str,
        node,
        heading_level=node.heading_level if is_heading else None,
        recognized_structure_id=node.recognized_structure_id,
        level=node.level,
        page_number=page_num,
    )

    if is_heading:
        # heading level inserted between base fields and text fields
        d["heading level"] = node.heading_level if node.heading_level is not None else 0

    _text_info_into(d, node)
    return d


def _serialize_image(img: ImageChunk) -> OrderedDict:
    """ImageSerializer — bare ImageChunk (no EnrichedImageChunk)."""
    page_num = img.bounding_box.page_number if img.bounding_box else 0
    d = _essential_info("image", img, page_number=page_num)
    # No alt text for a bare ImageChunk → alt_source = "missing"
    d["alt_source"] = "missing"
    # No data or source in non-embedded/non-external mode (default)
    return d


def _serialize_text_chunk(chunk: TextChunk) -> OrderedDict:
    """TextChunkSerializer — type="text chunk", content only."""
    page_num = chunk.bounding_box.page_number if chunk.bounding_box else 0
    d = _essential_info("text chunk", chunk, page_number=page_num)
    d["content"] = chunk.value
    return d


def _serialize_object(obj: IObject) -> OrderedDict | None:
    """Dispatch an IObject to its serializer; return None to skip (LineArt)."""
    if isinstance(obj, LineArtChunk):
        logger.debug("Skipping LineArtChunk (filtered per Java JsonWriter)")
        return None

    if isinstance(obj, SemanticTextNode):
        return _serialize_semantic_node(obj)

    if isinstance(obj, ImageChunk):
        return _serialize_image(obj)

    if isinstance(obj, TextChunk):
        return _serialize_text_chunk(obj)

    if isinstance(obj, TableBorder):
        return _serialize_table(obj)

    if isinstance(obj, TableBorderCell):
        return _serialize_table_cell(obj)

    if isinstance(obj, TableBorderRow):
        return _serialize_table_row(obj)

    if isinstance(obj, PDFList):
        return _serialize_list(obj)

    if isinstance(obj, ListItem):
        return _serialize_list_item(obj)

    logger.warning("Unknown IObject type %s — skipping", type(obj).__name__)
    return None


# ---------------------------------------------------------------------------
# Table serializer
# ---------------------------------------------------------------------------

def _serialize_table(table: TableBorder) -> OrderedDict:
    """TableSerializer — emits 'text block' for 1×1, 'table' otherwise."""
    page_num = 0
    if table.bounding_box is not None:
        page_num = table.bounding_box.page_number

    is_text_block = table.is_text_block

    type_str = "text block" if is_text_block else "table"
    d = _essential_info(
        type_str,
        table,
        recognized_structure_id=getattr(table, "recognized_structure_id", None),
        page_number=page_num,
    )

    if is_text_block:
        # Only "kids" — the contents of cell(0,0), non-LineArt
        cell = table.cell(0, 0)
        kids: list[Any] = []
        if cell is not None:
            for content in cell.contents:
                if isinstance(content, LineArtChunk):
                    logger.debug("Skipping LineArtChunk inside text block cell")
                    continue
                serialized = _serialize_object(content)
                if serialized is not None:
                    kids.append(serialized)
        d["kids"] = kids
        logger.debug("text block: %d kids", len(kids))
    else:
        d["number of rows"] = table.number_of_rows
        d["number of columns"] = table.number_of_columns
        if getattr(table, "previous_table_id", None) is not None:
            d["previous table id"] = table.previous_table_id
        if getattr(table, "next_table_id", None) is not None:
            d["next table id"] = table.next_table_id
        rows_json: list[Any] = []
        for row in table.rows:
            rows_json.append(_serialize_table_row(row))
        d["rows"] = rows_json
        logger.debug("table: %d rows x %d cols", table.number_of_rows, table.number_of_columns)

    return d


def _serialize_table_row(row: TableBorderRow) -> OrderedDict:
    """TableRowSerializer — no base fields; just type/row number/cells."""
    d: OrderedDict = OrderedDict()
    d["type"] = "table row"
    d["row number"] = row.row_number + 1  # 1-based

    cells_json: list[Any] = []
    for col_idx, cell in enumerate(row.cells):
        # Only write origin cells (matches Java: cell.getColNumber() == columnNumber)
        if cell.col_number == col_idx and cell.row_number == row.row_number:
            cells_json.append(_serialize_table_cell(cell))
    d["cells"] = cells_json
    logger.debug("table row %d: %d cells emitted", row.row_number + 1, len(cells_json))
    return d


def _serialize_table_cell(cell: TableBorderCell) -> OrderedDict:
    """TableCellSerializer — base fields + row/col numbers + span + kids."""
    page_num = 0
    if cell.bounding_box is not None:
        page_num = cell.bounding_box.page_number

    d = _essential_info("table cell", cell, page_number=page_num)
    d["row number"] = cell.row_number + 1    # 1-based
    d["column number"] = cell.col_number + 1  # 1-based
    d["row span"] = cell.row_span
    d["column span"] = cell.col_span

    kids: list[Any] = []
    for content in cell.contents:
        if isinstance(content, LineArtChunk):
            logger.debug("Skipping LineArtChunk inside table cell")
            continue
        serialized = _serialize_object(content)
        if serialized is not None:
            kids.append(serialized)
    d["kids"] = kids
    return d


# ---------------------------------------------------------------------------
# List serializer
# ---------------------------------------------------------------------------

def _serialize_list(pdf_list: PDFList) -> OrderedDict:
    """ListSerializer."""
    page_num = 0
    if pdf_list.bounding_box is not None:
        page_num = pdf_list.bounding_box.page_number

    d = _essential_info(
        "list",
        pdf_list,
        recognized_structure_id=getattr(pdf_list, "recognized_structure_id", None),
        page_number=page_num,
    )
    d["numbering style"] = pdf_list.numbering_style or ""
    d["number of list items"] = pdf_list.number_of_items

    if getattr(pdf_list, "previous_list_id", None) is not None:
        d["previous list id"] = pdf_list.previous_list_id
    if getattr(pdf_list, "next_list_id", None) is not None:
        d["next list id"] = pdf_list.next_list_id

    items_json: list[Any] = []
    for item in pdf_list.items:
        items_json.append(_serialize_list_item(item))
    d["list items"] = items_json
    logger.debug("list: %d items, style=%r", pdf_list.number_of_items, pdf_list.numbering_style)
    return d


def _serialize_list_item(item: ListItem) -> OrderedDict:
    """ListItemSerializer — base fields + font/size/color/content (from first chunk) + kids."""
    page_num = 0
    if item.bounding_box is not None:
        page_num = item.bounding_box.page_number

    d = _essential_info("list item", item, page_number=page_num)

    # Font info comes from item.getFirstLine().getFirstTextChunk() in Java.
    # In Python, walk the contents to find the first TextChunk.
    first_chunk = _first_text_chunk_in_contents(item.contents)
    if first_chunk is not None:
        d["font"] = first_chunk.font_name
        d["font size"] = round_double(first_chunk.font_size)
        if first_chunk.text_color:
            d["text color"] = _text_color_string(list(first_chunk.text_color))
    else:
        d["font"] = ""
        d["font size"] = 0.0

    # content = item.toString() — the text of the item
    parts: list[str] = []
    for content in item.contents:
        if isinstance(content, SemanticTextNode):
            parts.append(content.value)
        elif isinstance(content, TextChunk):
            parts.append(content.value)
    d["content"] = "\n".join(parts)

    kids: list[Any] = []
    for content in item.contents:
        if isinstance(content, LineArtChunk):
            logger.debug("Skipping LineArtChunk inside list item")
            continue
        serialized = _serialize_object(content)
        if serialized is not None:
            kids.append(serialized)
    d["kids"] = kids
    return d


def _first_text_chunk_in_contents(contents: list[IObject]) -> TextChunk | None:
    """Walk a content list and return the first TextChunk found."""
    for obj in contents:
        if isinstance(obj, TextChunk):
            return obj
        if isinstance(obj, SemanticTextNode):
            for col in obj.columns:
                for block in col.blocks:
                    for line in block.lines:
                        if line.chunks:
                            return line.chunks[0]
    return None


# ---------------------------------------------------------------------------
# Page kids extraction
# ---------------------------------------------------------------------------

def _kids_for_page(page: object) -> list[IObject]:
    """Extract the IObject list from a page.

    The pipeline attaches a ``_kids`` attribute to pages after processing.
    Fall back to an empty list when no objects have been attached yet.
    """
    return getattr(page, "_kids", [])


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def write_document_json(document: Document) -> str:
    """Serialize a ``Document`` to a JSON string matching the Java oracle.

    Implements ``JsonWriter.writeToJson`` semantics:
    - Document envelope (file name, number of pages, author, title,
      creation date, modification date, kids).
    - ``"kids"`` = flat list of all page IObjects in page order, each
      serialized per its type.
    - LineArtChunk objects are silently skipped.
    - All ``Double`` values rounded via ``round_double``.

    Args:
        document: The parsed document to serialize.

    Returns:
        A pretty-printed JSON string (indent=2, matching Jackson's
        ``DefaultPrettyPrinter``).
    """
    logger.info(
        "write_document_json: starting — file=%r pages=%d",
        document.metadata.file_name,
        document.number_of_pages,
    )

    envelope: OrderedDict = OrderedDict()
    envelope["file name"] = document.metadata.file_name
    envelope["number of pages"] = document.number_of_pages
    envelope["author"] = document.metadata.author
    envelope["title"] = document.metadata.title
    envelope["creation date"] = document.metadata.creation_date
    envelope["modification date"] = document.metadata.modification_date

    # Collect all kids across all pages in document (page) order
    all_kids: list[Any] = []
    for page in document.pages:
        page_kids = _kids_for_page(page)
        logger.debug(
            "write_document_json: page %d — %d IObjects",
            page.page_number,
            len(page_kids),
        )
        dropped = 0
        for obj in page_kids:
            if isinstance(obj, LineArtChunk):
                dropped += 1
                logger.debug(
                    "Skipping top-level LineArtChunk on page %d", page.page_number
                )
                continue
            serialized = _serialize_object(obj)
            if serialized is not None:
                all_kids.append(serialized)
            else:
                dropped += 1
                logger.warning(
                    "write_document_json: failed to serialize object %s on page %d — dropped",
                    type(obj).__name__,
                    page.page_number,
                )
        if dropped:
            logger.warning(
                "write_document_json: page %d dropped %d objects",
                page.page_number,
                dropped,
            )

    envelope["kids"] = all_kids

    total_kids = len(all_kids)
    logger.info(
        "write_document_json: done — %d kids across %d pages",
        total_kids,
        document.number_of_pages,
    )

    return json.dumps(envelope, indent=2, ensure_ascii=False)
