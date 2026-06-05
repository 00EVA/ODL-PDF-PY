# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""JSON → Markdown adapter.

Renders the document JSON tree (the same structure
:func:`odl_pdf.output.json_writer.write_document_json` produces) into clean
Markdown suitable for LLM/RAG ingestion — a drop-in replacement for
pymupdf4llm's ``to_markdown``, but driven by *structured* extraction:

- headings → ``#``..``######`` from the real heading level (not font-size guesses)
- paragraphs / captions → text joined with blank lines
- lists → ``-`` (unordered) or ``1.`` (ordered) from the detected numbering style
- tables → GitHub pipe tables, first row as header (matches the TH detection)
- figures → ``![alt](source)`` with the alt text when present

Because reading order is already applied upstream in the pipeline, the emitted
Markdown is in human reading order with no extra work here.
"""

from __future__ import annotations

from odl_pdf.entities import Document
from odl_pdf.logging_config import get_logger
from odl_pdf.output.json_writer import write_document_json
import json

logger = get_logger(__name__)

_HEADING_HASHES = {1: "#", 2: "##", 3: "###", 4: "####", 5: "#####", 6: "######"}


def document_to_markdown(document: Document, page_separator: str | None = None) -> str:
    """Render a :class:`Document` to Markdown.

    ``page_separator`` (e.g. ``"\\n---\\n"``) is inserted between pages when set;
    ``%page-number%`` in it is replaced with the 1-indexed page number.
    """
    tree = json.loads(write_document_json(document))
    return _tree_to_markdown(tree, page_separator)


def _tree_to_markdown(tree: dict, page_separator: str | None = None) -> str:
    kids = tree.get("kids", [])
    logger.info("markdown: rendering %d top-level element(s)", len(kids))
    blocks: list[str] = []
    last_page: int | None = None
    for el in kids:
        page = el.get("page number")
        if page_separator and last_page is not None and page != last_page:
            blocks.append(page_separator.replace("%page-number%", str(page or "")))
        last_page = page
        md = _element_to_markdown(el)
        if md:
            blocks.append(md)
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _element_to_markdown(el: dict) -> str:
    etype = el.get("type")
    if etype == "heading":
        level = int(el.get("heading level", 1))
        hashes = _HEADING_HASHES.get(level, "######")
        return f"{hashes} {_clean(el.get('content', ''))}"
    if etype in ("paragraph", "caption", "text block"):
        return _clean(el.get("content", ""))
    if etype == "list":
        return _list_to_markdown(el)
    if etype == "table":
        return _table_to_markdown(el)
    if etype in ("image", "figure", "picture"):
        return _figure_to_markdown(el)
    if etype in ("header", "footer"):
        # Header/footer are filtered from the default output; if present, render
        # their kids inline (rare — only with --include-header-footer upstream).
        return "\n\n".join(
            _element_to_markdown(k) for k in el.get("kids", []) if _element_to_markdown(k)
        )
    # Unknown type: fall back to its content if any.
    content = el.get("content")
    if content:
        logger.debug("markdown: unknown type %r rendered as plain text", etype)
        return _clean(content)
    return ""


def _list_to_markdown(el: dict) -> str:
    style = (el.get("numbering style") or "").lower()
    ordered = any(tok in style for tok in ("decimal", "number", "letter", "roman", "ordered"))
    lines: list[str] = []
    for i, item in enumerate(el.get("list items", []), start=1):
        # An item's text is the concatenation of its kid contents.
        text = " ".join(
            _clean(k.get("content", "")) for k in item.get("kids", []) if k.get("content")
        ).strip()
        if not text:
            text = _clean(item.get("content", ""))
        marker = f"{i}." if ordered else "-"
        lines.append(f"{marker} {text}")
    return "\n".join(lines)


def _table_to_markdown(el: dict) -> str:
    rows = el.get("rows", [])
    if not rows:
        return ""
    ncols = el.get("number of columns") or max(
        (len(r.get("cells", [])) for r in rows), default=0
    )
    if ncols == 0:
        return ""

    def cell_text(cell: dict) -> str:
        txt = " ".join(
            _clean(k.get("content", "")) for k in cell.get("kids", []) if k.get("content")
        ).strip()
        # Escape pipes so they don't break the table.
        return txt.replace("|", "\\|").replace("\n", " ")

    def row_cells(row: dict) -> list[str]:
        cells = sorted(row.get("cells", []), key=lambda c: c.get("column number", 0))
        out = [cell_text(c) for c in cells]
        # Pad/truncate to the column count.
        out += [""] * (ncols - len(out))
        return out[:ncols]

    lines = []
    header = row_cells(rows[0])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * ncols) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row_cells(row)) + " |")
    return "\n".join(lines)


def _figure_to_markdown(el: dict) -> str:
    alt = _clean(el.get("alt") or el.get("description") or "image")
    src = el.get("source") or el.get("data") or ""
    md = f"![{alt}]({src})"
    desc = el.get("description")
    if desc and desc != alt:
        md += f"\n\n*{_clean(desc)}*"
    return md


def _clean(text: str) -> str:
    """Collapse runaway whitespace while preserving intentional newlines."""
    if not text:
        return ""
    # Normalize spaces within lines; keep paragraph newlines.
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(lines).strip()
