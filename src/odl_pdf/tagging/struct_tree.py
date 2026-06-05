# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of the structure-tree building logic from
# ``processors/AutoTaggingProcessor.java`` (Phases 1-2 and 5).
# Reconstructed from the upstream opendataloader-pdf Java source
# (Apache-2.0). No veraPDF source copied.
#
# All structure-tree building is done with pikepdf's low-level dictionary/array
# API — pikepdf has no tagged-PDF helper. Each StructElem is an indirect object
# so the cross-reference table is valid for every reader.
"""Structure-tree builders for tagged PDF production.

Provides :func:`build_struct_tree` which:
  * Creates and attaches the /StructTreeRoot + /MarkInfo on the catalog.
  * Builds one root "Document" StructElem.
  * Walks per-page IObject lists and emits the full StructElem subtree
    (headings, paragraphs, tables, lists, figures).
  * Normalizes heading levels across the document (first heading = H1,
    never skip levels going down — PDF/UA-1 §7.4.2).
  * Does NOT inject BDC/EMC marked-content operators into content streams
    (MCIDs). That requires per-glyph StreamInfo which is not yet tracked;
    see auto_tagger.py for the WARNING and known-limitation note.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pikepdf

from odl_pdf.entities import (
    IObject,
    ImageChunk,
    PDFList,
    SemanticTextNode,
    SemanticType,
    TableBorder,
)
from odl_pdf.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Heading-level normalization
# ---------------------------------------------------------------------------


def build_normalized_heading_levels(
    pages_kids: list[list[IObject]],
) -> dict[int, int]:
    """Compute normalized heading levels for every heading in the document.

    Algorithm (mirrors ``AutoTaggingProcessor.buildNormalizedHeadingLevels``):
    - First heading → H1.
    - A heading may not skip levels going down (H1 → H3 becomes H1 → H2).
    - A heading may jump back up freely (H3 → H1 is fine).

    Returns a dict mapping ``id(heading_object)`` → normalized level (1..6).
    Uses object identity (``id()``) exactly as the Java code uses
    ``IdentityHashMap``.
    """
    # Collect headings in document order (page 0 … page N).
    headings: list[tuple[int, SemanticTextNode]] = []  # (original_level, node)
    for kids in pages_kids:
        for obj in kids:
            if isinstance(obj, SemanticTextNode) and obj.semantic_type is SemanticType.HEADING:
                level = obj.heading_level or 1
                headings.append((level, obj))

    result: dict[int, int] = {}
    if not headings:
        return result

    current_norm = 1
    prev_orig = headings[0][0]
    result[id(headings[0][1])] = 1

    for orig, node in headings[1:]:
        if orig > prev_orig:
            # Going deeper — only one step allowed at a time.
            current_norm = min(current_norm + 1, 6)
        elif orig < prev_orig:
            # Going back up — free, but floor at 1.
            delta = prev_orig - orig
            current_norm = max(current_norm - delta, 1)
        # else: same level → keep current_norm
        result[id(node)] = current_norm
        prev_orig = orig

    logger.debug(
        "struct_tree: normalized %d headings; levels=%s",
        len(headings),
        sorted(set(result.values())),
    )
    return result


# ---------------------------------------------------------------------------
# StructElem factory helpers
# ---------------------------------------------------------------------------


def _add_struct_element(
    pdf: pikepdf.Pdf,
    parent: pikepdf.Object,
    tag: str,
    page_obj: pikepdf.Object | None,
    *,
    is_first_kid: bool = False,
) -> pikepdf.Object:
    """Create a new indirect StructElem dict and append it to ``parent /K``.

    Mirrors ``AutoTaggingProcessor.addStructElement``:
    - Sets ``/S``, ``/Type /StructElem``, ``/P``, ``/Pg`` (when page_obj given).
    - Initialises ``/K []`` so children can be appended later.
    - Makes the element an indirect object so it has an xref entry.
    """
    elem = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.StructElem,
            S=pikepdf.Name(f"/{tag}"),
            P=parent,
            K=pikepdf.Array(),
        )
    )
    if page_obj is not None:
        elem["/Pg"] = page_obj

    # Append (or prepend) to parent /K.
    k = parent.get("/K")
    if k is None or not isinstance(k, pikepdf.Array):
        parent["/K"] = pikepdf.Array([elem])
    elif is_first_kid:
        # prepend
        new_k = pikepdf.Array([elem])
        for item in k:
            new_k.append(item)
        parent["/K"] = new_k
    else:
        k.append(elem)

    return elem


def _page_obj(pdf: pikepdf.Pdf, page_index: int | None) -> pikepdf.Object | None:
    """Return the pikepdf indirect page object for the given 0-based index."""
    if page_index is None or page_index >= len(pdf.pages):
        return None
    return pdf.pages[page_index].obj


# ---------------------------------------------------------------------------
# Per-type StructElem builders
# ---------------------------------------------------------------------------


def _emit_heading(
    pdf: pikepdf.Pdf,
    parent: pikepdf.Object,
    node: SemanticTextNode,
    normalized_level: int,
) -> pikepdf.Object:
    """Emit H1…H6 StructElem for a heading node."""
    pg = _page_for_node(pdf, node)
    tag = f"H{normalized_level}"
    elem = _add_struct_element(pdf, parent, tag, pg)
    logger.debug("struct_tree: emitted %s elem", tag)
    return elem


def _emit_paragraph(
    pdf: pikepdf.Pdf,
    parent: pikepdf.Object,
    node: SemanticTextNode,
) -> pikepdf.Object:
    """Emit P StructElem for a paragraph node."""
    pg = _page_for_node(pdf, node)
    elem = _add_struct_element(pdf, parent, "P", pg)
    logger.debug("struct_tree: emitted P elem")
    return elem


def _emit_figure(
    pdf: pikepdf.Pdf,
    parent: pikepdf.Object,
    img: ImageChunk,
    figure_counter: list[int],
) -> pikepdf.Object:
    """Emit Figure StructElem for an image.

    /Alt is required by PDF/UA-1 rule 7.3-1. We synthesize "image N" when
    no description is available (same fallback as the Java implementation).
    """
    bb = img.bounding_box
    pg = _page_for_bbox(pdf, bb)
    figure_counter[0] += 1
    alt_text = f"image {figure_counter[0]}"
    elem = _add_struct_element(pdf, parent, "Figure", pg)
    elem["/Alt"] = pikepdf.String(alt_text)
    if bb is not None:
        elem["/A"] = pikepdf.Dictionary(
            O=pikepdf.Name.Layout,
            BBox=pikepdf.Array([bb.left, bb.bottom, bb.right, bb.top]),
        )
    logger.debug("struct_tree: emitted Figure elem alt=%r", alt_text)
    return elem


def _emit_list(
    pdf: pikepdf.Pdf,
    parent: pikepdf.Object,
    pdf_list: PDFList,
) -> pikepdf.Object:
    """Emit L > LI > [LBL] + LBody StructElems for a list."""
    bb = pdf_list.bounding_box
    pg = _page_for_bbox(pdf, bb)
    list_elem = _add_struct_element(pdf, parent, "L", pg)

    for item in pdf_list.items:
        item_bb = item.bounding_box
        item_pg = _page_for_bbox(pdf, item_bb)
        li_elem = _add_struct_element(pdf, list_elem, "LI", item_pg)

        # LBL (bullet/number label) — emit when there is a label.
        # ListItem.contents in the Python model is an IObject list; label
        # extraction at the item level is not yet tracked (the Java model
        # has label_length on ListItem). We emit LBody unconditionally.
        lbody_elem = _add_struct_element(pdf, li_elem, "LBody", item_pg)
        # Recurse into item contents (may contain nested paragraphs/headings).
        for child in item.contents:
            _emit_iobject(pdf, lbody_elem, child, {}, [0])

    logger.debug("struct_tree: emitted L elem with %d items", len(pdf_list.items))
    return list_elem


def _emit_table(
    pdf: pikepdf.Pdf,
    parent: pikepdf.Object,
    table: TableBorder,
) -> pikepdf.Object:
    """Emit Table > TR > TH/TD StructElems.

    First row uses TH + /A [/O /Table /Scope /Column].
    Span attributes (ColSpan, RowSpan) written when > 1.
    No THead/TBody wrappers — intentional (Acrobat+veraPDF compatibility).
    """
    bb = table.bounding_box
    pg = _page_for_bbox(pdf, bb)
    table_elem = _add_struct_element(pdf, parent, "Table", pg)

    for row_idx, row in enumerate(table.rows):
        row_bb = row.bounding_box
        row_pg = _page_for_bbox(pdf, row_bb)
        row_elem = _add_struct_element(pdf, table_elem, "TR", row_pg)
        is_header_row = (row_idx == 0)

        for col_idx, cell in enumerate(row.cells):
            # Only emit origin cells (avoids duplicates for spanned cells).
            # Mirrors Java: cell.getRowNumber()==rowNumber && cell.getColNumber()==colNumber.
            if cell.row_number != row_idx or cell.col_number != col_idx:
                continue
            cell_tag = "TH" if is_header_row else "TD"
            cell_bb = cell.bounding_box
            cell_pg = _page_for_bbox(pdf, cell_bb)
            cell_elem = _add_struct_element(pdf, row_elem, cell_tag, cell_pg)

            if is_header_row:
                cell_elem["/A"] = pikepdf.Dictionary(
                    O=pikepdf.Name.Table,
                    Scope=pikepdf.Name.Column,
                )
            if cell.col_span > 1:
                _add_span_attr(cell_elem, "ColSpan", cell.col_span, is_header_row)
            if cell.row_span > 1:
                _add_span_attr(cell_elem, "RowSpan", cell.row_span, is_header_row)

            # Recurse into cell contents.
            for child in cell.contents:
                _emit_iobject(pdf, cell_elem, child, {}, [0])

    logger.debug(
        "struct_tree: emitted Table elem rows=%d cols=%d",
        table.number_of_rows, table.number_of_columns,
    )
    return table_elem


def _add_span_attr(
    elem: pikepdf.Object,
    attr_name: str,
    value: int,
    has_table_attr: bool,
) -> None:
    """Append a ColSpan or RowSpan entry to the cell's /A attribute dict/array.

    The /A entry may already exist (set for /Scope on TH cells). This helper
    either merges into the existing Table owner dict or creates a new one.
    """
    a = elem.get("/A")
    table_owner = pikepdf.Name.Table
    if a is None:
        elem["/A"] = pikepdf.Dictionary(
            O=table_owner,
            **{attr_name: value},
        )
    elif isinstance(a, pikepdf.Dictionary):
        if a.get("/O") == table_owner:
            a[f"/{attr_name}"] = value
        else:
            # Different owner — promote to array.
            new_dict = pikepdf.Dictionary(O=table_owner, **{attr_name: value})
            elem["/A"] = pikepdf.Array([a, new_dict])
    elif isinstance(a, pikepdf.Array):
        for entry in a:
            if isinstance(entry, pikepdf.Dictionary) and entry.get("/O") == table_owner:
                entry[f"/{attr_name}"] = value
                return
        a.append(pikepdf.Dictionary(O=table_owner, **{attr_name: value}))


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


def _emit_iobject(
    pdf: pikepdf.Pdf,
    parent: pikepdf.Object,
    obj: IObject,
    normalized_levels: dict[int, int],
    figure_counter: list[int],
) -> None:
    """Dispatch one IObject to its StructElem emitter.

    ``normalized_levels`` maps ``id(SemanticTextNode)`` → normalized heading
    level. ``figure_counter`` is a mutable single-element list used as a
    shared counter across recursive calls.

    Failures on individual objects are caught and logged as WARNING — one bad
    object must not abort the whole document.
    """
    try:
        if isinstance(obj, SemanticTextNode):
            stype = obj.semantic_type
            if stype is SemanticType.HEADING:
                norm_level = normalized_levels.get(id(obj)) or (obj.heading_level or 1)
                _emit_heading(pdf, parent, obj, norm_level)
            elif stype is SemanticType.PARAGRAPH:
                _emit_paragraph(pdf, parent, obj)
            elif stype in (SemanticType.HEADER, SemanticType.FOOTER):
                # Skip header/footer — not part of the logical structure.
                pass
            else:
                # Fallback: unknown semantic type → emit as P.
                logger.debug(
                    "struct_tree: unknown SemanticType %s — emitting as P", stype
                )
                _emit_paragraph(pdf, parent, obj)

        elif isinstance(obj, ImageChunk):
            _emit_figure(pdf, parent, obj, figure_counter)

        elif isinstance(obj, PDFList):
            _emit_list(pdf, parent, obj)

        elif isinstance(obj, TableBorder):
            if obj.is_text_block:
                # 1×1 table = text block: emit as Art container.
                bb = obj.bounding_box
                pg = _page_for_bbox(pdf, bb)
                art_elem = _add_struct_element(pdf, parent, "Art", pg)
                cell = obj.cell(0, 0)
                if cell is not None:
                    for child in cell.contents:
                        _emit_iobject(pdf, art_elem, child, normalized_levels, figure_counter)
            else:
                _emit_table(pdf, parent, obj)

        else:
            logger.debug(
                "struct_tree: unrecognised IObject type %s — skipping",
                type(obj).__name__,
            )
    except Exception:
        logger.warning(
            "struct_tree: failed to emit StructElem for %s — skipping",
            type(obj).__name__,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _page_for_node(pdf: pikepdf.Pdf, node: SemanticTextNode) -> pikepdf.Object | None:
    """Page indirect object for a SemanticTextNode (from its bounding box)."""
    bb = node.bounding_box
    return _page_for_bbox(pdf, bb)


def _page_for_bbox(pdf: pikepdf.Pdf, bb: object | None) -> pikepdf.Object | None:
    """Page indirect object for a bounding box (page_number attribute)."""
    if bb is None:
        return None
    pg_idx = getattr(bb, "page_number", None)
    if pg_idx is None or pg_idx >= len(pdf.pages):
        return None
    return pdf.pages[pg_idx].obj


# ---------------------------------------------------------------------------
# Top-level structure-tree builder
# ---------------------------------------------------------------------------


def build_struct_tree(
    pdf: pikepdf.Pdf,
    pages_kids: list[list[IObject]],
) -> pikepdf.Object:
    """Build a full /StructTreeRoot and attach it to the PDF catalog.

    Steps (mirrors ``AutoTaggingProcessor.createStructTreeRoot`` +
    ``createStructureTreeElements``):

    1. Set catalog /MarkInfo <<  /Marked true >>.
    2. Create /StructTreeRoot as indirect object; set on catalog.
    3. Create a "Document" StructElem as single child of StructTreeRoot.
    4. Normalize heading levels across all pages.
    5. For each page, for each IObject in reading order, emit a StructElem.
    6. Build a minimal /ParentTree (empty Nums array — no MCIDs yet).

    Returns the StructTreeRoot indirect object.
    """
    catalog = pdf.trailer["/Root"]

    # Phase 1 — /MarkInfo.
    catalog["/MarkInfo"] = pikepdf.Dictionary(Marked=True)
    logger.info("struct_tree: set /MarkInfo Marked=true")

    # Phase 2 — /StructTreeRoot.
    struct_tree_root = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.StructTreeRoot,
            ParentTreeNextKey=len(pdf.pages),
        )
    )
    catalog["/StructTreeRoot"] = struct_tree_root
    logger.info("struct_tree: created /StructTreeRoot")

    # Phase 3 — root Document StructElem.
    doc_elem = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.StructElem,
            S=pikepdf.Name.Document,
            P=struct_tree_root,
            K=pikepdf.Array(),
        )
    )
    struct_tree_root["/K"] = doc_elem
    logger.info("struct_tree: created Document StructElem")

    # Phase 4 — heading normalization (document-wide, single pass).
    normalized_levels = build_normalized_heading_levels(pages_kids)

    # Phase 5 — emit per-page StructElems.
    figure_counter: list[int] = [0]
    total_elems = 0
    for page_idx, kids in enumerate(pages_kids):
        page_elem_count = 0
        for obj in kids:
            before = _count_k(doc_elem)
            _emit_iobject(pdf, doc_elem, obj, normalized_levels, figure_counter)
            after = _count_k(doc_elem)
            page_elem_count += after - before
        total_elems += page_elem_count
        logger.info(
            "struct_tree: page %d — %d IObjects -> %d StructElems emitted",
            page_idx + 1, len(kids), page_elem_count,
        )

    logger.info(
        "struct_tree: total StructElems under Document: %d", total_elems
    )

    # Phase 6 — minimal /ParentTree (empty for now; MCIDs not yet linked).
    parent_tree = pdf.make_indirect(
        pikepdf.Dictionary(
            Nums=pikepdf.Array(),
        )
    )
    struct_tree_root["/ParentTree"] = parent_tree
    struct_tree_root["/ParentTreeNextKey"] = 0

    return struct_tree_root


def _count_k(elem: pikepdf.Object) -> int:
    """Number of items in the /K array of a StructElem."""
    k = elem.get("/K")
    if k is None:
        return 0
    if isinstance(k, pikepdf.Array):
        return len(k)
    return 1  # single non-array child


# ---------------------------------------------------------------------------
# Tag-tree pretty printer (for smoke-test output)
# ---------------------------------------------------------------------------


def print_struct_tree(
    struct_tree_root: pikepdf.Object,
    *,
    max_depth: int = 8,
    max_children: int = 20,
) -> str:
    """Return a human-readable indented representation of the structure tree.

    Useful for smoke-test verification. Terminates at ``max_depth`` and
    truncates child lists at ``max_children`` per node.
    """
    lines: list[str] = []

    def _walk(node: pikepdf.Object, depth: int, label: str) -> None:
        if depth > max_depth:
            lines.append("  " * depth + "...")
            return
        # Determine the tag name.
        s_val = node.get("/S")
        tag = str(s_val) if s_val is not None else "(no /S)"
        pg_val = node.get("/Pg")
        pg_info = f" pg={pg_val.objgen[0]}" if pg_val is not None else ""
        lines.append("  " * depth + f"{label}{tag}{pg_info}")

        k = node.get("/K")
        if k is None:
            return
        if not isinstance(k, pikepdf.Array):
            # Single child (direct or indirect).
            try:
                _walk(k, depth + 1, "")
            except Exception:
                lines.append("  " * (depth + 1) + "<unreadable child>")
            return
        children = list(k)
        if len(children) > max_children:
            shown = children[:max_children]
            truncated = len(children) - max_children
        else:
            shown = children
            truncated = 0
        for child in shown:
            try:
                if isinstance(child, pikepdf.Object) and child.get("/S") is not None:
                    _walk(child, depth + 1, "")
                else:
                    # MCID integer or MCR dict.
                    lines.append("  " * (depth + 1) + f"<MCID/ref: {child!r}>")
            except Exception:
                lines.append("  " * (depth + 1) + "<unreadable>")
        if truncated:
            lines.append("  " * (depth + 1) + f"... ({truncated} more)")

    try:
        lines.append("StructTreeRoot")
        k = struct_tree_root.get("/K")
        if k is not None:
            if not isinstance(k, pikepdf.Array):
                _walk(k, 1, "")
            else:
                for child in k:
                    _walk(child, 1, "")
    except Exception as exc:
        lines.append(f"<error walking tree: {exc}>")

    return "\n".join(lines)
