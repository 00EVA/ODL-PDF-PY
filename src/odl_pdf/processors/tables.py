# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of Strategy-1 (bordered) table detection from
# ``TableBorderProcessor.java`` and ``LinesPreprocessingConsumer``.
# Behavior is reconstructed from the architecture doc
# ``05-tables-lists.md`` §3 and from the accessor/constant table in §7.
# No veraPDF source was copied.
#
# Strategy 2 (cluster/borderless) is implemented as ``detect_cluster_tables``
# below, reconstructed from §4 of ``05-tables-lists.md``.  It is intentionally
# NOT wired into the default pipeline; it is kept opt-in so it cannot regress
# the 95–99% recall of the bordered path + frame-rejection pass.
# Strategy 3 (Korean) and ``TableStructureNormalizer`` remain out-of-scope.
"""Strategy-1 bordered + Strategy-2 cluster/borderless table detection.

:func:`detect_bordered_tables` scans a list of :class:`LineArtChunk` objects
for horizontal and vertical rules, infers a row/column grid from the rule
intersections, and assigns :class:`TextChunk` objects to cells by geometric
containment.

:func:`detect_cluster_tables` is Strategy 2 — it detects borderless tables
from text-alignment patterns only (no ruling lines required). See §4 of
``05-tables-lists.md``. This function is *not* wired into the default pipeline;
it is opt-in / additive so it cannot regress the bordered path.

Out of scope (future phases):
  - Strategy 3 — Korean official-document header table detection
  - ``TableStructureNormalizer`` — under-segmentation repair
  - Neighbor-table linking (``checkNeighborTables``)
  - Recursive cell-content processing (nested tables / lists inside cells)
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import LineArtChunk, TextChunk
from odl_pdf.entities.table import TableBorder, TableBorderCell, TableBorderRow
from odl_pdf.logging_config import get_logger

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (from TableBorderProcessor.java and architecture doc §9)
# ---------------------------------------------------------------------------

# A line-art segment is treated as a grid rule when its width-to-height ratio
# indicates it is nearly 1-dimensional (a stroke rather than a filled region).
# We use an aspect ratio threshold: one dimension ≤ RULE_THICKNESS_RATIO * the
# other dimension means it is a rule.
_RULE_THICKNESS_RATIO: float = 0.15  # a rule is ≤15% as thick as it is long

# Tolerance in PDF points for snapping rule endpoints to the same coordinate.
_SNAP_EPSILON: float = 1.5

# Fraction of a cell's width/height that the rule must span to count as a
# full-length separator (avoids short tick marks being misread as full rules).
# Not used for external border segments — only for interior segments.
_MIN_INTERIOR_SPAN_RATIO: float = 0.5

# Minimum number of interior rules needed on each axis to infer a grid.
# (An outer border has 2 H-rules and 2 V-rules but 0 interior ones →
# that's a 1×1 table, which we still detect but flag as is_text_block.)
_MIN_RULES_PER_AXIS: int = 2  # need at least top + bottom (or left + right)


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _is_horizontal(chunk: LineArtChunk) -> bool:
    """True when the chunk is a nearly-horizontal stroke (wider than tall)."""
    w = chunk.bounding_box.width
    h = chunk.bounding_box.height
    if w <= 0 and h <= 0:
        return False
    return h <= _RULE_THICKNESS_RATIO * max(w, _SNAP_EPSILON)


def _is_vertical(chunk: LineArtChunk) -> bool:
    """True when the chunk is a nearly-vertical stroke (taller than wide)."""
    w = chunk.bounding_box.width
    h = chunk.bounding_box.height
    if w <= 0 and h <= 0:
        return False
    return w <= _RULE_THICKNESS_RATIO * max(h, _SNAP_EPSILON)


def _snap(value: float, pool: list[float]) -> float:
    """Snap *value* to the nearest entry in *pool* within ``_SNAP_EPSILON``."""
    best = value
    best_dist = float("inf")
    for v in pool:
        d = abs(v - value)
        if d < best_dist:
            best_dist = d
            best = v
    return best if best_dist <= _SNAP_EPSILON else value


def _cluster_coords(values: list[float]) -> list[float]:
    """Collapse near-duplicate coordinate values into a sorted, unique list.

    Input values within ``_SNAP_EPSILON`` of each other are merged to their
    mean.  Returns the resulting sorted coordinate list.
    """
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters: list[list[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= _SNAP_EPSILON:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _overlaps_1d(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """True when the two 1-D intervals share any interior point."""
    return a_lo < b_hi and b_lo < a_hi


def _center_in_interval(lo: float, hi: float, v: float) -> bool:
    """True when *v* lies strictly inside [lo, hi]."""
    return lo <= v <= hi


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

def _extract_rules(
    line_art: Sequence[LineArtChunk],
) -> tuple[list[LineArtChunk], list[LineArtChunk]]:
    """Partition non-background line art into (h_rules, v_rules).

    Background chunks are silently dropped.  Chunks that are neither
    horizontal nor vertical (diagonal strokes, thick fills) are also dropped
    with a DEBUG log.
    """
    h_rules: list[LineArtChunk] = []
    v_rules: list[LineArtChunk] = []
    skipped = 0
    for chunk in line_art:
        if chunk.is_background:
            logger.debug("skip background LineArtChunk bbox=%s", chunk.bounding_box.to_list())
            skipped += 1
            continue
        if _is_horizontal(chunk):
            h_rules.append(chunk)
        elif _is_vertical(chunk):
            v_rules.append(chunk)
        else:
            logger.debug(
                "skip non-rule LineArtChunk (not H or V) bbox=%s",
                chunk.bounding_box.to_list(),
            )
            skipped += 1
    logger.debug(
        "extract_rules: %d H-rules, %d V-rules, %d skipped",
        len(h_rules),
        len(v_rules),
        skipped,
    )
    return h_rules, v_rules


# ---------------------------------------------------------------------------
# Grid inference
# ---------------------------------------------------------------------------

def _infer_grid(
    h_rules: list[LineArtChunk],
    v_rules: list[LineArtChunk],
) -> tuple[list[float], list[float]] | None:
    """Infer sorted Y-separators and X-separators from the rule sets.

    Returns ``(y_seps, x_seps)`` where:
    - ``y_seps`` is a sorted list of Y-coordinates where H-rules lie
      (length = number_of_rows + 1).
    - ``x_seps`` is a sorted list of X-coordinates where V-rules lie
      (length = number_of_columns + 1).

    Returns ``None`` when the rules do not form a usable grid (fewer than
    two H-rules or two V-rules after clustering).
    """
    if len(h_rules) < _MIN_RULES_PER_AXIS or len(v_rules) < _MIN_RULES_PER_AXIS:
        logger.debug(
            "infer_grid: too few rules (H=%d, V=%d), need %d each",
            len(h_rules),
            len(v_rules),
            _MIN_RULES_PER_AXIS,
        )
        return None

    # Collect representative Y-coordinates from H-rules (use center Y).
    y_raw = [(r.bounding_box.bottom + r.bounding_box.top) / 2.0 for r in h_rules]
    # Collect representative X-coordinates from V-rules (use center X).
    x_raw = [(r.bounding_box.left + r.bounding_box.right) / 2.0 for r in v_rules]

    y_seps = _cluster_coords(y_raw)
    x_seps = _cluster_coords(x_raw)

    if len(y_seps) < _MIN_RULES_PER_AXIS or len(x_seps) < _MIN_RULES_PER_AXIS:
        logger.debug(
            "infer_grid: after clustering still too few separators (Y=%d, X=%d)",
            len(y_seps),
            len(x_seps),
        )
        return None

    # Sort: Y descending (top row first, y-up coord system means larger Y = higher)
    y_seps_sorted = sorted(y_seps, reverse=True)
    x_seps_sorted = sorted(x_seps)

    logger.debug(
        "infer_grid: %d Y-seps=%s, %d X-seps=%s",
        len(y_seps_sorted),
        [round(v, 1) for v in y_seps_sorted],
        len(x_seps_sorted),
        [round(v, 1) for v in x_seps_sorted],
    )
    return y_seps_sorted, x_seps_sorted


# ---------------------------------------------------------------------------
# Cell-boundary helpers
# ---------------------------------------------------------------------------

def _cell_bbox(
    page: int,
    y_seps: list[float],
    x_seps: list[float],
    row: int,
    col: int,
) -> BoundingBox:
    """Bounding box for grid cell (row, col).

    With y_seps sorted descending (y-up), cell row 0 spans
    ``[y_seps[1], y_seps[0]]`` (i.e., bottom=y_seps[1], top=y_seps[0]).
    """
    top = y_seps[row]
    bottom = y_seps[row + 1]
    left = x_seps[col]
    right = x_seps[col + 1]
    return BoundingBox.of(page, left, bottom, right, top)


def _chunk_center(chunk: TextChunk) -> tuple[float, float]:
    """(center_x, center_y) of a TextChunk."""
    bb = chunk.bounding_box
    cx = (bb.left + bb.right) / 2.0
    cy = (bb.bottom + bb.top) / 2.0
    return cx, cy


def _find_row_by_overlap(
    chunk_bottom: float,
    chunk_top: float,
    y_seps: list[float],
) -> int | None:
    """Return the row index with the most vertical overlap with [chunk_bottom, chunk_top].

    Strategy (§3.2, "best overlap / centroid-row"):
    1. Measure the 1-D overlap of the chunk's Y span with each row band's Y span.
    2. Return the row with the largest *positive* overlap (tie → smallest row index).
    3. If no row has positive overlap, fall back to the row whose mid-Y is closest
       to the chunk's center Y (nearest-row fallback for out-of-grid chunks).

    Returns ``None`` only when ``y_seps`` is empty (degenerate grid).
    """
    n_rows = len(y_seps) - 1
    if n_rows <= 0:
        return None

    chunk_cy = (chunk_bottom + chunk_top) / 2.0

    # Pass 1: find the row with maximum positive overlap.
    best_overlap_row: int | None = None
    best_overlap = 0.0  # must be > 0 to count

    for r in range(n_rows):
        row_top = y_seps[r]
        row_bot = y_seps[r + 1]
        overlap = min(chunk_top, row_top) - max(chunk_bottom, row_bot)
        if overlap > best_overlap or (overlap == best_overlap and best_overlap_row is None):
            best_overlap = overlap
            best_overlap_row = r

    if best_overlap_row is not None:
        return best_overlap_row

    # Pass 2 (fallback): no positive overlap — assign to nearest row by midpoint distance.
    nearest_row: int | None = None
    nearest_dist = float("inf")
    for r in range(n_rows):
        row_top = y_seps[r]
        row_bot = y_seps[r + 1]
        row_mid = (row_top + row_bot) / 2.0
        dist = abs(chunk_cy - row_mid)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_row = r

    return nearest_row


def _find_col_by_overlap(
    chunk_left: float,
    chunk_right: float,
    x_seps: list[float],
) -> int | None:
    """Return the column index with the most horizontal overlap with [chunk_left, chunk_right].

    Strategy (§3.2, "left-edge-column / max horizontal-overlap-area"):
    1. Measure the 1-D overlap of the chunk's X span with each column's X range.
    2. Return the column with the largest *positive* overlap (tie → smallest col index).
    3. If no column has positive overlap, assign to the column whose left boundary
       is closest to the chunk's left edge (nearest-column fallback).

    Returns ``None`` only when ``x_seps`` is empty (degenerate grid).
    """
    n_cols = len(x_seps) - 1
    if n_cols <= 0:
        return None

    # Pass 1: find the column with maximum positive overlap.
    best_overlap_col: int | None = None
    best_overlap = 0.0  # must be > 0 to count

    for c in range(n_cols):
        col_left = x_seps[c]
        col_right = x_seps[c + 1]
        overlap = min(chunk_right, col_right) - max(chunk_left, col_left)
        if overlap > best_overlap or (overlap == best_overlap and best_overlap_col is None):
            best_overlap = overlap
            best_overlap_col = c

    if best_overlap_col is not None:
        return best_overlap_col

    # Pass 2 (fallback): no positive overlap — assign to column nearest by left edge.
    nearest_col: int | None = None
    nearest_dist = float("inf")
    for c in range(n_cols):
        col_left = x_seps[c]
        dist = abs(chunk_left - col_left)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_col = c

    return nearest_col


def _find_cell_for_chunk(
    chunk: TextChunk,
    y_seps: list[float],
    x_seps: list[float],
) -> tuple[int, int] | None:
    """Return the ``(row, col)`` index that best matches the chunk's position.

    Uses best-overlap assignment (§3.2 of the architecture doc):
    - ROW: determined by maximum vertical overlap between the chunk's Y span and
      each row band; falls back to nearest row midpoint when no positive overlap.
    - COLUMN: determined by maximum horizontal overlap between the chunk's X span
      and each column's X range; falls back to nearest column left-edge.

    This is robust to wide chunks whose bounding box overruns adjacent columns,
    since a chunk is always assigned to exactly one cell (largest overlap wins,
    ties broken by smaller index).

    Returns ``None`` only when the separator lists are empty (degenerate grid).
    """
    bb = chunk.bounding_box
    row_idx = _find_row_by_overlap(bb.bottom, bb.top, y_seps)
    col_idx = _find_col_by_overlap(bb.left, bb.right, x_seps)

    if row_idx is None or col_idx is None:
        return None
    return row_idx, col_idx


# ---------------------------------------------------------------------------
# TableBorder assembly
# ---------------------------------------------------------------------------

def _build_table_border(
    page: int,
    y_seps: list[float],
    x_seps: list[float],
    text_chunks: Sequence[TextChunk],
) -> TableBorder:
    """Assemble a :class:`TableBorder` from the inferred grid and text chunks.

    Assignment strategy (§3.2 of the architecture doc):
    - Each TextChunk is assigned to EXACTLY ONE cell: the row with the best
      vertical overlap and the column with the best horizontal overlap.
    - Chunks that cannot be assigned to any row are dropped with a WARNING.
    """
    n_rows = len(y_seps) - 1
    n_cols = len(x_seps) - 1

    logger.info(
        "build_table_border: page=%d, grid=%dx%d, %d text_chunks",
        page,
        n_rows,
        n_cols,
        len(text_chunks),
    )

    # Build empty grid of cells.
    rows: list[TableBorderRow] = []
    cell_grid: list[list[TableBorderCell]] = []
    for r in range(n_rows):
        row = TableBorderRow(r)
        row_cells: list[TableBorderCell] = []
        for c in range(n_cols):
            cell = TableBorderCell(r, c)
            cell.set_bounding_box(_cell_bbox(page, y_seps, x_seps, r, c))
            row.push(cell)
            row_cells.append(cell)
        rows.append(row)
        cell_grid.append(row_cells)

    # Assign text chunks to cells (primary + secondary columns).
    assigned = 0
    dropped = 0
    for chunk in text_chunks:
        pos = _find_cell_for_chunk(chunk, y_seps, x_seps)
        if pos is None:
            logger.warning(
                "build_table_border: TextChunk outside grid dropped: "
                "value=%r bbox=%s grid_y=%s grid_x=%s",
                chunk.value[:40],
                chunk.bounding_box.to_list(),
                [round(v, 1) for v in y_seps],
                [round(v, 1) for v in x_seps],
            )
            dropped += 1
            continue
        r, primary_c = pos
        # Each chunk is assigned to EXACTLY ONE cell (its best-overlap column).
        #
        # A previous "secondary column" rule copied a wide chunk's *entire text*
        # into every column it overlapped ≥50%. On bordered procedure templates
        # (where a whole line is one wide TextChunk spanning all columns) that
        # smeared the same text across every cell — 3× byte inflation and
        # garbage structure. A chunk that truly embeds multi-column data must be
        # *split* at column boundaries, not duplicated; until that split exists,
        # one-cell assignment is strictly correct (no data is lost — the chunk
        # still appears once, in its dominant column).
        cell_grid[r][primary_c].add_content(chunk)
        logger.debug(
            "build_table_border: chunk %r → cell(%d,%d)", chunk.value[:20], r, primary_c
        )
        assigned += 1

    if dropped:
        logger.warning(
            "build_table_border: %d chunk(s) dropped (outside grid)", dropped
        )

    cells_with_text = sum(
        1 for row in cell_grid for cell in row if cell.contents
    )
    logger.info(
        "build_table_border: assigned %d chunks; %d/%d cells have text",
        assigned,
        cells_with_text,
        n_rows * n_cols,
    )

    # Compute table bounding box from outer separator coordinates.
    table_bbox = BoundingBox.of(
        page,
        x_seps[0],
        y_seps[-1],   # minimum Y (bottom of table, y_seps are desc)
        x_seps[-1],
        y_seps[0],    # maximum Y (top of table)
    )
    table = TableBorder(rows)
    table.set_bounding_box(table_bbox)
    return table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_bordered_tables(
    line_art: Sequence[LineArtChunk],
    text_chunks: Sequence[TextChunk],
) -> list[TableBorder]:
    """Detect Strategy-1 bordered tables on a single page.

    Reconstructs a row/column grid from horizontal and vertical rules in
    *line_art*, then assigns each element of *text_chunks* to the cell whose
    bounding box contains the chunk's centre.

    Parameters
    ----------
    line_art:
        All :class:`LineArtChunk` objects from a single page (background
        chunks are filtered out internally).
    text_chunks:
        All :class:`TextChunk` objects from the same page.

    Returns
    -------
    list[TableBorder]
        Zero or more detected tables.  Currently at most one table per page
        is returned (Strategy-1 assumes the page's rules form a single grid);
        multi-table pages are a future extension.

    Notes
    -----
    - Strategy 2 (cluster/borderless), Strategy 3 (Korean), and
      ``TableStructureNormalizer`` are *not* implemented here.
    - Neighbor-table linking and recursive cell-content processing are also
      deferred to a later phase.
    """
    logger.info(
        "detect_bordered_tables: entry — %d line_art chunks, %d text_chunks",
        len(line_art),
        len(text_chunks),
    )

    if not line_art:
        logger.debug("detect_bordered_tables: no line art → no tables")
        return []

    # Determine the page number from the first line art chunk.
    page = line_art[0].bounding_box.page_number

    h_rules, v_rules = _extract_rules(line_art)

    grid = _infer_grid(h_rules, v_rules)
    if grid is None:
        logger.info(
            "detect_bordered_tables: page=%d — could not infer grid from rules "
            "(H=%d, V=%d)",
            page,
            len(h_rules),
            len(v_rules),
        )
        return []

    y_seps, x_seps = grid
    n_rows = len(y_seps) - 1
    n_cols = len(x_seps) - 1

    logger.info(
        "detect_bordered_tables: page=%d — inferred %d×%d grid",
        page,
        n_rows,
        n_cols,
    )

    # Filter text chunks to those on the same page.
    page_chunks = [c for c in text_chunks if c.bounding_box.page_number == page]
    if len(page_chunks) < len(text_chunks):
        logger.warning(
            "detect_bordered_tables: %d text_chunks on different pages dropped "
            "(only page %d processed)",
            len(text_chunks) - len(page_chunks),
            page,
        )

    # Frame-rejection guard: a page frame / header-box (page outline + a few
    # dividers) clusters into a small grid that then swallows the page's
    # flowing prose into a handful of cells. A *real* data table spreads its
    # text roughly one chunk per cell. Reject when the grid behaves like a
    # frame (high chunks-per-cell AND one cell hoards most of the text), so
    # section headings/paragraphs survive as text instead of becoming cells.
    if _looks_like_prose_frame(page_chunks, y_seps, x_seps):
        logger.info(
            "detect_bordered_tables: page=%d — grid rejected as prose frame "
            "(text dumps into few cells, not tabular); leaving content as text",
            page,
        )
        return []

    table = _build_table_border(page, y_seps, x_seps, page_chunks)

    logger.info(
        "detect_bordered_tables: page=%d — produced 1 table (%d rows × %d cols), "
        "is_text_block=%s",
        page,
        table.number_of_rows,
        table.number_of_columns,
        table.is_text_block,
    )
    return [table]


# Frame-rejection threshold (tuned on technical procedure procedure templates vs the
# Bialetti financial table). A real data table holds roughly one text chunk per
# cell (cell = one datum); the worked corpus shows real tables at ≤1.0
# chunks/cell while procedure-template page frames swallow flowing prose at
# 1.3–10.8 chunks/cell. Reject above this ratio so headings/paragraphs trapped
# inside a page border survive as text instead of becoming table cells.
_FRAME_CHUNKS_PER_CELL: float = 1.2

# A line/chunk that begins with an outline section number ("1 Scope",
# "6.8 Out of Tolerance") — the section-heading shape. Real data tables don't carry
# these; a grid holding several is a numbered procedure page, not a table.
# First number must be >= 1 ([1-9]\d*): outline sections start at 1, so a
# leading "0" (a revision number, list value, etc.) is not a section heading.
_SECTION_HEADING_CHUNK_RE = re.compile(r"^\s*[1-9]\d*(?:\.\d+){0,4}\.?\s+[A-Za-z]")
# A grid with at least this many section-heading chunks is a procedure page.
_FRAME_MIN_SECTION_HEADINGS: int = 2


def _count_section_headings(text_chunks: Sequence[TextChunk]) -> int:
    """Count chunks that start with an outline section number + title word.

    Excludes colon-bearing metadata labels ("8 Date: ...") and over-long chunks
    (body paragraphs that merely open with a number).
    """
    count = 0
    for chunk in text_chunks:
        txt = chunk.value.strip()
        if ":" in txt:
            continue
        if _SECTION_HEADING_CHUNK_RE.match(txt) and len(txt.split()) <= 12:
            count += 1
    return count


def _looks_like_prose_frame(
    text_chunks: Sequence[TextChunk],
    y_seps: list[float],
    x_seps: list[float],
) -> bool:
    """Heuristic: does this grid look like a page frame swallowing prose?

    Two independent signals, either of which rejects the grid as a non-table:
    1. **Density** — substantially more text chunks than cells (a real table is
       ~one datum per cell; procedure frames run far higher).
    2. **Section headings** — the grid contains several outline-numbered section
       headings ("1 Scope", "6.8 ..."), which real data tables never do. This
       catches dense procedure pages whose density looks table-like.
    """
    n_cells = (len(y_seps) - 1) * (len(x_seps) - 1)
    if n_cells <= 0:
        return False

    assigned = sum(
        1 for chunk in text_chunks
        if _find_cell_for_chunk(chunk, y_seps, x_seps) is not None
    )
    if assigned == 0:
        return False

    chunks_per_cell = assigned / n_cells
    section_headings = _count_section_headings(text_chunks)
    is_frame = (
        chunks_per_cell > _FRAME_CHUNKS_PER_CELL
        or section_headings >= _FRAME_MIN_SECTION_HEADINGS
    )
    logger.debug(
        "_looks_like_prose_frame: cells=%d chunks=%d chunks/cell=%.1f "
        "section_headings=%d -> %s",
        n_cells, assigned, chunks_per_cell, section_headings,
        "FRAME" if is_frame else "table",
    )
    return is_frame


# ===========================================================================
# Strategy 2 — Cluster / Borderless Table Detection
# ===========================================================================
#
# Reconstructed from §4 of 05-tables-lists.md and the veraPDF
# ClusterTableProcessor description.  The underlying veraPDF
# ClusterTableConsumer groups tokens by shared row Y-bands and recurring
# column X-positions.  We replicate that spatial logic here with the same
# constants documented in §2 of the architecture doc.
#
# IMPORTANT: This function is NOT wired into the default pipeline.
# The bordered path + frame-rejection already achieves 95–99% recall on the
# document corpus; wiring cluster detection in by default risks regressions on
# heavily-formatted PDFs where paragraph columns would be misread as table
# columns. Keep it opt-in until it can be validated on the full corpus.

# --- Cluster-detection constants (from §2 of 05-tables-lists.md) -----------

# Two text chunks share a baseline when their center-Y values are within
# Y_DIFFERENCE_EPSILON × chunk_height of each other.  Use 0.1 (= 10%) as
# documented in AbstractTableProcessor.java:35.
_CLUSTER_Y_EPSILON_RATIO: float = 0.1

# When two chunks have the same baseline, a large horizontal gap (>
# X_DIFFERENCE_EPSILON × height) is required to consider them as separate
# columns rather than words in a sentence.  Value 3.0 from the same source.
_CLUSTER_X_GAP_RATIO: float = 3.0

# Minimum number of distinct rows that must share a column X-position before
# that position is accepted as a real column boundary.  Directly mirrors the
# "≥3 aligned rows" requirement stated in the task.
_CLUSTER_MIN_ROWS: int = 3

# Minimum number of distinct column positions required before a cluster is
# treated as a table.  A single column would be a paragraph, not a table.
_CLUSTER_MIN_COLS: int = 2

# Tolerance in PDF points for snapping column X-positions to the same slot.
# We use the same _SNAP_EPSILON already defined for Strategy 1.
# (reuses _SNAP_EPSILON = 1.5 pt from above)


# --- Internal helpers for cluster detection ---------------------------------

def _chunk_baseline(chunk: TextChunk) -> float:
    """Representative Y baseline (center Y) of a text chunk."""
    bb = chunk.bounding_box
    return (bb.bottom + bb.top) / 2.0


def _chunk_height(chunk: TextChunk) -> float:
    """Height of a text chunk's bounding box."""
    bb = chunk.bounding_box
    return bb.top - bb.bottom


def _group_chunks_into_rows(
    chunks: Sequence[TextChunk],
) -> list[list[TextChunk]]:
    """Group text chunks into horizontal rows by shared baseline.

    Two chunks belong to the same row when their center-Y values are within
    ``_CLUSTER_Y_EPSILON_RATIO × avg_height`` of the row's running mean Y.
    Rows are sorted top-to-bottom (descending center-Y, as PDF y-up coords).

    This mirrors the veraPDF ``ClusterTableConsumer`` row-band logic
    described in §4 of 05-tables-lists.md.
    """
    if not chunks:
        return []

    # Sort top-to-bottom (descending center Y) then left-to-right.
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (-_chunk_baseline(c), c.bounding_box.left),
    )

    rows: list[list[TextChunk]] = []
    row_centers: list[float] = []  # running mean center-Y per row

    for chunk in sorted_chunks:
        cy = _chunk_baseline(chunk)
        h = max(_chunk_height(chunk), 1.0)  # avoid zero-height degeneracy
        epsilon = _CLUSTER_Y_EPSILON_RATIO * h

        # Find the nearest existing row whose center-Y is within epsilon.
        best_row: int | None = None
        best_dist = float("inf")
        for i, rc in enumerate(row_centers):
            dist = abs(cy - rc)
            if dist <= epsilon and dist < best_dist:
                best_dist = dist
                best_row = i

        if best_row is None:
            # Start a new row.
            rows.append([chunk])
            row_centers.append(cy)
        else:
            rows[best_row].append(chunk)
            # Update running mean center-Y.
            n = len(rows[best_row])
            row_centers[best_row] = (row_centers[best_row] * (n - 1) + cy) / n

    # Sort rows top-to-bottom.
    rows_with_centers = sorted(
        zip(row_centers, rows), key=lambda t: -t[0]
    )
    return [row for _, row in rows_with_centers]


def _left_x_of_chunk(chunk: TextChunk) -> float:
    """Left X of a text chunk — used as the column anchor."""
    return chunk.bounding_box.left


def _find_column_positions(
    rows: list[list[TextChunk]],
) -> list[float]:
    """Find X positions that recur as column anchors across ≥ _CLUSTER_MIN_ROWS rows.

    For each chunk in each row we record its left-X as a candidate column
    position.  We then cluster near-duplicate X values (within _SNAP_EPSILON)
    and keep only those clusters that have contributions from at least
    _CLUSTER_MIN_ROWS distinct rows.  The returned list is sorted ascending.

    The "large horizontal gap" filter from §2 (X_DIFFERENCE_EPSILON) is
    implicitly enforced: within a single row, neighbouring chunks that are
    close together would produce overlapping left-X positions that collapse
    into the same cluster — so only genuinely spaced-out columns survive.
    """
    # Collect (left_x, row_index) pairs.
    candidates: list[tuple[float, int]] = []
    for row_idx, row_chunks in enumerate(rows):
        # Sort row left-to-right.
        for chunk in sorted(row_chunks, key=_left_x_of_chunk):
            candidates.append((_left_x_of_chunk(chunk), row_idx))

    if not candidates:
        return []

    # Cluster candidate X positions within _SNAP_EPSILON.
    # Each cluster records which row indices contributed to it.
    sorted_cands = sorted(candidates, key=lambda t: t[0])
    clusters: list[tuple[list[float], set[int]]] = []  # (x_values, row_set)

    for x_val, row_idx in sorted_cands:
        if clusters and x_val - clusters[-1][0][-1] <= _SNAP_EPSILON:
            clusters[-1][0].append(x_val)
            clusters[-1][1].add(row_idx)
        else:
            clusters.append(([x_val], {row_idx}))

    # Keep only clusters that span ≥ _CLUSTER_MIN_ROWS rows.
    col_positions: list[float] = []
    for x_vals, row_set in clusters:
        if len(row_set) >= _CLUSTER_MIN_ROWS:
            col_positions.append(sum(x_vals) / len(x_vals))  # cluster mean

    return sorted(col_positions)


def _assign_chunks_to_columns(
    row_chunks: list[TextChunk],
    col_positions: list[float],
) -> list[list[TextChunk]]:
    """Assign each chunk in a row to the nearest column (by left-X proximity).

    Returns a list of length ``len(col_positions)`` where each element is the
    list of chunks assigned to that column.
    """
    cols: list[list[TextChunk]] = [[] for _ in col_positions]
    for chunk in row_chunks:
        lx = _left_x_of_chunk(chunk)
        best_col = 0
        best_dist = float("inf")
        for c_idx, cx in enumerate(col_positions):
            dist = abs(lx - cx)
            if dist < best_dist:
                best_dist = dist
                best_col = c_idx
        cols[best_col].append(chunk)
    return cols


def _build_cluster_table_border(
    page: int,
    rows: list[list[TextChunk]],
    col_positions: list[float],
) -> TableBorder:
    """Assemble a TableBorder from cluster-detected rows and column positions.

    Row boundaries are derived from the bounding boxes of the chunks in each
    row (top-most and bottom-most Y across all chunks in the row).  Column
    boundaries are derived from the column positions: each column spans from
    its anchor X to the next column's anchor X (last column uses the right
    edge of its widest chunk).

    We reuse :func:`_build_table_border` rather than duplicating the cell-
    assembly logic — we just need to synthesize compatible y_seps / x_seps
    separator lists from the cluster geometry.
    """
    n_rows = len(rows)
    n_cols = len(col_positions)

    logger.info(
        "_build_cluster_table_border: page=%d, grid=%dx%d",
        page, n_rows, n_cols,
    )

    # ---- Build Y separators (descending, one per row boundary) ----
    # For each row: row_top = max top-Y of all chunks in the row
    #               row_bot = min bottom-Y of all chunks in the row
    row_tops: list[float] = []
    row_bots: list[float] = []
    for row_chunks in rows:
        if not row_chunks:
            row_tops.append(0.0)
            row_bots.append(0.0)
            continue
        tops = [c.bounding_box.top for c in row_chunks]
        bots = [c.bounding_box.bottom for c in row_chunks]
        row_tops.append(max(tops))
        row_bots.append(min(bots))

    # y_seps[i] is the TOP boundary of row i (and BOTTOM of row i-1).
    # We need n_rows + 1 separators.  Inter-row gaps: use midpoint between
    # consecutive row_bot / row_top pairs.
    y_seps: list[float] = [row_tops[0]]  # outermost top
    for i in range(n_rows - 1):
        gap_mid = (row_bots[i] + row_tops[i + 1]) / 2.0
        y_seps.append(gap_mid)
    y_seps.append(row_bots[-1])  # outermost bottom
    # y_seps is already descending (PDF y-up, rows go top→bottom).

    # ---- Build X separators (ascending, one per column boundary) ----
    # For each column: compute its right boundary as the midpoint to the next
    # column's left anchor.  The last column extends to the right edge of its
    # widest chunk.
    x_seps: list[float] = [col_positions[0]]
    for i in range(n_cols - 1):
        x_seps.append((col_positions[i + 1] + col_positions[i]) / 2.0)

    # Right edge of the last column: max right-X of all chunks assigned there.
    all_chunks_flat = [c for row_chunks in rows for c in row_chunks]
    # Assign to last column
    last_col_chunks = _assign_chunks_to_columns(all_chunks_flat, col_positions)[n_cols - 1]
    if last_col_chunks:
        last_right = max(c.bounding_box.right for c in last_col_chunks)
    else:
        # Fall back: last col position + typical column width
        if len(col_positions) >= 2:
            last_right = col_positions[-1] + (col_positions[-1] - col_positions[-2])
        else:
            last_right = col_positions[-1] + 50.0  # arbitrary fallback
    x_seps.append(last_right)

    logger.debug(
        "_build_cluster_table_border: y_seps=%s x_seps=%s",
        [round(v, 1) for v in y_seps],
        [round(v, 1) for v in x_seps],
    )

    # Delegate to the same cell-assembly helper used by Strategy 1.
    return _build_table_border(page, y_seps, x_seps, all_chunks_flat)


# --- Public API for Strategy 2 ----------------------------------------------

def detect_cluster_tables(
    text_chunks: Sequence[TextChunk],
) -> list[TableBorder]:
    """Detect Strategy-2 borderless (cluster) tables on a single page.

    Borderless tables have NO ruling lines; they are detected purely from
    text alignment: multiple rows where text chunks share consistent X
    positions across ≥ ``_CLUSTER_MIN_ROWS`` rows.

    **This function is NOT wired into the default pipeline.**  The bordered
    path plus frame-rejection already achieves 95–99% recall on the technical
    corpus; adding cluster detection by default risks regressions on
    heavily-formatted procedure PDFs whose multi-column prose would look
    tabular.  Wire it in once it has been validated on the full corpus.

    Parameters
    ----------
    text_chunks:
        All :class:`TextChunk` objects from a single page.  Whitespace-only
        chunks are filtered out internally (mirrors veraPDF's
        ``TextChunkUtils.splitTextChunkByWhiteSpaces`` pre-filter).

    Returns
    -------
    list[TableBorder]
        Zero or more detected borderless tables.

    Notes
    -----
    - Strategy 1 (bordered) tables should take priority; callers that combine
      both strategies should reject any cluster-detected table whose bounding
      box overlaps an already-registered bordered table by more than 1%
      (``TABLE_INTERSECTION_PERCENT = 0.01``), mirroring the Java
      ``addTablesToTableCollection`` deduplication guard.
    - Images inside borderless tables are not clustered (mirrors the
      commented-out ImageChunk block in ``ClusterTableProcessor.java:64–66``).
    """
    logger.info(
        "detect_cluster_tables: entry — %d text_chunks",
        len(text_chunks),
    )

    if not text_chunks:
        logger.debug("detect_cluster_tables: no text chunks → no tables")
        return []

    # Determine page from the first chunk.
    page = text_chunks[0].bounding_box.page_number

    # Filter to same page + non-whitespace chunks.
    page_chunks = [
        c for c in text_chunks
        if c.bounding_box.page_number == page and c.value.strip()
    ]
    if not page_chunks:
        logger.debug("detect_cluster_tables: no non-whitespace chunks on page %d", page)
        return []

    # Step 1: group chunks into horizontal rows by shared baseline.
    rows = _group_chunks_into_rows(page_chunks)
    logger.debug("detect_cluster_tables: %d rows after grouping", len(rows))

    # Step 2: find column X-positions that recur across ≥ _CLUSTER_MIN_ROWS rows.
    col_positions = _find_column_positions(rows)
    logger.debug(
        "detect_cluster_tables: %d recurring column positions: %s",
        len(col_positions),
        [round(x, 1) for x in col_positions],
    )

    # Guard: require minimum rows AND minimum columns.
    if len(rows) < _CLUSTER_MIN_ROWS:
        logger.info(
            "detect_cluster_tables: page=%d — only %d rows (need %d) → no table",
            page, len(rows), _CLUSTER_MIN_ROWS,
        )
        return []

    if len(col_positions) < _CLUSTER_MIN_COLS:
        logger.info(
            "detect_cluster_tables: page=%d — only %d column positions (need %d) → no table",
            page, len(col_positions), _CLUSTER_MIN_COLS,
        )
        return []

    # Step 3: build the TableBorder from the detected grid.
    table = _build_cluster_table_border(page, rows, col_positions)

    logger.info(
        "detect_cluster_tables: page=%d — produced 1 cluster table "
        "(%d rows × %d cols)",
        page,
        table.number_of_rows,
        table.number_of_columns,
    )
    return [table]
