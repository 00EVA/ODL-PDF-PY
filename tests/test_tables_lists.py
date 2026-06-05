# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for Strategy-1 bordered table detection and list
detection (mirror of the Rust ``test_tables_lists`` suite).

Tests are written BEFORE the implementation and cover:
  - Tables: worked 2x2 grid example from §3 of 05-tables-lists.md
  - Lists:  "1. / 2. / 3." example from §8 of 05-tables-lists.md
  - Edge cases: empty inputs, single-row tables, unordered/mixed lists
"""

from __future__ import annotations

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import LineArtChunk, TextChunk
from odl_pdf.entities.table import TableBorder
from odl_pdf.entities.list_entity import PDFList
from odl_pdf.entities.text import TextLine
from odl_pdf.processors.tables import detect_bordered_tables, detect_cluster_tables
from odl_pdf.processors.lists import detect_lists


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bbox(page: int, left: float, bottom: float, right: float, top: float) -> BoundingBox:
    return BoundingBox.of(page, left, bottom, right, top)


def line_art(page: int, left: float, bottom: float, right: float, top: float,
             is_background: bool = False) -> LineArtChunk:
    return LineArtChunk(bbox(page, left, bottom, right, top), is_background=is_background)


def text_chunk(page: int, left: float, bottom: float, right: float, top: float,
               value: str, font_size: float = 10.0) -> TextChunk:
    return TextChunk(
        bbox(page, left, bottom, right, top),
        value=value,
        font_name="Helvetica",
        font_size=font_size,
    )


def text_line(*chunks: TextChunk) -> TextLine:
    """Build a TextLine from one or more TextChunks."""
    tl = TextLine()
    for c in chunks:
        tl.push(c)
    return tl


# ---------------------------------------------------------------------------
# Table tests
# ---------------------------------------------------------------------------

def _make_2x2_rules_and_chunks():
    """
    2×2 grid with explicit x/y separators.

    Grid layout (y-up, page 1):
      ┌─────────┬─────────┐  y=100
      │ r0c0    │ r0c1    │
      │ (0-50,  │ (50-100,│
      │  50-100)│  50-100)│
      ├─────────┼─────────┤  y=50
      │ r1c0    │ r1c1    │
      │ (0-50,  │ (50-100,│
      │  0-50)  │  0-50)  │
      └─────────┴─────────┘  y=0
      x=0       x=50       x=100

    Rules:
      - H-rule top:    y=100, x: 0..100
      - H-rule middle: y=50,  x: 0..100
      - H-rule bottom: y=0,   x: 0..100
      - V-rule left:   x=0,   y: 0..100
      - V-rule middle: x=50,  y: 0..100
      - V-rule right:  x=100, y: 0..100
    """
    page = 1
    rules = [
        # Horizontal rules (very thin height, spanning full width)
        line_art(page, 0.0, 99.9, 100.0, 100.0),   # top H
        line_art(page, 0.0, 49.9, 100.0, 50.0),    # middle H
        line_art(page, 0.0, 0.0, 100.0, 0.1),      # bottom H
        # Vertical rules (very thin width, spanning full height)
        line_art(page, 0.0, 0.0, 0.1, 100.0),      # left V
        line_art(page, 49.9, 0.0, 50.0, 100.0),    # middle V
        line_art(page, 99.9, 0.0, 100.0, 100.0),   # right V
    ]
    chunks = [
        text_chunk(page, 1.0, 51.0, 49.0, 99.0, "r0c0"),
        text_chunk(page, 51.0, 51.0, 99.0, 99.0, "r0c1"),
        text_chunk(page, 1.0, 1.0, 49.0, 49.0, "r1c0"),
        text_chunk(page, 51.0, 1.0, 99.0, 49.0, "r1c1"),
    ]
    return rules, chunks


def test_bordered_table_2x2_basic():
    """Worked example: 4 rules forming a 2×2 grid → one TableBorder with shape (2,2)."""
    rules, chunks = _make_2x2_rules_and_chunks()
    tables = detect_bordered_tables(rules, chunks)
    assert len(tables) == 1, f"expected 1 table, got {len(tables)}"
    t: TableBorder = tables[0]
    assert t.number_of_rows == 2, f"expected 2 rows, got {t.number_of_rows}"
    assert t.number_of_columns == 2, f"expected 2 cols, got {t.number_of_columns}"


def test_bordered_table_2x2_cell_contents():
    """Each cell must contain exactly the TextChunk whose centre falls in it."""
    rules, chunks = _make_2x2_rules_and_chunks()
    tables = detect_bordered_tables(rules, chunks)
    assert len(tables) == 1
    t = tables[0]

    for r in range(2):
        for c in range(2):
            cell = t.cell(r, c)
            assert cell is not None, f"cell({r},{c}) is None"
            values = [obj.value for obj in cell.contents if hasattr(obj, "value")]
            expected = f"r{r}c{c}"
            assert expected in values, (
                f"cell({r},{c}) expected '{expected}', got {values}"
            )


def test_bordered_table_no_rules_no_table():
    """No line art → no tables detected."""
    chunks = [text_chunk(1, 0.0, 0.0, 50.0, 12.0, "lone text")]
    tables = detect_bordered_tables([], chunks)
    assert tables == []


def test_bordered_table_empty_inputs():
    """Both empty → no tables."""
    assert detect_bordered_tables([], []) == []


def test_bordered_table_background_art_ignored():
    """LineArtChunks marked is_background are skipped during rule analysis."""
    page = 1
    rules = [
        line_art(page, 0.0, 49.9, 100.0, 50.0, is_background=True),
        line_art(page, 0.0, 99.9, 100.0, 100.0, is_background=True),
    ]
    chunks = [text_chunk(page, 1.0, 1.0, 49.0, 49.0, "x")]
    tables = detect_bordered_tables(rules, chunks)
    # Background art only → no usable grid → no table
    assert tables == []


def test_bordered_table_bounding_box():
    """The returned TableBorder's bounding_box must cover all cells."""
    rules, chunks = _make_2x2_rules_and_chunks()
    tables = detect_bordered_tables(rules, chunks)
    assert len(tables) == 1
    bb = tables[0].bounding_box
    assert bb is not None
    # Table spans (0..100, 0..100) — cells run 1..99 so bbox covers that range
    assert bb.left < 2.0
    assert bb.right > 98.0
    assert bb.bottom < 2.0
    assert bb.top > 98.0


def test_bordered_table_1x1_is_text_block():
    """A table with a single cell is flagged as a text block (not a real table)."""
    page = 1
    rules = [
        line_art(page, 0.0, 99.9, 100.0, 100.0),
        line_art(page, 0.0, 0.0, 100.0, 0.1),
        line_art(page, 0.0, 0.0, 0.1, 100.0),
        line_art(page, 99.9, 0.0, 100.0, 100.0),
    ]
    chunks = [text_chunk(page, 5.0, 5.0, 95.0, 95.0, "solo")]
    tables = detect_bordered_tables(rules, chunks)
    if tables:
        # If a 1×1 table was produced, its is_text_block must be True
        assert tables[0].is_text_block


# ---------------------------------------------------------------------------
# List tests
# ---------------------------------------------------------------------------

def _make_numbered_lines():
    """Three consecutive '1. Item' / '2. Item' / '3. Item' text lines on page 1."""
    page = 1
    lines = []
    for i, label in enumerate(["1. ", "2. ", "3. "]):
        y_top = 100.0 - i * 15.0
        y_bot = y_top - 12.0
        chunk = text_chunk(page, 0.0, y_bot, 100.0, y_top,
                           f"{label}Item {i + 1}", font_size=12.0)
        tl = text_line(chunk)
        lines.append(tl)
    return lines


def test_detect_lists_numbered_basic():
    """Three '1. / 2. / 3.' lines → one ordered PDFList with 3 items."""
    lines = _make_numbered_lines()
    lists = detect_lists(lines)
    assert len(lists) == 1, f"expected 1 list, got {len(lists)}: {lists}"
    lst: PDFList = lists[0]
    assert lst.number_of_items == 3, f"expected 3 items, got {lst.number_of_items}"


def test_detect_lists_numbered_style():
    """Numbered list must carry 'DECIMAL' (or equivalent ordered) numbering style."""
    lines = _make_numbered_lines()
    lists = detect_lists(lines)
    assert len(lists) == 1
    style = lists[0].numbering_style
    assert style is not None, "numbering_style must not be None for a numbered list"
    # Accept "DECIMAL" or "ARABIC_NUMBERS" — both map to ordered decimal
    assert style in {"DECIMAL", "ARABIC_NUMBERS"}, (
        f"expected DECIMAL or ARABIC_NUMBERS, got {style!r}"
    )


def test_detect_lists_bullet_unordered():
    """Bullet lines (•) → one unordered PDFList."""
    page = 1
    lines = []
    for i, bullet in enumerate(["• ", "• ", "• "]):
        y_top = 100.0 - i * 15.0
        y_bot = y_top - 12.0
        chunk = text_chunk(page, 0.0, y_bot, 100.0, y_top,
                           f"{bullet}item {i + 1}", font_size=12.0)
        lines.append(text_line(chunk))
    lists = detect_lists(lines)
    assert len(lists) == 1
    assert lists[0].number_of_items == 3
    style = lists[0].numbering_style
    assert style in {"BULLET", "UNORDERED"}, (
        f"expected BULLET or UNORDERED, got {style!r}"
    )


def test_detect_lists_dash_unordered():
    """Dash-prefixed lines (- ) → one unordered PDFList."""
    page = 1
    lines = []
    for i in range(3):
        y_top = 100.0 - i * 15.0
        chunk = text_chunk(page, 0.0, y_top - 12.0, 100.0, y_top,
                           f"- item {i + 1}", font_size=12.0)
        lines.append(text_line(chunk))
    lists = detect_lists(lines)
    assert len(lists) == 1
    assert lists[0].numbering_style in {"BULLET", "UNORDERED"}


def test_detect_lists_star_unordered():
    """Star-prefixed lines (* ) → one unordered PDFList."""
    page = 1
    lines = []
    for i in range(2):
        y_top = 80.0 - i * 15.0
        chunk = text_chunk(page, 0.0, y_top - 12.0, 100.0, y_top,
                           f"* item {i + 1}", font_size=12.0)
        lines.append(text_line(chunk))
    lists = detect_lists(lines)
    assert len(lists) == 1
    assert lists[0].numbering_style in {"BULLET", "UNORDERED"}


def test_detect_lists_alpha_ordered():
    """'a. / b. / c.' lines → one ordered PDFList with ENGLISH_LETTERS style."""
    page = 1
    lines = []
    for i, label in enumerate(["a. ", "b. ", "c. "]):
        y_top = 100.0 - i * 15.0
        chunk = text_chunk(page, 0.0, y_top - 12.0, 100.0, y_top,
                           f"{label}Text {i + 1}", font_size=12.0)
        lines.append(text_line(chunk))
    lists = detect_lists(lines)
    assert len(lists) == 1
    assert lists[0].numbering_style in {
        "ENGLISH_LETTERS", "ENGLISH_LETTERS_LOWER_CASE"
    }


def test_detect_lists_roman_ordered():
    """'i. / ii. / iii.' lines → one ordered PDFList with roman style."""
    page = 1
    lines = []
    for i, label in enumerate(["i. ", "ii. ", "iii. "]):
        y_top = 100.0 - i * 15.0
        chunk = text_chunk(page, 0.0, y_top - 12.0, 100.0, y_top,
                           f"{label}Point {i + 1}", font_size=12.0)
        lines.append(text_line(chunk))
    lists = detect_lists(lines)
    assert len(lists) == 1
    assert lists[0].numbering_style in {
        "ROMAN_NUMBERS_LOWER_CASE", "ROMAN_NUMBERS", "ROMAN_NUMBERS_UPPER_CASE"
    }


def test_detect_lists_single_item_not_a_list():
    """A single labelled line does not form a list (need ≥ 2 items)."""
    page = 1
    chunk = text_chunk(page, 0.0, 0.0, 100.0, 12.0, "1. solo item", font_size=12.0)
    lines = [text_line(chunk)]
    lists = detect_lists(lines)
    assert lists == [], f"single item should not form a list, got {lists}"


def test_detect_lists_empty_input():
    """No lines → no lists."""
    assert detect_lists([]) == []


def test_detect_lists_plain_text_no_list():
    """Lines without list labels are not grouped into lists."""
    page = 1
    lines = [
        text_line(text_chunk(page, 0.0, 80.0, 100.0, 92.0, "Paragraph one.")),
        text_line(text_chunk(page, 0.0, 60.0, 100.0, 72.0, "Paragraph two.")),
        text_line(text_chunk(page, 0.0, 40.0, 100.0, 52.0, "Paragraph three.")),
    ]
    lists = detect_lists(lines)
    assert lists == []


def test_detect_lists_paren_number_style():
    """'1) / 2) / 3)' style lines → one ordered list."""
    page = 1
    lines = []
    for i in range(3):
        y_top = 100.0 - i * 15.0
        chunk = text_chunk(page, 0.0, y_top - 12.0, 100.0, y_top,
                           f"{i + 1}) item", font_size=12.0)
        lines.append(text_line(chunk))
    lists = detect_lists(lines)
    assert len(lists) == 1
    assert lists[0].number_of_items == 3
    assert lists[0].numbering_style in {"DECIMAL", "ARABIC_NUMBERS"}


def test_detect_lists_items_have_contents():
    """Each list item must carry at least one content object."""
    lines = _make_numbered_lines()
    lists = detect_lists(lines)
    assert len(lists) == 1
    for idx, item in enumerate(lists[0].items):
        assert len(item.contents) >= 1, (
            f"item {idx} has no contents"
        )


def test_table_no_cell_text_duplication():
    """A wide chunk spanning multiple columns must land in ONE cell, not be
    copied into every overlapping column (the 3x-inflation regression)."""
    page = 0
    rules = [
        # horizontal rules: top, middle, bottom (2 row bands)
        line_art(page, 0.0, 99.9, 300.0, 100.0),
        line_art(page, 0.0, 49.9, 300.0, 50.0),
        line_art(page, 0.0, 0.0, 300.0, 0.1),
        # vertical rules: 3 columns at x = 0, 100, 200, 300
        line_art(page, 0.0, 0.0, 0.1, 100.0),
        line_art(page, 99.9, 0.0, 100.0, 100.0),
        line_art(page, 199.9, 0.0, 200.0, 100.0),
        line_art(page, 299.9, 0.0, 300.0, 100.0),
    ]
    # One very wide chunk spanning all 3 columns in the top band.
    wide = text_chunk(page, 5.0, 55.0, 295.0, 95.0, "wide spanning text", font_size=10.0)
    tables = detect_bordered_tables(rules, [wide])
    assert tables, "expected a table"
    placements = sum(
        1 for r in tables[0].rows for c in r.cells for _ in c.contents
    )
    assert placements == 1, f"chunk duplicated into {placements} cells"


# ---------------------------------------------------------------------------
# Strategy-2 cluster / borderless table tests
# ---------------------------------------------------------------------------

def _make_3x3_borderless_chunks(page: int = 1) -> list:
    """Synthetic 3×3 borderless grid: 3 rows × 3 columns, no ruling lines.

    Layout (y-up, PDF coords):
      Row 0 (top):    y = 80–90   cols at x = 0, 100, 200
      Row 1 (middle): y = 50–60   cols at x = 0, 100, 200
      Row 2 (bottom): y = 20–30   cols at x = 0, 100, 200

    Each chunk occupies a 40-point-wide slot (left..left+40) so columns are
    well-separated (60 pt gap between chunks in the same row > 3×height=30).
    """
    chunks = []
    col_xs = [0.0, 100.0, 200.0]
    row_bots = [80.0, 50.0, 20.0]
    row_tops = [90.0, 60.0, 30.0]
    for r, (y_bot, y_top) in enumerate(zip(row_bots, row_tops)):
        for c, x in enumerate(col_xs):
            chunks.append(
                text_chunk(page, x, y_bot, x + 40.0, y_top, f"r{r}c{c}")
            )
    return chunks


def test_cluster_table_3x3_detected():
    """Synthetic 3×3 aligned grid → detect_cluster_tables returns one table."""
    chunks = _make_3x3_borderless_chunks()
    tables = detect_cluster_tables(chunks)
    assert len(tables) == 1, f"expected 1 cluster table, got {len(tables)}"


def test_cluster_table_3x3_shape():
    """Detected cluster table must have exactly 3 rows and 3 columns."""
    chunks = _make_3x3_borderless_chunks()
    tables = detect_cluster_tables(chunks)
    assert len(tables) == 1
    t: TableBorder = tables[0]
    assert t.number_of_rows == 3, f"expected 3 rows, got {t.number_of_rows}"
    assert t.number_of_columns == 3, f"expected 3 cols, got {t.number_of_columns}"


def test_cluster_table_3x3_cell_contents():
    """Each of the 9 cells in the cluster table must hold the right chunk."""
    chunks = _make_3x3_borderless_chunks()
    tables = detect_cluster_tables(chunks)
    assert len(tables) == 1
    t = tables[0]
    for r in range(3):
        for c in range(3):
            cell = t.cell(r, c)
            assert cell is not None, f"cell({r},{c}) is None"
            values = [obj.value for obj in cell.contents if hasattr(obj, "value")]
            expected = f"r{r}c{c}"
            assert expected in values, (
                f"cell({r},{c}) expected {expected!r}, got {values}"
            )


def test_cluster_table_paragraph_not_detected():
    """Plain paragraph (single column of text) is NOT detected as a table.

    A paragraph produces only one column position; with fewer than
    _CLUSTER_MIN_COLS = 2 columns, detect_cluster_tables must return [].
    """
    page = 1
    # 5 lines of prose stacked vertically, all left-aligned at x=10
    chunks = []
    for i in range(5):
        y_top = 100.0 - i * 15.0
        chunks.append(
            text_chunk(page, 10.0, y_top - 12.0, 200.0, y_top, f"Paragraph line {i + 1}.")
        )
    tables = detect_cluster_tables(chunks)
    assert tables == [], (
        f"paragraph should not be detected as a table, got {len(tables)} table(s)"
    )


def test_cluster_table_too_few_rows_not_detected():
    """Only 2 rows (< _CLUSTER_MIN_ROWS = 3) → no cluster table."""
    page = 1
    # 2-row, 3-column aligned grid — not enough rows
    chunks = []
    for r, y_bot in enumerate([60.0, 20.0]):
        for c, x in enumerate([0.0, 100.0, 200.0]):
            chunks.append(text_chunk(page, x, y_bot, x + 40.0, y_bot + 10.0, f"r{r}c{c}"))
    tables = detect_cluster_tables(chunks)
    assert tables == [], (
        f"2 rows should not form a cluster table, got {len(tables)} table(s)"
    )


def test_cluster_table_empty_input():
    """No chunks → no tables."""
    assert detect_cluster_tables([]) == []


def test_cluster_table_whitespace_only_not_detected():
    """Whitespace-only chunks are filtered out; result is no table."""
    page = 1
    chunks = [
        text_chunk(page, 0.0, 80.0, 50.0, 90.0, "   "),
        text_chunk(page, 100.0, 80.0, 150.0, 90.0, "\t"),
        text_chunk(page, 0.0, 50.0, 50.0, 60.0, ""),
    ]
    tables = detect_cluster_tables(chunks)
    assert tables == []


def test_cluster_table_not_in_default_pipeline():
    """detect_bordered_tables must NOT call detect_cluster_tables internally.

    A synthetic borderless 3×3 grid (no ruling lines) must produce zero
    tables from detect_bordered_tables, confirming Strategy 2 is opt-in only.
    """
    chunks = _make_3x3_borderless_chunks()
    # No line art → Strategy 1 has no rules to infer a grid from.
    tables = detect_bordered_tables([], chunks)
    assert tables == [], (
        "detect_bordered_tables must not run cluster detection (Strategy 2 is opt-in)"
    )
