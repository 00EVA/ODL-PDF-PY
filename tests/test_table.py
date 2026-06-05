# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for table entities (mirror of the Rust suite)."""

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import TextChunk
from odl_pdf.entities.table import TableBorder, TableBorderCell, TableBorderRow


def text(left, bottom, right, top, s) -> TextChunk:
    return TextChunk(
        BoundingBox.of(1, left, bottom, right, top), value=s, font_name="F", font_size=10.0
    )


def grid_2x2() -> TableBorder:
    rows = []
    for r in range(2):
        row = TableBorderRow(r)
        for c in range(2):
            cell = TableBorderCell(r, c)
            x = c * 50.0
            y = (1 - r) * 20.0
            cell.add_content(text(x, y, x + 50.0, y + 20.0, f"r{r}c{c}"))
            row.push(cell)
        rows.append(row)
    return TableBorder(rows)


def test_grid_dimensions_and_lookup():
    t = grid_2x2()
    assert t.number_of_rows == 2
    assert t.number_of_columns == 2
    assert not t.is_one_cell_table
    assert t.cell(1, 0).contents[0].value == "r1c0"
    assert t.cell(5, 5) is None


def test_one_cell_table_is_text_block():
    cell = TableBorderCell(0, 0)
    cell.add_content(text(0.0, 0.0, 10.0, 12.0, "solo"))
    t = TableBorder([TableBorderRow(0, [cell])])
    assert t.is_one_cell_table
    assert t.is_text_block


def test_column_count_accounts_for_col_span():
    wide = TableBorderCell(0, 0, col_span=3)
    wide.add_content(text(0.0, 0.0, 150.0, 12.0, "header"))
    t = TableBorder([TableBorderRow(0, [wide])])
    assert t.number_of_columns == 3
    assert t.cell(0, 0).col_span == 3


def test_cell_box_falls_back_to_contents():
    cell = TableBorderCell(0, 0)
    cell.add_content(text(0.0, 0.0, 20.0, 10.0, "a"))
    cell.add_content(text(20.0, 0.0, 40.0, 10.0, "b"))
    assert cell.bounding_box.to_list() == [0.0, 0.0, 40.0, 10.0]
    cell.set_bounding_box(BoundingBox.of(1, 0.0, 0.0, 100.0, 30.0))
    assert cell.bounding_box.to_list() == [0.0, 0.0, 100.0, 30.0]


def test_table_box_unions_all_cells():
    t = grid_2x2()
    assert t.bounding_box.to_list() == [0.0, 0.0, 100.0, 40.0]


def test_table_id_links():
    t = grid_2x2()
    t.recognized_structure_id = 7
    t.previous_table_id = 6
    t.next_table_id = 8
    assert t.recognized_structure_id == 7
    assert t.previous_table_id == 6
    assert t.next_table_id == 8
