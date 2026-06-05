# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of veraPDF's table grid model: ``TableBorder``,
# ``TableBorderRow``, ``TableBorderCell``. Accessors and the grid shape are
# reconstructed from the OpenDataLoader processor/writer call sites
# (``TableBorderProcessor``, ``PDFWriter``, ``MarkdownGenerator``) and the
# ``TableSerializer`` JSON field names. No veraPDF source copied.
"""Table grid entities."""

from __future__ import annotations

from dataclasses import dataclass, field

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.object import IObject, union_objects


def _union_boxes(boxes: list[BoundingBox | None]) -> BoundingBox | None:
    acc: BoundingBox | None = None
    for box in boxes:
        if box is None:
            continue
        if acc is None:
            acc = BoundingBox(**vars(box))
        else:
            acc.union(box)
    return acc


@dataclass
class TableBorderCell:
    """One cell of a table grid.

    Position is ``(row_number, col_number)``; a cell may span multiple rows or
    columns. ``contents`` are the ``IObject``s laid out inside the cell.
    """

    row_number: int
    col_number: int
    row_span: int = 1
    col_span: int = 1
    contents: list[IObject] = field(default_factory=list)
    _bounding_box: BoundingBox | None = None

    def add_content(self, content: IObject) -> None:
        self.contents.append(content)

    def set_bounding_box(self, bounding_box: BoundingBox) -> None:
        self._bounding_box = bounding_box

    @property
    def bounding_box(self) -> BoundingBox | None:
        """Explicit box if set, else the union of the cell's contents."""
        if self._bounding_box is not None:
            return self._bounding_box
        return union_objects(self.contents)


@dataclass
class TableBorderRow:
    """A row of a table grid."""

    row_number: int
    cells: list[TableBorderCell] = field(default_factory=list)

    def push(self, cell: TableBorderCell) -> None:
        self.cells.append(cell)

    @property
    def bounding_box(self) -> BoundingBox | None:
        return _union_boxes([cell.bounding_box for cell in self.cells])


@dataclass
class TableBorder:
    """A detected table: a grid of rows and cells."""

    rows: list[TableBorderRow] = field(default_factory=list)
    recognized_structure_id: int | None = None
    previous_table_id: int | None = None
    next_table_id: int | None = None
    _bounding_box: BoundingBox | None = None

    @property
    def number_of_rows(self) -> int:
        return len(self.rows)

    @property
    def number_of_columns(self) -> int:
        return max(
            (sum(cell.col_span for cell in row.cells) for row in self.rows),
            default=0,
        )

    def row(self, row_number: int) -> TableBorderRow | None:
        return next((r for r in self.rows if r.row_number == row_number), None)

    def cell(self, row: int, col: int) -> TableBorderCell | None:
        """The cell at ``(row, col)`` by its declared position."""
        found = self.row(row)
        if found is None:
            return None
        return next((c for c in found.cells if c.col_number == col), None)

    @property
    def is_one_cell_table(self) -> bool:
        """Whether the grid holds exactly one cell."""
        return self.number_of_rows == 1 and self.number_of_columns == 1

    @property
    def is_text_block(self) -> bool:
        """A 1x1 grid is treated as a plain text block, not a real table."""
        return self.is_one_cell_table

    def set_bounding_box(self, bounding_box: BoundingBox) -> None:
        self._bounding_box = bounding_box

    @property
    def bounding_box(self) -> BoundingBox | None:
        if self._bounding_box is not None:
            return self._bounding_box
        return _union_boxes([row.bounding_box for row in self.rows])
