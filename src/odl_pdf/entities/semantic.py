# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of the semantic node layer from veraPDF's ``wcag-algorithms``
# model (``SemanticType``, ``SemanticTextNode``, ``SemanticHeading``, ...). The
# ``SemanticType`` members are reconstructed from the ``SemanticType.*``
# constants referenced by the OpenDataLoader processors; the type->PDF/UA tag
# mapping is ported from the upstream opendataloader-pdf code
# (Apache-2.0, SerializerUtil.pdfuaTagFor). No veraPDF source was copied.
"""Semantic nodes: the classified structure assigned on top of grouping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.text import TextColumn, TextLine


class SemanticType(Enum):
    """The structural role a node plays, mirroring veraPDF's ``SemanticType``."""

    PART = "part"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CAPTION = "caption"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_HEADER = "table_header"
    TABLE_HEADERS = "table_headers"
    TABLE_FOOTER = "table_footer"
    TABLE_BODY = "table_body"
    TABLE_CELL = "table_cell"

    def pdfua_tag(self, heading_level: int | None = None) -> str | None:
        """The PDF/UA structure tag for this type, or ``None`` when the node has
        no canonical tag.

        Headings choose ``H1``..``H6`` from ``heading_level``, falling back to
        the unleveled ``H`` (PDF/UA-2 deprecated, but the only honest output
        without a level).
        """
        if self is SemanticType.HEADING:
            if heading_level is not None and 1 <= heading_level <= 6:
                return f"H{heading_level}"
            return "H"
        return {
            SemanticType.PARAGRAPH: "P",
            SemanticType.FIGURE: "Figure",
            SemanticType.LIST: "L",
            SemanticType.LIST_ITEM: "LI",
            SemanticType.TABLE: "Table",
            SemanticType.TABLE_CELL: "TD",
        }.get(self)


def _union_columns(columns: list[TextColumn]) -> BoundingBox | None:
    acc: BoundingBox | None = None
    for column in columns:
        box = column.bounding_box
        if box is None:
            continue
        if acc is None:
            acc = BoundingBox(**vars(box))
        else:
            acc.union(box)
    return acc


@dataclass
class SemanticTextNode:
    """A classified run of text, owning the column grouping beneath it.

    Carries both the current ``semantic_type`` and the *initial* type assigned
    at construction; later passes may reclassify while ``initial_semantic_type``
    is preserved for heuristics that compare against it.
    """

    semantic_type: SemanticType
    columns: list[TextColumn] = field(default_factory=list)
    heading_level: int | None = None
    recognized_structure_id: int | None = None
    level: str | None = None
    initial_semantic_type: SemanticType | None = None

    def __post_init__(self) -> None:
        if self.initial_semantic_type is None:
            self.initial_semantic_type = self.semantic_type

    @classmethod
    def paragraph(cls, columns: list[TextColumn]) -> "SemanticTextNode":
        """A paragraph node (the default classification for a text run)."""
        return cls(SemanticType.PARAGRAPH, columns)

    @classmethod
    def heading(cls, level: int, columns: list[TextColumn]) -> "SemanticTextNode":
        """A heading node at the given level (1..6)."""
        return cls(SemanticType.HEADING, columns, heading_level=level)

    def set_semantic_type(self, semantic_type: SemanticType) -> None:
        """Reclassify the node; the initial type is left unchanged."""
        self.semantic_type = semantic_type

    @property
    def pdfua_tag(self) -> str | None:
        """The PDF/UA tag for this node, accounting for heading level."""
        return self.semantic_type.pdfua_tag(self.heading_level)

    @property
    def bounding_box(self) -> BoundingBox | None:
        """Box spanning every column, or ``None`` when the node is empty."""
        return _union_columns(self.columns)

    @property
    def first_line(self) -> TextLine | None:
        """First line across all columns, in reading order."""
        for column in self.columns:
            if column.first_line is not None:
                return column.first_line
        return None

    @property
    def value(self) -> str:
        return "\n".join(column.value for column in self.columns)

    @property
    def is_empty(self) -> bool:
        return all(column.is_empty for column in self.columns)
