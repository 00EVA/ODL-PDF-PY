# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for semantic nodes (mirror of the Rust suite)."""

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import TextChunk
from odl_pdf.entities.semantic import SemanticTextNode, SemanticType
from odl_pdf.entities.text import TextBlock, TextColumn, TextLine


def column(text, left, bottom, right, top) -> TextColumn:
    chunk = TextChunk(
        BoundingBox.of(1, left, bottom, right, top),
        value=text,
        font_name="F",
        font_size=12.0,
    )
    return TextColumn([TextBlock([TextLine([chunk])])])


def test_pdfua_tag_for_common_types():
    assert SemanticType.PARAGRAPH.pdfua_tag() == "P"
    assert SemanticType.FIGURE.pdfua_tag() == "Figure"
    assert SemanticType.LIST.pdfua_tag() == "L"
    assert SemanticType.LIST_ITEM.pdfua_tag() == "LI"
    assert SemanticType.TABLE.pdfua_tag() == "Table"
    assert SemanticType.TABLE_CELL.pdfua_tag() == "TD"
    assert SemanticType.CAPTION.pdfua_tag() is None
    assert SemanticType.HEADER.pdfua_tag() is None


def test_heading_tag_uses_level():
    h = SemanticType.HEADING
    assert h.pdfua_tag(1) == "H1"
    assert h.pdfua_tag(6) == "H6"
    assert h.pdfua_tag(7) == "H"
    assert h.pdfua_tag(None) == "H"


def test_heading_node_carries_level_and_tag():
    node = SemanticTextNode.heading(2, [column("Title", 0.0, 0.0, 50.0, 14.0)])
    assert node.semantic_type is SemanticType.HEADING
    assert node.heading_level == 2
    assert node.pdfua_tag == "H2"
    assert node.value == "Title"


def test_reclassify_preserves_initial_type():
    node = SemanticTextNode.paragraph([column("x", 0.0, 0.0, 10.0, 12.0)])
    assert node.initial_semantic_type is SemanticType.PARAGRAPH
    node.set_semantic_type(SemanticType.HEADING)
    node.heading_level = 1
    assert node.semantic_type is SemanticType.HEADING
    assert node.initial_semantic_type is SemanticType.PARAGRAPH
    assert node.pdfua_tag == "H1"


def test_node_bounding_box_spans_columns():
    node = SemanticTextNode.paragraph(
        [column("a", 0.0, 40.0, 40.0, 52.0), column("b", 5.0, 0.0, 80.0, 12.0)]
    )
    assert node.bounding_box.to_list() == [0.0, 0.0, 80.0, 52.0]
    assert node.first_line.value == "a"
