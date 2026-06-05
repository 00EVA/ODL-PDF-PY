# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for text grouping (mirror of the Rust suite)."""

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import TextChunk
from odl_pdf.entities.text import TextBlock, TextColumn, TextLine


def chunk(left, bottom, right, top, text) -> TextChunk:
    return TextChunk(
        BoundingBox.of(1, left, bottom, right, top),
        value=text,
        font_name="F",
        font_size=10.0,
    )


def test_empty_line_has_no_geometry():
    line = TextLine()
    assert line.is_empty
    assert line.bounding_box is None
    assert line.base_line is None


def test_line_unions_chunk_boxes_and_concats_text():
    line = TextLine()
    line.push(chunk(0.0, 0.0, 10.0, 12.0, "Lorem "))
    line.push(chunk(10.0, 0.0, 25.0, 12.0, "ipsum"))
    assert line.value == "Lorem ipsum"
    assert line.bounding_box.to_list() == [0.0, 0.0, 25.0, 12.0]
    assert line.base_line == 0.0
    assert line.font_size == 10.0


def test_space_line_vs_blank():
    spacer = TextLine([chunk(0.0, 0.0, 5.0, 12.0, "   ")])
    assert spacer.is_space_line
    assert not spacer.is_blank

    blank = TextLine([chunk(0.0, 0.0, 0.0, 0.0, "")])
    assert not blank.is_space_line
    assert blank.is_blank


def test_block_first_non_space_line_skips_spacers():
    spacer = TextLine([chunk(0.0, 24.0, 5.0, 36.0, "  ")])
    real = TextLine([chunk(0.0, 0.0, 30.0, 12.0, "Heading")])
    block = TextBlock([spacer, real])
    assert block.first_non_space_line.value == "Heading"
    assert block.first_line.is_space_line


def test_block_unions_lines_and_joins_with_newline():
    l1 = TextLine([chunk(0.0, 20.0, 40.0, 32.0, "first")])
    l2 = TextLine([chunk(0.0, 0.0, 60.0, 12.0, "second")])
    block = TextBlock([l1, l2])
    assert block.value == "first\nsecond"
    assert block.bounding_box.to_list() == [0.0, 0.0, 60.0, 32.0]


def test_column_aggregates_blocks():
    b1 = TextBlock([TextLine([chunk(0.0, 40.0, 40.0, 52.0, "a")])])
    b2 = TextBlock([TextLine([chunk(5.0, 0.0, 80.0, 12.0, "b")])])
    col = TextColumn([b1, b2])
    assert col.bounding_box.to_list() == [0.0, 0.0, 80.0, 52.0]
    assert col.first_line.value == "a"
    assert col.value == "a\nb"


def test_union_does_not_mutate_child_box():
    c = chunk(0.0, 0.0, 10.0, 12.0, "x")
    line = TextLine([c])
    _ = line.bounding_box
    # The chunk's own box must be untouched by the line's union accumulator.
    assert c.bounding_box.to_list() == [0.0, 0.0, 10.0, 12.0]
