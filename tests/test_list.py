# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for list entities (mirror of the Rust suite)."""

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import TextChunk
from odl_pdf.entities.list_entity import ListItem, PDFList


def text(left, bottom, right, top, s) -> TextChunk:
    return TextChunk(
        BoundingBox.of(1, left, bottom, right, top), value=s, font_name="F", font_size=10.0
    )


def test_list_item_box_unions_contents():
    item = ListItem()
    item.add_content(text(0.0, 0.0, 10.0, 12.0, "a"))
    item.add_content(text(10.0, 0.0, 30.0, 12.0, "b"))
    assert item.bounding_box.to_list() == [0.0, 0.0, 30.0, 12.0]


def test_list_counts_items_and_spans_box():
    i1 = ListItem([text(0.0, 20.0, 40.0, 32.0, "first")])
    i2 = ListItem([text(0.0, 0.0, 60.0, 12.0, "second")])
    lst = PDFList([i1, i2])
    assert lst.number_of_items == 2
    assert lst.bounding_box.to_list() == [0.0, 0.0, 60.0, 32.0]


def test_numbering_style_and_labels():
    lst = PDFList()
    assert lst.numbering_style is None
    lst.numbering_style = "ENGLISH_LETTERS"
    lst.label_length = 3
    assert lst.numbering_style == "ENGLISH_LETTERS"
    assert lst.label_length == 3


def test_sibling_list_links():
    lst = PDFList()
    lst.recognized_structure_id = 2
    lst.previous_list_id = 1
    lst.next_list_id = 3
    assert lst.recognized_structure_id == 2
    assert lst.previous_list_id == 1
    assert lst.next_list_id == 3
