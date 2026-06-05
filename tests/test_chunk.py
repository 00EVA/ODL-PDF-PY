# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for the chunk entities (mirror of the Rust suite)."""

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import ImageChunk, LineArtChunk, TextChunk


def bb() -> BoundingBox:
    return BoundingBox.of(1, 10.0, 20.0, 110.0, 40.0)


def test_text_chunk_geometry_delegates_to_bbox():
    c = TextChunk(bb(), value="hello", font_name="Helvetica", font_size=12.0)
    assert c.page_number == 1
    assert c.left == 10.0
    assert c.right == 110.0
    assert c.width == 100.0
    assert c.height == 20.0
    assert c.base_line == 20.0


def test_set_value_mutates_in_place():
    c = TextChunk(bb(), value="a�c", font_name="Helvetica", font_size=12.0)
    c.value = c.value.replace("�", "")
    assert c.value == "ac"


def test_whitespace_and_empty_flags():
    assert TextChunk(bb(), value="   ").is_whitespace
    assert not TextChunk(bb(), value=" x ").is_whitespace
    assert TextChunk(bb(), value="").is_empty
    assert not TextChunk(bb(), value="").is_whitespace


def test_rounded_font_weight():
    c = TextChunk(bb(), value="x")
    assert c.rounded_font_weight is None
    c.font_weight = 673.0
    assert c.rounded_font_weight == 700.0
    c.font_weight = 449.0
    assert c.rounded_font_weight == 400.0


def test_contrast_ratio_unset_then_set():
    c = TextChunk(bb(), value="x")
    assert c.contrast_ratio is None
    c.contrast_ratio = 2.5
    c.hidden_text = c.contrast_ratio < 4.5
    assert c.hidden_text


def test_line_art_background_flag():
    line = LineArtChunk(bb())
    assert not line.is_background
    line.is_background = True
    assert line.is_background


def test_image_chunk_indexing():
    img = ImageChunk(bb())
    img.index = 7
    assert img.index == 7
    assert img.page_number == 1
