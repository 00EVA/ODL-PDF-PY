# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for IObject helpers (mirror of the Rust suite)."""

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import ImageChunk, TextChunk
from odl_pdf.entities.object import IObject, union_objects


def test_bounding_box_dispatches_per_variant():
    text = TextChunk(BoundingBox.of(1, 0.0, 0.0, 10.0, 12.0), value="x")
    assert text.bounding_box.to_list() == [0.0, 0.0, 10.0, 12.0]
    img = ImageChunk(BoundingBox.of(1, 5.0, 5.0, 20.0, 20.0))
    assert img.bounding_box.to_list() == [5.0, 5.0, 20.0, 20.0]


def test_union_objects_spans_all():
    objs = [
        TextChunk(BoundingBox.of(1, 0.0, 40.0, 40.0, 52.0), value="a"),
        ImageChunk(BoundingBox.of(1, 5.0, 0.0, 80.0, 12.0)),
    ]
    assert union_objects(objs).to_list() == [0.0, 0.0, 80.0, 52.0]
    assert union_objects([]) is None


def test_entities_satisfy_iobject_protocol():
    text = TextChunk(BoundingBox.of(1, 0.0, 0.0, 1.0, 1.0), value="x")
    assert isinstance(text, IObject)
