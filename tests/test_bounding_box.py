# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for BoundingBox (mirror of the Rust test suite)."""

from odl_pdf.entities.bounding_box import BoundingBox


def test_of_normalizes_swapped_edges():
    b = BoundingBox.of(1, 10.0, 20.0, 5.0, 8.0)
    assert b.left == 5.0
    assert b.right == 10.0
    assert b.bottom == 8.0
    assert b.top == 20.0


def test_width_and_height():
    b = BoundingBox.of(1, 200.891, 706.938, 394.152, 745.132)
    assert abs(b.width - 193.261) < 1e-6
    assert abs(b.height - 38.194) < 1e-6


def test_to_list_matches_oracle_order():
    # lorem.pdf H1 box from the JSON conformance oracle.
    b = BoundingBox.of(1, 200.891, 706.938, 394.152, 745.132)
    assert b.to_list() == [200.891, 706.938, 394.152, 745.132]


def test_union_expands_bounds():
    a = BoundingBox.of(1, 0.0, 0.0, 10.0, 10.0)
    a.union(BoundingBox.of(1, 5.0, -5.0, 20.0, 8.0))
    assert a.to_list() == [0.0, -5.0, 20.0, 10.0]
    assert a.page_number == 1
    assert a.last_page_number == 1


def test_union_from_empty_accumulator():
    acc = BoundingBox.empty(2)
    acc.union(BoundingBox.of(2, 100.0, 100.0, 150.0, 120.0))
    # The (0,0) origin of the empty box still pins the lower-left corner.
    assert acc.to_list() == [0.0, 0.0, 150.0, 120.0]


def test_union_across_pages_records_span():
    a = BoundingBox.of(3, 0.0, 0.0, 10.0, 10.0)
    a.union(BoundingBox.of(5, 0.0, 0.0, 10.0, 10.0))
    assert a.page_number == 3
    assert a.last_page_number == 5


def test_contains_with_epsilon():
    outer = BoundingBox.of(1, 0.0, 0.0, 100.0, 100.0)
    inner = BoundingBox.of(1, 10.0, 10.0, 90.0, 90.0)
    assert outer.contains(inner)
    assert not inner.contains(outer)

    poke = BoundingBox.of(1, 10.0, 10.0, 101.0, 90.0)
    assert not outer.contains(poke)
    assert outer.contains(poke, x_epsilon=1.0)


def test_contains_is_page_aware():
    a = BoundingBox.of(1, 0.0, 0.0, 100.0, 100.0)
    other_page = BoundingBox.of(2, 10.0, 10.0, 20.0, 20.0)
    assert not a.contains(other_page, 5.0, 5.0)


def test_move_translates():
    b = BoundingBox.of(1, 10.0, 10.0, 20.0, 20.0)
    b.move(-10.0, 5.0)
    assert b.to_list() == [0.0, 15.0, 10.0, 25.0]
