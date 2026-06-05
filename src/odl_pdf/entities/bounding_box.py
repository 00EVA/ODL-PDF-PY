# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room reimplementation of the geometry contract that
# ``org.verapdf.wcag.algorithms.entities.geometry.BoundingBox`` exposes to the
# OpenDataLoader processors. No veraPDF source was copied; the behavior is
# reconstructed from the documented call sites in the upstream
# opendataloader-pdf project and its JSON conformance oracle
# (coordinate order ``[left, bottom, right, top]``).
"""Axis-aligned bounding box in PDF user space (y-up)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundingBox:
    """An axis-aligned rectangle on one or more PDF pages.

    Coordinates are in PDF points (1/72 inch), y-up, so ``bottom <= top`` and
    ``left <= right``. The serialized form used by the JSON oracle is
    ``[left, bottom, right, top]``.

    Construct via :meth:`of` (single page) or :meth:`empty`; the four edges are
    normalized so callers may pass them in any order.
    """

    page_number: int
    last_page_number: int
    left: float
    bottom: float
    right: float
    top: float

    @classmethod
    def of(
        cls, page_number: int, left: float, bottom: float, right: float, top: float
    ) -> "BoundingBox":
        """A single-page box from its four edges, normalized."""
        return cls(
            page_number=page_number,
            last_page_number=page_number,
            left=min(left, right),
            bottom=min(bottom, top),
            right=max(left, right),
            top=max(bottom, top),
        )

    @classmethod
    def empty(cls, page_number: int) -> "BoundingBox":
        """A degenerate box at the origin, for use as a ``union`` accumulator.

        Mirrors the Java ``new BoundingBox(pageIndex, 0, 0, 0, 0)`` idiom.
        """
        return cls.of(page_number, 0.0, 0.0, 0.0, 0.0)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def to_list(self) -> list[float]:
        """``[left, bottom, right, top]`` — the JSON oracle's coordinate order."""
        return [self.left, self.bottom, self.right, self.top]

    def union(self, other: "BoundingBox") -> None:
        """Expand this box in place to also cover ``other``.

        The page span widens to ``min(page)..max(last_page)`` so a union across
        pages records a multi-page span.
        """
        self.left = min(self.left, other.left)
        self.bottom = min(self.bottom, other.bottom)
        self.right = max(self.right, other.right)
        self.top = max(self.top, other.top)
        self.page_number = min(self.page_number, other.page_number)
        self.last_page_number = max(self.last_page_number, other.last_page_number)

    def contains(
        self, other: "BoundingBox", x_epsilon: float = 0.0, y_epsilon: float = 0.0
    ) -> bool:
        """Whether ``other`` lies within this box, expanded by the epsilons.

        Page-aware: only contains a box on the same page span. Used by the
        caption heuristic where the epsilons are derived from font size.
        """
        return (
            self.page_number == other.page_number
            and self.last_page_number == other.last_page_number
            and self.left - x_epsilon <= other.left
            and self.bottom - y_epsilon <= other.bottom
            and self.right + x_epsilon >= other.right
            and self.top + y_epsilon >= other.top
        )

    def move(self, dx: float, dy: float) -> None:
        """Translate the box by ``(dx, dy)`` in place."""
        self.left += dx
        self.right += dx
        self.bottom += dy
        self.top += dy
