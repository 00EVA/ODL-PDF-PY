# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of the text-grouping containers from veraPDF's
# ``wcag-algorithms`` model: ``TextLine``, ``TextBlock``, ``TextColumn``. The
# grouping hierarchy and accessors are reconstructed from the OpenDataLoader
# processor call sites (``TextLineProcessor``, ``ParagraphProcessor``,
# ``HeadingProcessor``); see ``docs/architecture/02-pdf-parsing-layer.md`` §5.
# No veraPDF source was copied.
"""Spatial text grouping: chunks -> lines -> blocks -> columns.

Each container derives its bounding box from its children (union), so geometry
stays consistent as the grouping processors build the hierarchy bottom-up.
Containers may be empty during construction; geometry accessors return ``None``
until at least one child is present.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import TextChunk


class TextAlignment(str, Enum):
    """Horizontal alignment of a text line relative to its containing column/page.

    Computed from the line's bounding box geometry and the available page width.
    Used as a heading-detection signal: short centered lines are strong title
    candidates even when they carry no section-number prefix.
    """

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFIED = "justified"
    UNKNOWN = "unknown"


# Tolerance thresholds for alignment classification.
# A line whose left/right margins are within ALIGNMENT_MARGIN_RATIO of each
# other (relative to page width) is considered CENTER.
_CENTER_MARGIN_RATIO_TOL: float = 0.08  # margins must agree within 8% of page width
# A line is "short" (not spanning the full column) if it is narrower than this
# fraction of the page width.  Short + centered => strong heading signal.
_SHORT_LINE_WIDTH_RATIO: float = 0.70   # shorter than 70% of page width = "short"


def _compute_alignment(
    line: "TextLine",
    page_width: float,
) -> TextAlignment:
    """Compute the horizontal alignment of *line* within a page of *page_width*.

    Algorithm:
    - left_margin  = line.bounding_box.left
    - right_margin = page_width - line.bounding_box.right
    - If |left_margin - right_margin| <= TOLERANCE * page_width -> CENTER
    - elif left_margin < right_margin -> LEFT
    - elif right_margin < left_margin -> RIGHT
    - A full-width (or nearly full-width) line at the body font size is JUSTIFIED;
      we classify it as LEFT (justified is not a heading signal, so we treat it
      conservatively as LEFT for downstream use).

    Requires the page_width to be > 0; returns UNKNOWN otherwise.
    """
    if page_width <= 0.0:
        return TextAlignment.UNKNOWN
    bb = line.bounding_box
    if bb is None:
        return TextAlignment.UNKNOWN

    left_margin = bb.left
    right_margin = page_width - bb.right

    # Clamp negative margins (chunks that spill outside page bounds in some PDFs)
    left_margin = max(left_margin, 0.0)
    right_margin = max(right_margin, 0.0)

    margin_diff = abs(left_margin - right_margin)
    tol = _CENTER_MARGIN_RATIO_TOL * page_width

    if margin_diff <= tol:
        return TextAlignment.CENTER
    if left_margin <= right_margin:
        return TextAlignment.LEFT
    return TextAlignment.RIGHT


def _union_of(boxes: Iterable[BoundingBox | None]) -> BoundingBox | None:
    """Union of an iterable of (optional) boxes, skipping ``None``s."""
    acc: BoundingBox | None = None
    for box in boxes:
        if box is None:
            continue
        if acc is None:
            # Copy so we never mutate a child's box in place.
            acc = BoundingBox(**vars(box))
        else:
            acc.union(box)
    return acc


@dataclass
class TextLine:
    """A single baseline-aligned run of :class:`TextChunk`."""

    chunks: list[TextChunk] = field(default_factory=list)

    def push(self, chunk: TextChunk) -> None:
        self.chunks.append(chunk)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def bounding_box(self) -> BoundingBox | None:
        return _union_of(c.bounding_box for c in self.chunks)

    @property
    def value(self) -> str:
        return "".join(c.value for c in self.chunks)

    @property
    def base_line(self) -> float | None:
        return self.chunks[0].base_line if self.chunks else None

    @property
    def font_size(self) -> float | None:
        return self.chunks[0].font_size if self.chunks else None

    @property
    def is_space_line(self) -> bool:
        """Whether every chunk is whitespace (a spacer line)."""
        return bool(self.chunks) and all(c.is_whitespace for c in self.chunks)

    @property
    def is_blank(self) -> bool:
        """Whether the line has no chunks or all chunks are empty."""
        return all(c.is_empty for c in self.chunks)

    @property
    def is_hidden_text(self) -> bool:
        return bool(self.chunks) and all(c.hidden_text for c in self.chunks)

    def alignment(self, page_width: float) -> TextAlignment:
        """Compute this line's horizontal alignment relative to *page_width*.

        Returns :class:`TextAlignment` (CENTER / LEFT / RIGHT / UNKNOWN).
        A centered short line is a strong heading signal; see
        :func:`_compute_alignment` for the margin-based algorithm.
        """
        return _compute_alignment(self, page_width)

    def is_centered(self, page_width: float, *, require_short: bool = True) -> bool:
        """Whether the line is centered within the page.

        Args:
            page_width: The page width in points.
            require_short: When *True* (default), also requires the line to be
                shorter than ``_SHORT_LINE_WIDTH_RATIO * page_width`` so that
                full-width justified lines are not falsely classified as
                centered headings.
        """
        if self.alignment(page_width) != TextAlignment.CENTER:
            return False
        if require_short:
            bb = self.bounding_box
            if bb is None:
                return False
            line_width = bb.right - bb.left
            return line_width < _SHORT_LINE_WIDTH_RATIO * page_width
        return True


@dataclass
class TextBlock:
    """A vertically-stacked group of :class:`TextLine` (a paragraph-sized run)."""

    lines: list[TextLine] = field(default_factory=list)

    def push(self, line: TextLine) -> None:
        self.lines.append(line)

    @property
    def is_empty(self) -> bool:
        return not self.lines

    @property
    def first_line(self) -> TextLine | None:
        return self.lines[0] if self.lines else None

    @property
    def first_non_space_line(self) -> TextLine | None:
        """First line that is not a space line, for font/heading heuristics."""
        return next((line for line in self.lines if not line.is_space_line), None)

    @property
    def bounding_box(self) -> BoundingBox | None:
        return _union_of(line.bounding_box for line in self.lines)

    @property
    def value(self) -> str:
        return "\n".join(line.value for line in self.lines)


@dataclass
class TextColumn:
    """A column of :class:`TextBlock` — the grouping a semantic node holds."""

    blocks: list[TextBlock] = field(default_factory=list)

    def push(self, block: TextBlock) -> None:
        self.blocks.append(block)

    @property
    def is_empty(self) -> bool:
        return not self.blocks

    @property
    def bounding_box(self) -> BoundingBox | None:
        return _union_of(block.bounding_box for block in self.blocks)

    @property
    def first_line(self) -> TextLine | None:
        """First line across all blocks, in reading order."""
        for block in self.blocks:
            if block.first_line is not None:
                return block.first_line
        return None

    @property
    def value(self) -> str:
        return "\n".join(block.value for block in self.blocks)
