# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of the leaf-chunk entity types from veraPDF's
# ``wcag-algorithms`` model (``TextChunk``, ``ImageChunk``, ``LineArtChunk``).
# Fields and methods are reconstructed from the OpenDataLoader processor call
# sites documented in ``docs/architecture/02-pdf-parsing-layer.md`` §5 and from
# the JSON serializer field names (``org.opendataloader.pdf.json.JsonName``).
# No veraPDF source was copied.
"""Leaf content chunks: the raw output of the parsing layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from odl_pdf.entities.bounding_box import BoundingBox


@dataclass
class Chunk:
    """Geometry shared by every leaf chunk (the ``IChunk`` surface).

    A bounding box, the page it sits on, and a stable extraction index.
    """

    bounding_box: BoundingBox
    index: int = 0

    @property
    def page_number(self) -> int:
        return self.bounding_box.page_number

    @property
    def left(self) -> float:
        return self.bounding_box.left

    @property
    def right(self) -> float:
        return self.bounding_box.right

    @property
    def bottom(self) -> float:
        return self.bounding_box.bottom

    @property
    def top(self) -> float:
        return self.bounding_box.top

    @property
    def width(self) -> float:
        return self.bounding_box.width

    @property
    def height(self) -> float:
        return self.bounding_box.height


@dataclass
class TextChunk(Chunk):
    """A run of characters sharing a font, the dominant leaf type.

    ``value`` is mutated in place by the text processors (undefined-character
    replacement), matching the Java ``TextChunk.setValue()`` usage.
    """

    value: str = ""
    font_name: str = ""
    font_size: float = 0.0
    font_weight: float | None = None
    text_color: list[float] = field(default_factory=list)
    italic: bool = False
    base_line: float | None = None
    is_strikethrough: bool = False
    hidden_text: bool = False
    contrast_ratio: float | None = None

    def __post_init__(self) -> None:
        # Baseline defaults to the bbox bottom, like the Rust port.
        if self.base_line is None:
            self.base_line = self.bounding_box.bottom

    @property
    def rounded_font_weight(self) -> float | None:
        """Font weight rounded to the nearest 100, as the heading classifier
        uses it; ``None`` if no weight is set."""
        if self.font_weight is None:
            return None
        return round(self.font_weight / 100.0) * 100.0

    @property
    def is_whitespace(self) -> bool:
        """Whether the chunk is entirely whitespace (and non-empty)."""
        return bool(self.value) and self.value.isspace()

    @property
    def is_empty(self) -> bool:
        """Whether the chunk carries no characters at all."""
        return self.value == ""


@dataclass
class ImageChunk(Chunk):
    """A raster image placed on the page."""


@dataclass
class LineArtChunk(Chunk):
    """Vector line-art (paths/strokes), the input to table-border detection.

    ``is_background`` marks fills the content filter discards so they are not
    mistaken for table rules.
    """

    is_background: bool = False
