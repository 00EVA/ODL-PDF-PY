# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# The container the parsing layer fills and the processors consume. Document
# metadata fields mirror the JSON writer's top-level keys (``JsonWriter`` /
# ``JsonName``): file name, number of pages, author, title, creation/
# modification date. No veraPDF source copied.
"""Document and page containers."""

from __future__ import annotations

from dataclasses import dataclass, field

from odl_pdf.entities.chunk import ImageChunk, LineArtChunk, TextChunk


@dataclass
class Page:
    """One page: its dimensions and the leaf chunks extracted from it.

    ``page_number`` is zero-based internally (the JSON writer emits
    ``page_number + 1``). Dimensions are the crop-box width/height in points.
    """

    page_number: int
    width: float
    height: float
    text_chunks: list[TextChunk] = field(default_factory=list)
    image_chunks: list[ImageChunk] = field(default_factory=list)
    line_art_chunks: list[LineArtChunk] = field(default_factory=list)

    def push_text(self, chunk: TextChunk) -> None:
        self.text_chunks.append(chunk)

    def push_image(self, chunk: ImageChunk) -> None:
        self.image_chunks.append(chunk)

    def push_line_art(self, chunk: LineArtChunk) -> None:
        self.line_art_chunks.append(chunk)

    @property
    def text(self) -> str:
        """Concatenated text of every text chunk in extraction order."""
        return "".join(c.value for c in self.text_chunks)


@dataclass
class DocumentMetadata:
    """Document-level metadata, mirroring the JSON writer's top-level fields."""

    file_name: str | None = None
    author: str | None = None
    title: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None


@dataclass
class Document:
    """A parsed document: metadata plus pages in reading order."""

    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    pages: list[Page] = field(default_factory=list)

    @property
    def number_of_pages(self) -> int:
        return len(self.pages)

    def push_page(self, page: Page) -> None:
        self.pages.append(page)
