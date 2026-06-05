# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for the document/page model (mirror of Rust)."""

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import ImageChunk, LineArtChunk, TextChunk
from odl_pdf.entities.document import Document, DocumentMetadata, Page


def test_page_partitions_chunks_by_kind():
    page = Page(0, 595.0, 842.0)
    page.push_text(TextChunk(BoundingBox.of(0, 0.0, 0.0, 10.0, 12.0), value="Hi"))
    page.push_image(ImageChunk(BoundingBox.of(0, 0.0, 0.0, 50.0, 50.0)))
    page.push_line_art(LineArtChunk(BoundingBox.of(0, 0.0, 0.0, 100.0, 1.0)))
    assert len(page.text_chunks) == 1
    assert len(page.image_chunks) == 1
    assert len(page.line_art_chunks) == 1
    assert page.text == "Hi"
    assert (page.width, page.height) == (595.0, 842.0)


def test_document_tracks_pages_and_metadata():
    meta = DocumentMetadata(file_name="lorem.pdf", title="Lorem")
    doc = Document(meta)
    doc.push_page(Page(0, 595.0, 842.0))
    doc.push_page(Page(1, 595.0, 842.0))
    assert doc.number_of_pages == 2
    assert doc.metadata.file_name == "lorem.pdf"
    assert doc.metadata.author is None
