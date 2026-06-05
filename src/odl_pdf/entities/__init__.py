# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Entity and geometry model.

Clean-room port of the parts of veraPDF's ``wcag-algorithms`` entity layer that
the OpenDataLoader processors operate on directly (see
``docs/architecture/02-pdf-parsing-layer.md`` §6.2). No parsing logic lives
here — the byte layer is supplied by pikepdf/pypdf (Track B).
"""

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import Chunk, ImageChunk, LineArtChunk, TextChunk
from odl_pdf.entities.document import Document, DocumentMetadata, Page
from odl_pdf.entities.list_entity import ListItem, PDFList
from odl_pdf.entities.object import IObject, union_objects
from odl_pdf.entities.semantic import SemanticTextNode, SemanticType
from odl_pdf.entities.table import TableBorder, TableBorderCell, TableBorderRow
from odl_pdf.entities.text import TextAlignment, TextBlock, TextColumn, TextLine

__all__ = [
    "BoundingBox",
    "Chunk",
    "ImageChunk",
    "LineArtChunk",
    "TextChunk",
    "TextAlignment",
    "TextBlock",
    "TextColumn",
    "TextLine",
    "SemanticTextNode",
    "SemanticType",
    "IObject",
    "union_objects",
    "TableBorder",
    "TableBorderCell",
    "TableBorderRow",
    "ListItem",
    "PDFList",
    "Document",
    "DocumentMetadata",
    "Page",
]
