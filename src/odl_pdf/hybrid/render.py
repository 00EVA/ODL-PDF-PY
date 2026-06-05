# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Page rasterizer for vision backends. Mirrors the role of Java's PageImageCache
# / pdf2img (docs/architecture/08-hybrid-ai-mode.md §5): render a page to PNG so
# an AI vision model can see it. Uses pypdfium2 (BSD/Apache, same PDFium engine
# the Rust path already links) — NOT pymupdf (AGPL).
"""Render PDF pages to PNG bytes for vision backends."""

from __future__ import annotations

import io
from pathlib import Path

from odl_pdf.logging_config import get_logger

logger = get_logger(__name__)


def render_page_png(pdf_path: str | Path, page_index: int, dpi: int = 150) -> bytes:
    """Render one page (0-indexed) of ``pdf_path`` to PNG bytes at ``dpi``.

    Raises ImportError if the optional ``hybrid`` extra (pypdfium2) is missing.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "page rendering needs the 'hybrid' extra: pip install 'odl-pdf[hybrid]'"
        ) from e

    scale = dpi / 72.0  # PDFium scale is relative to 72-DPI user space
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_index]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        data = buf.getvalue()
        logger.debug(
            "rendered page %d at %d dpi -> %d KB PNG (%dx%d px)",
            page_index + 1, dpi, len(data) // 1024, pil.width, pil.height,
        )
        return data
    finally:
        doc.close()
