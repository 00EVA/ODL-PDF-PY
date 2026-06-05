# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Parsing layer (Track B): native PDF libs -> entity chunks."""

from odl_pdf.parser.pdf_parser import PdfParser, parse_pdf

__all__ = ["PdfParser", "parse_pdf"]
