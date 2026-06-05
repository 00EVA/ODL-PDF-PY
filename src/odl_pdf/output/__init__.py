# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Output writers: JSON (oracle-exact) and Markdown (RAG/LLM)."""

from odl_pdf.output.json_writer import write_document_json
from odl_pdf.output.markdown_writer import document_to_markdown

__all__ = ["write_document_json", "document_to_markdown"]
