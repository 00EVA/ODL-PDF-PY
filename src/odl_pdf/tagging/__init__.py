# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Auto-tagging: untagged PDF -> tagged PDF (PDF/UA structure tree).

Built on pikepdf (the PRD-recommended primary for the write path). See
docs/architecture/06-auto-tagging.md.
"""

from odl_pdf.tagging.auto_tagger import TaggingResult, tag_pdf
from odl_pdf.tagging.struct_tree import (
    build_normalized_heading_levels,
    build_struct_tree,
    print_struct_tree,
)

__all__ = [
    "tag_pdf",
    "TaggingResult",
    "build_struct_tree",
    "build_normalized_heading_levels",
    "print_struct_tree",
]
