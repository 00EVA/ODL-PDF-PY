# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of the auto-tagging pipeline from
# ``processors/AutoTaggingProcessor.java`` and ``api/AutoTagger.java``.
# Reconstructed from the upstream opendataloader-pdf Java source
# (Apache-2.0). No veraPDF source copied.
#
# Implements phases 1-5 from docs/architecture/06-auto-tagging.md:
#   Phase 1: structure-tree root + MarkInfo
#   Phase 2: StructElem tree per semantic type
#   Phase 3: MCID + content-stream BDC/EMC injection   ← DEFERRED (see below)
#   Phase 4: content-stream rewriting                  ← DEFERRED (see below)
#   Phase 5: ParentTree number tree                    ← minimal stub (empty)
#
# MCID / content-stream marking (Phases 3-4) requires per-glyph StreamInfo
# (operator index, start/end character offsets within TJ/Tj strings) which the
# current extraction pipeline does not yet track. Full BDC/EMC injection also
# requires font-aware string splitting for CID fonts (ChunksWriter.processString
# in the Java code). Both are deferred to a follow-up implementation phase.
# A WARNING is emitted at runtime; the resulting PDF is structurally valid and
# opens correctly in all readers — it simply lacks the fine-grained content
# marking needed for strict PDF/UA-1 compliance.
"""Auto-tagging entry point: :func:`tag_pdf`.

Converts an untagged PDF to a tagged PDF with a PDF/UA structure tree by:

1. Running the full semantic extraction pipeline on the input PDF.
2. Opening the same PDF with pikepdf.
3. Building and attaching the /StructTreeRoot + /MarkInfo (see struct_tree.py).
4. Saving the result to *output_pdf_path*.

MCID / BDC-EMC content-stream marking is a known limitation — see module
docstring. The output PDF has a valid structure tree but AT tools that rely on
per-glyph MCID links (e.g. reading order in Adobe Reader) will not benefit from
the structure until Phase 3-4 is implemented.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pikepdf

from odl_pdf.logging_config import get_logger
from odl_pdf.pipeline import extract
from odl_pdf.tagging.struct_tree import build_struct_tree, print_struct_tree

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TaggingResult:
    """Summary of a :func:`tag_pdf` call.

    Attributes:
        output_path: The path the tagged PDF was saved to.
        total_struct_elems: Approximate number of StructElems under /Document.
        mcid_linked: Whether BDC/EMC MCID linking was performed (currently
            always ``False`` — see module-level MCID deferred note).
        struct_tree_str: A human-readable indented dump of the structure tree
            (useful for smoke tests and debugging).
    """

    output_path: str
    total_struct_elems: int
    mcid_linked: bool
    struct_tree_str: str


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tag_pdf(
    input_pdf_path: str | Path,
    output_pdf_path: str | Path,
) -> TaggingResult:
    """Tag an untagged PDF and write the result to *output_pdf_path*.

    Pipeline:
    a. Run :func:`odl_pdf.pipeline.extract` to get the semantic tree per page.
    b. Open the same PDF with pikepdf.
    c. Set catalog /MarkInfo << /Marked true >>.
    d. Build /StructTreeRoot with a "Document" root element.
    e. Walk each page's IObjects in reading order and emit StructElems:
       - heading  → /S H<normalizedLevel>   (H1..H6)
       - paragraph → /S P
       - PDFList   → /S L > LI > [LBL] + LBody
       - TableBorder → /S Table > TR > TH/TD  (with ColSpan/RowSpan attrs)
       - ImageChunk → /S Figure + /Alt
    f. Save to *output_pdf_path*.

    MCID/content-stream BDC-EMC injection is currently deferred (see module
    docstring). A WARNING is logged.

    Args:
        input_pdf_path: Path to the source PDF (may be tagged or untagged).
        output_pdf_path: Destination path for the tagged output.

    Returns:
        A :class:`TaggingResult` with summary information.

    Raises:
        FileNotFoundError: If *input_pdf_path* does not exist.
        pikepdf.PdfError: If the PDF is encrypted or otherwise unreadable.
    """
    input_pdf_path = Path(input_pdf_path)
    output_pdf_path = Path(output_pdf_path)

    logger.info(
        "tag_pdf: start — input=%s output=%s",
        input_pdf_path.name,
        output_pdf_path,
    )

    if not input_pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf_path}")

    # Step a — semantic extraction.
    logger.info("tag_pdf: running extraction pipeline on %s", input_pdf_path.name)
    try:
        document = extract(input_pdf_path)
    except Exception as exc:
        logger.warning(
            "tag_pdf: extraction pipeline failed for %s — %s",
            input_pdf_path.name, exc, exc_info=True,
        )
        raise

    pages_kids = [getattr(page, "_kids", []) for page in document.pages]
    total_input_objects = sum(len(kids) for kids in pages_kids)
    logger.info(
        "tag_pdf: extraction complete — %d pages, %d total IObjects",
        document.number_of_pages, total_input_objects,
    )

    # Step b — open with pikepdf.
    logger.info("tag_pdf: opening %s with pikepdf", input_pdf_path.name)
    pdf = pikepdf.Pdf.open(input_pdf_path)

    try:
        # Steps c–e — build the structure tree.
        logger.info("tag_pdf: building structure tree")
        struct_tree_root = build_struct_tree(pdf, pages_kids)

        # Count StructElems for the result summary.
        doc_k = struct_tree_root.get("/K")
        total_elems = _count_struct_elems(doc_k) if doc_k is not None else 0
        logger.info(
            "tag_pdf: structure tree complete — ~%d StructElem(s) in /Document subtree",
            total_elems,
        )

        # MCID / BDC-EMC deferred warning (Phase 3-4).
        logger.warning(
            "tag_pdf: KNOWN LIMITATION — MCIDs not linked. "
            "BDC/EMC marked-content injection into page content streams is deferred "
            "because the extraction pipeline does not yet expose per-glyph StreamInfo "
            "(operator index + character offsets). "
            "The output PDF has a valid StructTree and /MarkInfo but AT tools that rely "
            "on per-glyph MCID links for reading-order navigation will not benefit until "
            "Phase 3-4 is implemented."
        )

        # Pretty-print the tree for smoke-test output.
        tree_str = print_struct_tree(struct_tree_root)
        logger.info("tag_pdf: structure tree dump:\n%s", tree_str)

        # Step f — save.
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf.save(str(output_pdf_path))
        logger.info("tag_pdf: saved tagged PDF to %s", output_pdf_path)

    finally:
        pdf.close()

    return TaggingResult(
        output_path=str(output_pdf_path),
        total_struct_elems=total_elems,
        mcid_linked=False,
        struct_tree_str=tree_str,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_struct_elems(node: pikepdf.Object, depth: int = 0) -> int:
    """Recursively count StructElem nodes (nodes that have /S) under *node*.

    Counts the node itself if it has /S, then its children in /K.
    Terminates at depth 20 to guard against pathological documents.
    """
    if depth > 20:
        return 0
    count = 0
    try:
        if isinstance(node, pikepdf.Object) and node.get("/S") is not None:
            count += 1
        k = node.get("/K") if isinstance(node, pikepdf.Object) else None
        if k is None:
            return count
        if isinstance(k, pikepdf.Array):
            for child in k:
                try:
                    if isinstance(child, pikepdf.Object) and child.get("/S") is not None:
                        count += _count_struct_elems(child, depth + 1)
                except Exception:
                    pass
        elif isinstance(k, pikepdf.Object) and k.get("/S") is not None:
            count += _count_struct_elems(k, depth + 1)
    except Exception:
        pass
    return count
