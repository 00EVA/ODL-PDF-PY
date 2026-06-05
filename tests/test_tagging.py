# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Tests for the auto-tagging subsystem (odl_pdf.tagging).

Test categories:
  - Heading-level normalization (pure unit test, no PDF needed).
  - tag_pdf smoke test: produces a file, re-opens, has /StructTreeRoot +
    /MarkInfo, struct tree contains H1+P for lorem.
  - Edge cases: empty pages, single-page documents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
LOREM = ROOT / "opendataloader-pdf/samples/pdf/lorem.pdf"


# ---------------------------------------------------------------------------
# Heading normalization — pure unit test (no PDF required)
# ---------------------------------------------------------------------------


def test_heading_normalization_no_skip():
    """H1 → H3 must be clamped to H1 → H2 (never skip levels going down)."""
    from odl_pdf.entities import SemanticTextNode, SemanticType
    from odl_pdf.tagging.struct_tree import build_normalized_heading_levels

    def _heading(level: int) -> SemanticTextNode:
        node = SemanticTextNode(
            semantic_type=SemanticType.HEADING,
            heading_level=level,
        )
        return node

    h1 = _heading(1)
    h3 = _heading(3)
    h2 = _heading(2)

    # Single page: [H1, H3, H2]
    pages_kids = [[h1, h3, h2]]
    levels = build_normalized_heading_levels(pages_kids)

    assert levels[id(h1)] == 1, "first heading must always be H1"
    assert levels[id(h3)] == 2, "H1->H3 must become H1->H2 (no skip)"
    assert levels[id(h2)] == 1, "H3->H2 (going up by 1) from normalized H2 -> H1"


def test_heading_normalization_first_is_h1():
    """Even if the first detected heading is H3, it must be normalized to H1."""
    from odl_pdf.entities import SemanticTextNode, SemanticType
    from odl_pdf.tagging.struct_tree import build_normalized_heading_levels

    def _heading(level: int) -> SemanticTextNode:
        return SemanticTextNode(semantic_type=SemanticType.HEADING, heading_level=level)

    h3 = _heading(3)
    h5 = _heading(5)
    h3b = _heading(3)

    pages_kids = [[h3, h5, h3b]]
    levels = build_normalized_heading_levels(pages_kids)

    assert levels[id(h3)] == 1, "first heading -> H1 regardless of original level"
    assert levels[id(h5)] == 2, "H5 after H3 (now H1) -> H2 (one step deeper)"
    assert levels[id(h3b)] == 1, "H3 after H5 (now H2) going up by 2 -> H1"


def test_heading_normalization_same_level():
    """Adjacent same-level headings must stay at the same normalized level."""
    from odl_pdf.entities import SemanticTextNode, SemanticType
    from odl_pdf.tagging.struct_tree import build_normalized_heading_levels

    def _heading(level: int) -> SemanticTextNode:
        return SemanticTextNode(semantic_type=SemanticType.HEADING, heading_level=level)

    h2a = _heading(2)
    h2b = _heading(2)
    h2c = _heading(2)

    pages_kids = [[h2a, h2b, h2c]]
    levels = build_normalized_heading_levels(pages_kids)

    assert levels[id(h2a)] == 1, "first heading -> H1"
    assert levels[id(h2b)] == 1, "same level stays same"
    assert levels[id(h2c)] == 1, "same level stays same"


def test_heading_normalization_empty():
    """An empty document produces an empty normalization map."""
    from odl_pdf.tagging.struct_tree import build_normalized_heading_levels

    result = build_normalized_heading_levels([])
    assert result == {}

    result2 = build_normalized_heading_levels([[]])
    assert result2 == {}


def test_heading_normalization_multi_page():
    """Normalization is computed across all pages, not per-page."""
    from odl_pdf.entities import SemanticTextNode, SemanticType
    from odl_pdf.tagging.struct_tree import build_normalized_heading_levels

    def _heading(level: int) -> SemanticTextNode:
        return SemanticTextNode(semantic_type=SemanticType.HEADING, heading_level=level)

    h2 = _heading(2)   # page 0
    h4 = _heading(4)   # page 1 — would skip H3 without normalization

    pages_kids = [[h2], [h4]]
    levels = build_normalized_heading_levels(pages_kids)

    assert levels[id(h2)] == 1, "first heading -> H1"
    assert levels[id(h4)] == 2, "H4 after H2 (H1 after normalization) -> H2"


# ---------------------------------------------------------------------------
# tag_pdf smoke tests (require lorem.pdf)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_tag_pdf_produces_file(tmp_path):
    """tag_pdf writes an output file."""
    from odl_pdf.tagging.auto_tagger import tag_pdf

    out = tmp_path / "lorem-tagged.pdf"
    result = tag_pdf(LOREM, out)
    assert out.exists(), "output file not created"
    assert out.stat().st_size > 0, "output file is empty"
    assert result.output_path == str(out)


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_tag_pdf_reopens(tmp_path):
    """The tagged PDF can be re-opened with pikepdf without error."""
    import pikepdf

    from odl_pdf.tagging.auto_tagger import tag_pdf

    out = tmp_path / "lorem-tagged.pdf"
    tag_pdf(LOREM, out)
    # Must not raise.
    p = pikepdf.Pdf.open(out)
    p.close()


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_tag_pdf_has_struct_tree_root(tmp_path):
    """The tagged PDF catalog has /StructTreeRoot."""
    import pikepdf

    from odl_pdf.tagging.auto_tagger import tag_pdf

    out = tmp_path / "lorem-tagged.pdf"
    tag_pdf(LOREM, out)
    p = pikepdf.Pdf.open(out)
    try:
        cat = p.trailer["/Root"]
        assert "/StructTreeRoot" in cat, "/StructTreeRoot missing from catalog"
    finally:
        p.close()


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_tag_pdf_mark_info_marked_true(tmp_path):
    """The tagged PDF has /MarkInfo /Marked true."""
    import pikepdf

    from odl_pdf.tagging.auto_tagger import tag_pdf

    out = tmp_path / "lorem-tagged.pdf"
    tag_pdf(LOREM, out)
    p = pikepdf.Pdf.open(out)
    try:
        cat = p.trailer["/Root"]
        mark_info = cat.get("/MarkInfo")
        assert mark_info is not None, "/MarkInfo missing"
        assert mark_info.get("/Marked") == True, "/MarkInfo /Marked is not True"
    finally:
        p.close()


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_tag_pdf_struct_tree_has_h1_and_p(tmp_path):
    """The struct tree has a Document root containing H1 and P for lorem."""
    import pikepdf

    from odl_pdf.tagging.auto_tagger import tag_pdf

    out = tmp_path / "lorem-tagged.pdf"
    tag_pdf(LOREM, out)
    p = pikepdf.Pdf.open(out)
    try:
        cat = p.trailer["/Root"]
        root = cat["/StructTreeRoot"]

        # Document StructElem is the single child of StructTreeRoot.
        doc_elem = root.get("/K")
        assert doc_elem is not None, "/StructTreeRoot has no /K"
        assert str(doc_elem.get("/S")) == "/Document", (
            f"/StructTreeRoot /K /S is {doc_elem.get('/S')!r}, expected /Document"
        )

        # H1 and P must be children of Document.
        kids = doc_elem.get("/K")
        assert kids is not None, "Document StructElem has no /K"
        assert isinstance(kids, pikepdf.Array), "Document /K should be an array"
        kid_tags = [str(child.get("/S")) for child in kids]
        assert "/H1" in kid_tags, f"H1 not found in Document kids: {kid_tags}"
        assert "/P" in kid_tags, f"P not found in Document kids: {kid_tags}"
    finally:
        p.close()


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_tag_pdf_result_mcid_linked_false(tmp_path):
    """MCIDs are not yet linked — result.mcid_linked must be False."""
    from odl_pdf.tagging.auto_tagger import tag_pdf

    out = tmp_path / "lorem-tagged.pdf"
    result = tag_pdf(LOREM, out)
    assert result.mcid_linked is False, "mcid_linked should be False (deferred)"


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_tag_pdf_struct_tree_str_contains_tags(tmp_path):
    """The struct_tree_str in the result contains H1 and P tags."""
    from odl_pdf.tagging.auto_tagger import tag_pdf

    out = tmp_path / "lorem-tagged.pdf"
    result = tag_pdf(LOREM, out)
    assert "H1" in result.struct_tree_str, "H1 not found in tree dump"
    assert "/P" in result.struct_tree_str, "P not found in tree dump"
    assert "Document" in result.struct_tree_str, "Document not found in tree dump"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_tag_pdf_idempotent_reopen(tmp_path):
    """Calling tag_pdf twice on the same input produces valid output both times."""
    from odl_pdf.tagging.auto_tagger import tag_pdf

    out1 = tmp_path / "tagged1.pdf"
    out2 = tmp_path / "tagged2.pdf"
    r1 = tag_pdf(LOREM, out1)
    r2 = tag_pdf(LOREM, out2)
    # Both produce valid tagged PDFs.
    assert r1.total_struct_elems == r2.total_struct_elems
    assert out1.exists() and out2.exists()
