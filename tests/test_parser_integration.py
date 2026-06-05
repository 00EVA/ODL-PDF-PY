# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Integration tests: parse real sample PDFs and check against the JAR oracle.

These run only when the sample PDFs are present (the read-only Java reference
clone). They are skipped otherwise so the unit suite stays self-contained.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from odl_pdf.parser import parse_pdf

SAMPLES = Path(__file__).resolve().parents[2] / "opendataloader-pdf" / "samples" / "pdf"
LOREM = SAMPLES / "lorem.pdf"
CID = Path(__file__).resolve().parents[2] / "spike" / "cid" / "cid-german.pdf"


@pytest.mark.skipif(not LOREM.exists(), reason="sample lorem.pdf not present")
def test_lorem_matches_oracle_strings():
    doc = parse_pdf(LOREM)
    assert doc.number_of_pages == 1
    # Oracle: author "leebd-public", heading "Lorem Ipsum", paragraph text.
    assert doc.metadata.author == "leebd-public"
    text = "".join(p.text for p in doc.pages)
    assert "Lorem Ipsum" in text
    assert "consectetur adipiscing elit" in text
    page = doc.pages[0]
    assert page.text_chunks, "expected at least one text chunk"
    # Page dimensions are A4-ish (within a point of 595x841).
    assert abs(page.width - 595.0) < 2.0
    assert abs(page.height - 841.0) < 2.0


@pytest.mark.skipif(not CID.exists(), reason="cid-german.pdf spike fixture not present")
def test_cid_german_unicode_fidelity():
    # The #1 rewrite risk: CID font byte-offset decoding. Must round-trip
    # umlauts with zero replacement characters.
    doc = parse_pdf(CID)
    assert doc.number_of_pages == 21
    text = "".join(p.text for p in doc.pages)
    assert text.count("�") == 0, "no replacement characters allowed"
    assert "Grundüberlegungen" in text
    assert sum(text.count(c) for c in "üöä") > 100
