# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""End-to-end pipeline integration tests, checked against the JAR oracle shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from odl_pdf.output.json_writer import write_document_json
from odl_pdf.pipeline import extract

ROOT = Path(__file__).resolve().parents[2]
LOREM = ROOT / "opendataloader-pdf/samples/pdf/lorem.pdf"


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_lorem_pipeline_matches_oracle_structure():
    doc = extract(LOREM)
    out = json.loads(write_document_json(doc))

    # Document envelope.
    assert out["file name"] == "lorem.pdf"
    assert out["number of pages"] == 1
    assert out["author"] == "leebd-public"

    kids = out["kids"]
    assert len(kids) == 2, "expected a heading + a paragraph"

    heading, paragraph = kids[0], kids[1]
    assert heading["type"] == "heading"
    assert heading["pdfua_tag"] == "H1"
    assert heading["heading level"] == 1
    # Font sizes must match the oracle exactly (parser size-recovery fix).
    assert heading["font size"] == 32.005
    assert "Lorem" in heading["content"] and "Ipsum" in heading["content"]

    assert paragraph["type"] == "paragraph"
    assert paragraph["pdfua_tag"] == "P"
    assert paragraph["font size"] == 9.949
    assert "consectetur adipiscing elit" in paragraph["content"]

    # Bounding-box left/bottom origin matches the oracle (x within a point).
    assert abs(heading["bounding box"][0] - 200.891) < 0.01


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_pipeline_attaches_ordered_kids_to_pages():
    doc = extract(LOREM)
    for page in doc.pages:
        assert hasattr(page, "_kids")
        assert isinstance(page._kids, list)
