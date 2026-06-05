# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Unit tests for the hybrid triage classifier."""

from odl_pdf.entities import BoundingBox, ImageChunk, LineArtChunk, Page, TextChunk
from odl_pdf.hybrid.config import TriageDecision
from odl_pdf.hybrid.triage import TriageProcessor


def _page(width=595.0, height=842.0) -> Page:
    return Page(0, width, height)


def _h_line(y, x0=50.0, x1=500.0) -> LineArtChunk:
    return LineArtChunk(BoundingBox.of(0, x0, y, x1, y + 1.0))


def _v_line(x, y0=50.0, y1=500.0) -> LineArtChunk:
    return LineArtChunk(BoundingBox.of(0, x, y0, x + 1.0, y1))


def _text(s="word", x=50.0, y=700.0) -> TextChunk:
    return TextChunk(BoundingBox.of(0, x, y, x + 40.0, y + 10.0), value=s)


def test_plain_text_page_stays_local():
    page = _page()
    for i in range(20):
        page.push_text(_text(y=700.0 - i * 12))
    res = TriageProcessor().classify_page(page)
    assert res.decision is TriageDecision.LOCAL
    assert res.reason == "text-only"


def test_cid_failure_routes_to_backend_highest_priority():
    page = _page()
    # Mostly replacement characters -> CID extraction failure.
    page.push_text(_text(s="����abc"))
    res = TriageProcessor().classify_page(page)
    assert res.decision is TriageDecision.BACKEND
    assert res.confidence == 1.0
    assert "cid-failure" in res.reason


def test_cid_ratio_can_be_injected():
    page = _page()
    page.push_text(_text(s="clean text only"))
    proc = TriageProcessor(replacement_ratio_by_page={0: 0.5})
    res = proc.classify_page(page)
    assert res.decision is TriageDecision.BACKEND
    assert "cid-failure" in res.reason


def test_grid_lines_trigger_vector_table_signal():
    page = _page()
    page.push_text(_text())
    for i in range(3):
        page.push_line_art(_h_line(y=200.0 + i * 30))
    for i in range(3):
        page.push_line_art(_v_line(x=100.0 + i * 80))
    res = TriageProcessor().classify_page(page)
    assert res.decision is TriageDecision.BACKEND
    assert res.reason == "vector-table-signal"
    assert res.signals.has_grid_lines


def test_many_line_art_triggers_vector_signal():
    page = _page()
    page.push_text(_text())
    for i in range(8):
        page.push_line_art(_h_line(y=100.0 + i * 20))
    res = TriageProcessor().classify_page(page)
    assert res.decision is TriageDecision.BACKEND
    assert res.signals.line_art_count == 8


def test_large_image_triggers_backend():
    page = _page()
    page.push_text(_text())
    # Image covering ~22% of the page, aspect 2.0 (>1.75) -> chart/table shot.
    big = ImageChunk(BoundingBox.of(0, 0.0, 0.0, 540.0, 270.0))
    page.push_image(big)
    res = TriageProcessor().classify_page(page)
    assert res.decision is TriageDecision.BACKEND
    assert res.reason == "large-image"


def test_high_line_to_text_ratio():
    page = _page()
    page.push_text(_text())  # 1 text chunk
    # 5 line-art (below the count-8 vector threshold) but ratio 5/6 > 0.30.
    # Use vertical+horizontal mix that does NOT form a 3x3 grid.
    for i in range(5):
        page.push_line_art(LineArtChunk(BoundingBox.of(0, 10.0, 100.0 + i, 12.0, 400.0)))
    res = TriageProcessor().classify_page(page)
    assert res.decision is TriageDecision.BACKEND
    assert "line-ratio" in res.reason


def test_result_serializes_for_triage_json():
    page = _page()
    page.push_text(_text())
    res = TriageProcessor().classify_page(page)
    d = res.to_dict()
    assert d["page"] == 1  # 1-indexed
    assert d["decision"] in ("local", "backend")
    assert "signals" in d and "line_art_count" in d["signals"]
