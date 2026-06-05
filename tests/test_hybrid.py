# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Hybrid orchestration tests.

The mock backend tests run offline and deterministically. The live Bedrock test
is opt-in (ODL_TEST_BEDROCK=1) so the default suite needs no network/credentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from odl_pdf.entities import SemanticTextNode, TableBorder
from odl_pdf.hybrid import (
    BackendRegistry,
    HybridConfig,
    HybridMode,
    HybridOrchestrator,
    TriageDecision,
)
from odl_pdf.hybrid.backends import _parse_json_elements, _salvage_elements

ROOT = Path(__file__).resolve().parents[2]
BIALETTI = ROOT / "opendataloader-pdf/samples/pdf/issue-336-conto-economico-bialetti.pdf"
LOREM = ROOT / "opendataloader-pdf/samples/pdf/lorem.pdf"


def test_backends_are_registered():
    assert "mock" in BackendRegistry.supported()
    assert "bedrock-claude" in BackendRegistry.supported()


def test_disabled_mode_keeps_all_local():
    cfg = HybridConfig(mode=HybridMode.OFF)
    res = HybridOrchestrator(cfg).process(LOREM)
    assert res.backend_pages == []
    assert len(res.local_pages) == res.document.number_of_pages


@pytest.mark.skipif(not BIALETTI.exists(), reason="bialetti sample not present")
def test_mock_backend_processes_table_pages():
    cfg = HybridConfig(mode=HybridMode.AUTO, backend="mock")
    res = HybridOrchestrator(cfg).process(BIALETTI)
    # Both financial-table pages should route to the backend.
    assert len(res.backend_pages) == 2
    assert res.failed_pages == []
    for objs in res.backend_objects.values():
        types = {type(o).__name__ for o in objs}
        assert "TableBorder" in types
        assert "SemanticTextNode" in types


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_full_mode_routes_every_page_to_backend():
    cfg = HybridConfig(mode=HybridMode.FULL, backend="mock")
    res = HybridOrchestrator(cfg).process(LOREM)
    assert len(res.backend_pages) == res.document.number_of_pages
    assert all(t.reason == "mode=full" for t in res.triage)


@pytest.mark.skipif(not BIALETTI.exists(), reason="bialetti sample not present")
def test_page_cap_does_not_silently_drop(tmp_path):
    cfg = HybridConfig(mode=HybridMode.AUTO, backend="mock", backend_page_limit=1)
    res = HybridOrchestrator(cfg).process(BIALETTI)
    # Cap of 1: exactly one page to backend, the rest stay LOCAL (not dropped).
    assert len(res.backend_pages) == 1
    assert len(res.local_pages) == res.document.number_of_pages - 1


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_triage_report_written(tmp_path):
    cfg = HybridConfig(mode=HybridMode.AUTO, backend="mock")
    report = tmp_path / "triage.json"
    HybridOrchestrator(cfg).process_and_report(LOREM, report)
    data = json.loads(report.read_text())
    assert "triage" in data and "summary" in data
    assert data["summary"]["total_pages"] == data["pages"]


def test_unavailable_backend_falls_back_when_configured(monkeypatch):
    # A backend whose check_availability raises -> with fallback, pages stay local.
    from odl_pdf.hybrid.adapter import BackendError

    class Broken:
        name = "broken"

        def check_availability(self):
            raise BackendError("nope")

        def convert_page(self, *a):
            raise BackendError("nope")

    cfg = HybridConfig(mode=HybridMode.FULL, backend="broken", fallback_to_local=True)
    res = HybridOrchestrator(cfg, backend=Broken()).process(LOREM)
    assert res.backend_pages == []
    assert len(res.local_pages) == res.document.number_of_pages


# --- JSON salvage (truncation resilience) ---
def test_salvage_recovers_complete_elements_from_truncated_json():
    truncated = (
        '{"elements": ['
        '{"type":"heading","text":"Title","level":1,"bbox":[0,0,10,10]},'
        '{"type":"paragraph","text":"Body","bbox":[0,0,10,10]},'
        '{"type":"paragraph","text":"cut off he'  # truncated mid-object
    )
    els = _parse_json_elements(truncated, 0)
    assert len(els) == 2
    assert els[0]["type"] == "heading"
    assert els[1]["text"] == "Body"


def test_parse_handles_markdown_fences():
    fenced = '```json\n{"elements": [{"type":"paragraph","text":"x","bbox":[0,0,1,1]}]}\n```'
    els = _parse_json_elements(fenced, 0)
    assert len(els) == 1 and els[0]["text"] == "x"


# --- live Bedrock (opt-in) ---
@pytest.mark.skipif(
    os.environ.get("ODL_TEST_BEDROCK") != "1" or not BIALETTI.exists(),
    reason="set ODL_TEST_BEDROCK=1 to run the live Bedrock vision test",
)
def test_bedrock_live_extracts_structure():
    cfg = HybridConfig(
        mode=HybridMode.FULL, backend="bedrock-claude", backend_page_limit=1, render_dpi=120
    )
    res = HybridOrchestrator(cfg).process(BIALETTI)
    assert len(res.backend_pages) == 1
    objs = next(iter(res.backend_objects.values()))
    assert objs, "Bedrock returned no objects"
    assert any(isinstance(o, (SemanticTextNode, TableBorder)) for o in objs)
