# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Hybrid AI mode: route hard pages to an AI vision backend.

The local parser stays the primary, deterministic path. Triage classifies each
page as LOCAL or BACKEND; BACKEND pages are sent to a pluggable
:class:`BackendAdapter` (e.g. AWS Bedrock vision) and merged back into the
:class:`~odl_pdf.entities.Document`.

See ``docs/architecture/08-hybrid-ai-mode.md`` and PRD §5.6 / §9.
"""

from odl_pdf.hybrid.adapter import BackendAdapter, BackendError, BackendRegistry
from odl_pdf.hybrid.config import HybridConfig, HybridMode, OcrStrategy, TriageDecision
from odl_pdf.hybrid.orchestrator import HybridOrchestrator, HybridResult
from odl_pdf.hybrid.triage import TriageProcessor, TriageResult, TriageSignals

# Importing backends registers the "mock" and "bedrock-claude" adapters.
from odl_pdf.hybrid import backends as _backends  # noqa: E402,F401

__all__ = [
    "BackendAdapter",
    "BackendError",
    "BackendRegistry",
    "HybridConfig",
    "HybridMode",
    "OcrStrategy",
    "TriageDecision",
    "HybridOrchestrator",
    "HybridResult",
    "TriageProcessor",
    "TriageResult",
    "TriageSignals",
]
