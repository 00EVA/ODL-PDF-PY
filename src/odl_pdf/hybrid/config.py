# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Hybrid configuration value object, mirroring the Java HybridConfig fields that
# survive into the rewrite (docs/architecture/08-hybrid-ai-mode.md §13). Java's
# server-specific knobs (image-cache, save-crops, regionlist-strategy) are kept
# as fields for CLI parity but only those the rewrite acts on are wired.
"""Hybrid mode configuration and enums."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HybridMode(str, Enum):
    """How triage decides which pages go to the backend."""

    #: Disable hybrid entirely — every page stays local. The default.
    OFF = "off"
    #: Run triage; only pages it scores BACKEND are sent to the AI backend.
    AUTO = "auto"
    #: Skip triage; send every page to the backend.
    FULL = "full"


class TriageDecision(str, Enum):
    """Where a single page is routed."""

    LOCAL = "local"
    BACKEND = "backend"


class OcrStrategy(str, Enum):
    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


@dataclass
class HybridConfig:
    """All hybrid settings in one value object.

    ``backend`` names a registered :class:`~odl_pdf.hybrid.adapter.BackendAdapter`
    (e.g. ``"bedrock-claude"`` or ``"mock"``). ``mode`` gates triage.
    ``fallback_to_local`` keeps a page on the local path when the backend errors,
    instead of failing the document.
    """

    mode: HybridMode = HybridMode.OFF
    backend: str | None = None
    url: str | None = None
    timeout_ms: int = 0  # 0 = no timeout
    fallback_to_local: bool = True
    ocr_strategy: OcrStrategy = OcrStrategy.AUTO
    #: Cap on pages sent to the backend per document (cost/latency guard, PRD §9).
    backend_page_limit: int = 20
    #: DPI to render page images at for vision backends.
    render_dpi: int = 150

    @property
    def enabled(self) -> bool:
        return self.mode is not HybridMode.OFF
