# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Hybrid orchestrator: the rewrite's equivalent of HybridDocumentProcessor
# (docs/architecture/08-hybrid-ai-mode.md §3). Local path stays primary; triage
# routes hard pages to the backend; results merge per page. Cost/latency guard
# (backend_page_limit) and per-page fallback to local are honored. Heavy logging
# + an emitted triage report give the feedback/debugging loop the user asked for.
"""End-to-end hybrid pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from odl_pdf.entities import Document, IObject
from odl_pdf.hybrid.adapter import BackendAdapter, BackendError, BackendRegistry
from odl_pdf.hybrid.config import HybridConfig, HybridMode, TriageDecision
from odl_pdf.hybrid.render import render_page_png
from odl_pdf.hybrid.triage import TriageProcessor, TriageResult
from odl_pdf.logging_config import get_logger
from odl_pdf.parser import parse_pdf

logger = get_logger(__name__)


@dataclass
class HybridResult:
    """Outcome of a hybrid run: the parsed document, per-page routing, and the
    backend-produced content keyed by page number."""

    document: Document
    triage: list[TriageResult] = field(default_factory=list)
    backend_objects: dict[int, list[IObject]] = field(default_factory=dict)
    backend_pages: list[int] = field(default_factory=list)
    local_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)

    def triage_report(self) -> dict:
        """Serializable triage report (mirrors Java's triage.json)."""
        return {
            "file": self.document.metadata.file_name,
            "pages": self.document.number_of_pages,
            "triage": [t.to_dict() for t in self.triage],
            "summary": {
                "total_pages": self.document.number_of_pages,
                "local_pages": len(self.local_pages),
                "backend_pages": len(self.backend_pages),
                "failed_pages": len(self.failed_pages),
            },
        }


class HybridOrchestrator:
    """Runs the local parser, triages pages, and routes hard pages to a backend."""

    def __init__(self, config: HybridConfig, backend: BackendAdapter | None = None) -> None:
        self.config = config
        self._backend = backend

    def _resolve_backend(self) -> BackendAdapter:
        if self._backend is not None:
            return self._backend
        if not self.config.backend:
            raise BackendError("hybrid enabled but no backend configured")
        self._backend = BackendRegistry.create(self.config.backend, self.config)
        return self._backend

    def process(self, pdf_path: str | Path) -> HybridResult:
        pdf_path = Path(pdf_path)
        logger.info("hybrid: parsing %s (mode=%s)", pdf_path.name, self.config.mode.value)
        document = parse_pdf(pdf_path)
        result = HybridResult(document=document)

        if not self.config.enabled:
            logger.info("hybrid disabled; all %d page(s) local", document.number_of_pages)
            result.local_pages = [p.page_number for p in document.pages]
            return result

        # --- triage (or route everything to backend in FULL mode) ---
        triage = TriageProcessor()
        backend_targets: list[int] = []
        for page in document.pages:
            if self.config.mode is HybridMode.FULL:
                tr = TriageResult(page.page_number, TriageDecision.BACKEND, 1.0, "mode=full")
            else:
                tr = triage.classify_page(page)
            result.triage.append(tr)
            if tr.decision is TriageDecision.BACKEND:
                backend_targets.append(page.page_number)

        # --- cost/latency cap (PRD §9): keep the highest-confidence pages ---
        if len(backend_targets) > self.config.backend_page_limit:
            ranked = sorted(
                backend_targets,
                key=lambda pn: result.triage[pn].confidence,
                reverse=True,
            )
            kept = set(ranked[: self.config.backend_page_limit])
            dropped = [pn for pn in backend_targets if pn not in kept]
            logger.warning(
                "backend page cap %d hit: routing %d page(s) to backend, "
                "keeping %d on local (NOT silently dropped)",
                self.config.backend_page_limit, len(kept), len(dropped),
            )
            backend_targets = [pn for pn in backend_targets if pn in kept]

        # --- run the backend on the selected pages ---
        backend = None
        if backend_targets:
            try:
                backend = self._resolve_backend()
                backend.check_availability()
            except BackendError as e:
                if self.config.fallback_to_local:
                    logger.warning("backend unavailable (%s); all pages fall back to local", e)
                    backend_targets = []
                else:
                    raise

        for pn in backend_targets:
            page = document.pages[pn]
            try:
                png = render_page_png(pdf_path, pn, self.config.render_dpi)
                objs = backend.convert_page(pn, png, page.width, page.height)
                result.backend_objects[pn] = objs
                result.backend_pages.append(pn)
                logger.info("page %d processed by backend -> %d object(s)", pn + 1, len(objs))
            except BackendError as e:
                logger.warning("backend failed on page %d: %s", pn + 1, e)
                if self.config.fallback_to_local:
                    result.failed_pages.append(pn)
                else:
                    raise
            except Exception:  # noqa: BLE001
                logger.warning("render/convert error on page %d", pn + 1, exc_info=True)
                result.failed_pages.append(pn)

        backend_set = set(result.backend_pages)
        result.local_pages = [p.page_number for p in document.pages if p.page_number not in backend_set]
        logger.info(
            "hybrid done: %d local, %d backend, %d failed",
            len(result.local_pages), len(result.backend_pages), len(result.failed_pages),
        )
        return result

    def process_and_report(self, pdf_path: str | Path, report_path: str | Path) -> HybridResult:
        """Process and write the triage report JSON to ``report_path``."""
        result = self.process(pdf_path)
        Path(report_path).write_text(json.dumps(result.triage_report(), indent=2))
        logger.info("triage report -> %s", report_path)
        return result
