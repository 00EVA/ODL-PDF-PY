# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Advanced debug utilities for the ODL-PDF pipeline.

All helpers in this module are **zero-overhead when not needed**:

* :func:`log_kv` — emit a structured ``key=value`` line through an existing
  logger; callers are not required to format strings themselves.
* :class:`timed_stage` — decorator **and** context manager that times a
  pipeline stage and emits the elapsed milliseconds at DEBUG.  When the root
  logger is not at DEBUG the timing is skipped entirely.
* :func:`dump_page_stats` — optional per-page object-count dump; active only
  when ``ODL_TRACE=1`` (set automatically by ``configure()`` when
  ``ODL_LOG_LEVEL=TRACE``).
* :func:`run_summary` — one-line digest printed at INFO after a full pipeline
  run.

None of these alter pipeline behaviour; they are purely observational.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from odl_pdf.entities.document import Document, Page

# ---------------------------------------------------------------------------
# Module logger (sits under the odl_pdf root, same as every other module).
# ---------------------------------------------------------------------------
from odl_pdf.logging_config import TRACE, get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. Structured key=value logging
# ---------------------------------------------------------------------------


def log_kv(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log *message* with structured *fields* at DEBUG.

    Example::

        log_kv(logger, "parse", page=3, chunks=71)
        # → DEBUG odl_pdf.parser: parse page=3 chunks=71

    The call is a no-op when DEBUG is not enabled, so callers need not guard it.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    kv = " ".join(f"{k}={v!r}" for k, v in fields.items())
    logger.debug("%s %s", message, kv)


# ---------------------------------------------------------------------------
# 2. Per-stage timing
# ---------------------------------------------------------------------------

class timed_stage:
    """Time a pipeline stage and log the elapsed milliseconds at DEBUG.

    Can be used as a **decorator** or as a **context manager**.  When the root
    ``odl_pdf`` logger is not at DEBUG the timer is still measured but the
    log line is suppressed, keeping the overhead to a single
    ``logger.isEnabledFor`` check per stage entry.

    ``ODL_LOG_TIMING=1`` forces the timing line to be logged regardless of
    level (useful when you want timing data without the full DEBUG flood).

    Decorator usage::

        @timed_stage("group_lines")
        def group_lines(chunks): ...

    Context-manager usage::

        with timed_stage("detect_tables"):
            tables = detect_bordered_tables(page)
    """

    def __init__(self, name: str, logger: logging.Logger | None = None) -> None:
        self.name = name
        self._logger = logger or _log
        self._start: float = 0.0

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "timed_stage":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._emit(elapsed_ms)

    # ------------------------------------------------------------------
    # Decorator protocol
    # ------------------------------------------------------------------

    def __call__(self, func):  # type: ignore[override]
        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[return]
            with self.__class__(self.name, self._logger):
                return func(*args, **kwargs)
        return wrapper

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, elapsed_ms: float) -> None:
        force = os.environ.get("ODL_LOG_TIMING", "").strip() == "1"
        if force:
            # Emit at INFO so the line is visible even when DEBUG is off.
            self._logger.info("stage %r took %.1f ms [timing]", self.name, elapsed_ms)
        elif self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("stage %r took %.1f ms", self.name, elapsed_ms)


# ---------------------------------------------------------------------------
# 3. TRACE-mode per-page stats dump
# ---------------------------------------------------------------------------

def _trace_active() -> bool:
    """Return True when TRACE mode is active (ODL_TRACE=1 or root level <= TRACE)."""
    if os.environ.get("ODL_TRACE", "").strip() == "1":
        return True
    root = logging.getLogger("odl_pdf")
    return root.isEnabledFor(TRACE)


def dump_page_stats(page: "Page", kids: list[Any]) -> None:
    """Log per-page object counts at TRACE level.

    Called from the pipeline (opt-in: only when ``ODL_LOG_LEVEL=TRACE`` or
    ``ODL_TRACE=1``).  Does nothing when TRACE is inactive.

    Fields logged:

    * ``page``    — 1-based page number
    * ``chunks``  — raw text chunks extracted by the parser
    * ``lines``   — text lines after line-grouping (proxy: chunk count)
    * ``nodes``   — semantic nodes (paragraphs/headings/…) in the page kids
    * ``tables``  — table objects in the page kids
    * ``lists``   — list objects in the page kids
    """
    if not _trace_active():
        return

    from odl_pdf.entities.semantic import SemanticType
    from odl_pdf.entities.table import TableBorder
    from odl_pdf.entities.list_entity import PDFList

    n_nodes = sum(1 for k in kids
                  if hasattr(k, "semantic_type") and k.semantic_type in (
                      SemanticType.PARAGRAPH, SemanticType.HEADING,
                      SemanticType.HEADER, SemanticType.FOOTER,
                      SemanticType.CAPTION,
                  ))
    n_tables = sum(1 for k in kids if isinstance(k, TableBorder))
    n_lists = sum(1 for k in kids if isinstance(k, PDFList))

    _log.log(
        TRACE,
        "trace page=%d chunks=%d nodes=%d tables=%d lists=%d",
        page.page_number + 1,
        len(page.text_chunks),
        n_nodes,
        n_tables,
        n_lists,
    )


# ---------------------------------------------------------------------------
# 4. Run summary
# ---------------------------------------------------------------------------

def run_summary(document: "Document", elapsed_s: float | None = None) -> None:
    """Log a one-line digest at INFO after a complete pipeline run.

    Example output::

        [summary] lorem.pdf — pages=1 chunks=3 nodes=2 tables=0 headings=1 elapsed=0.24s

    Parameters
    ----------
    document:
        The :class:`~odl_pdf.entities.document.Document` returned by
        :func:`odl_pdf.pipeline.extract`.
    elapsed_s:
        Wall-clock seconds for the full pipeline run (optional).  When not
        supplied the ``[elapsed=…]`` field is omitted.
    """
    from odl_pdf.entities.semantic import SemanticType
    from odl_pdf.entities.table import TableBorder

    pages = document.number_of_pages
    total_chunks = sum(len(p.text_chunks) for p in document.pages)
    all_kids: list[Any] = []
    for p in document.pages:
        all_kids.extend(getattr(p, "_kids", []))

    n_nodes = sum(
        1 for k in all_kids
        if hasattr(k, "semantic_type")
        and k.semantic_type not in (SemanticType.LIST, SemanticType.TABLE)
    )
    n_tables = sum(1 for k in all_kids if isinstance(k, TableBorder))
    n_headings = sum(
        1 for k in all_kids
        if hasattr(k, "semantic_type") and k.semantic_type is SemanticType.HEADING
    )

    elapsed_part = f" elapsed={elapsed_s:.2f}s" if elapsed_s is not None else ""
    _log.info(
        "[summary] %s — pages=%d chunks=%d nodes=%d tables=%d headings=%d%s",
        document.metadata.file_name or "<unknown>",
        pages,
        total_chunks,
        n_nodes,
        n_tables,
        n_headings,
        elapsed_part,
    )
