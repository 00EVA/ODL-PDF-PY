# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Triage: decide per page whether the local engine can handle it or it should
# go to the AI backend. Signals, thresholds, priority order, and the two
# DISABLED signals (with their experiment provenance) are ported from
# docs/architecture/08-hybrid-ai-mode.md §4 — the authoritative description of
# TriageProcessor.classifyPage. No veraPDF/Java source copied; this is a
# clean-room reimplementation of the documented behavior operating on our own
# Page/chunk model.
"""Per-page triage classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

from odl_pdf.entities import LineArtChunk, Page
from odl_pdf.hybrid.config import TriageDecision
from odl_pdf.logging_config import get_logger

logger = get_logger(__name__)

# --- thresholds (from the architecture doc's signal table) ---
CID_REPLACEMENT_RATIO = 0.30  # priority 0
GRID_LINE_MIN = 3  # horizontal AND vertical line count for hasGridLines
TABLE_BORDER_LINE_MIN = 8  # total h+v lines for hasTableBorderLines
LINE_ART_MIN = 8  # lineArtCount for the vector signal
LINE_TO_TEXT_RATIO = 0.30  # priority 5
LARGE_IMAGE_AREA_FRAC = 0.11  # priority 3.5: area >= 11% of page
LARGE_IMAGE_ASPECT = 1.75  # priority 3.5: aspect >= 1.75


@dataclass
class TriageSignals:
    """Every raw measurement feeding the decision (snapshot, for triage.json)."""

    replacement_char_ratio: float = 0.0
    line_art_count: int = 0
    horizontal_lines: int = 0
    vertical_lines: int = 0
    text_chunk_count: int = 0
    largest_image_area_frac: float = 0.0
    largest_image_aspect: float = 0.0

    # --- derived vector-table sub-signals ---
    @property
    def has_grid_lines(self) -> bool:
        return self.horizontal_lines >= GRID_LINE_MIN and self.vertical_lines >= GRID_LINE_MIN

    @property
    def has_table_border_lines(self) -> bool:
        return (self.horizontal_lines + self.vertical_lines) >= TABLE_BORDER_LINE_MIN

    @property
    def has_vector_table_signal(self) -> bool:
        return (
            self.has_grid_lines
            or self.has_table_border_lines
            or self.line_art_count >= LINE_ART_MIN
        )

    @property
    def has_large_image(self) -> bool:
        return (
            self.largest_image_area_frac >= LARGE_IMAGE_AREA_FRAC
            and self.largest_image_aspect >= LARGE_IMAGE_ASPECT
        )

    @property
    def line_to_text_ratio(self) -> float:
        total = self.line_art_count + self.text_chunk_count
        return self.line_art_count / total if total else 0.0


@dataclass
class TriageResult:
    """A page's routing decision plus why."""

    page_number: int
    decision: TriageDecision
    confidence: float
    reason: str
    signals: TriageSignals = field(default_factory=TriageSignals)

    def to_dict(self) -> dict:
        s = self.signals
        return {
            "page": self.page_number + 1,  # 1-indexed, like the oracle
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "signals": {
                "replacement_char_ratio": round(s.replacement_char_ratio, 4),
                "line_art_count": s.line_art_count,
                "horizontal_lines": s.horizontal_lines,
                "vertical_lines": s.vertical_lines,
                "text_chunk_count": s.text_chunk_count,
                "largest_image_area_frac": round(s.largest_image_area_frac, 4),
            },
        }


class TriageProcessor:
    """Classifies pages as LOCAL or BACKEND.

    Signals are evaluated in priority order and short-circuit on the first
    match, exactly as documented for ``TriageProcessor.classifyPage``. Two
    signals are intentionally DISABLED (kept here as comments with their
    experiment provenance) because they caused high false-positive rates on the
    200-PDF experiments.
    """

    def __init__(self, replacement_ratio_by_page: dict[int, float] | None = None) -> None:
        # CID replacement ratio is computed by the parser/decoder; injected here.
        self._repl = replacement_ratio_by_page or {}

    def classify_page(self, page: Page) -> TriageResult:
        signals = self._gather_signals(page)
        pn = page.page_number

        def result(decision: TriageDecision, conf: float, reason: str) -> TriageResult:
            logger.debug(
                "triage page %d -> %s (%.2f) %s", pn + 1, decision.value, conf, reason
            )
            return TriageResult(pn, decision, conf, reason, signals)

        # Priority 0 — CID font extraction failure (garbage text).
        if signals.replacement_char_ratio >= CID_REPLACEMENT_RATIO:
            return result(TriageDecision.BACKEND, 1.0,
                          f"cid-failure (repl={signals.replacement_char_ratio:.2f})")

        # Priority 1 — a table border was already found locally.
        #   (Wired once TableBorderProcessor lands; for now subsumed by the
        #    vector signal below, which uses the same line-art evidence.)

        # Priority 2 — vector-graphics table signal.
        if signals.has_vector_table_signal:
            return result(TriageDecision.BACKEND, 0.95, "vector-table-signal")

        # Priority 3 — text-pattern table signal (needs grouping; deferred).

        # Priority 3.5 — large image (table/chart screenshot). Exp 005.
        if signals.has_large_image:
            return result(TriageDecision.BACKEND, 0.85, "large-image")

        # Priority 4 — DISABLED: suspicious text gap.
        #   Experiment 003D: 28.4% false-positive rate. Shipped disabled.

        # Priority 5 — high line-art-to-text ratio.
        if signals.line_to_text_ratio > LINE_TO_TEXT_RATIO:
            return result(TriageDecision.BACKEND, 0.80,
                          f"line-ratio={signals.line_to_text_ratio:.2f}")

        # Priority 6 — DISABLED: grid alignment.
        #   Experiment 004D: 21.8% FP rate, zero true positives. Shipped disabled.

        # Default — plain text page stays local.
        return result(TriageDecision.LOCAL, 0.90, "text-only")

    def _gather_signals(self, page: Page) -> TriageSignals:
        s = TriageSignals()
        s.replacement_char_ratio = self._compute_replacement_ratio(page)
        s.text_chunk_count = len(page.text_chunks)
        s.line_art_count = len(page.line_art_chunks)
        s.horizontal_lines, s.vertical_lines = _count_oriented_lines(page.line_art_chunks)

        page_area = page.width * page.height
        if page_area > 0 and page.image_chunks:
            biggest = max(page.image_chunks, key=lambda c: c.width * c.height)
            s.largest_image_area_frac = (biggest.width * biggest.height) / page_area
            short = min(biggest.width, biggest.height)
            s.largest_image_aspect = (max(biggest.width, biggest.height) / short) if short else 0.0
        return s

    def _compute_replacement_ratio(self, page: Page) -> float:
        if page.page_number in self._repl:
            return self._repl[page.page_number]
        # Fall back to measuring U+FFFD density in the extracted text.
        text = page.text
        if not text:
            return 0.0
        return text.count("�") / len(text)


def _count_oriented_lines(line_art: list[LineArtChunk]) -> tuple[int, int]:
    """Count horizontal vs vertical line-art chunks by aspect of their bbox.

    A near-zero-height box is a horizontal rule; near-zero-width is vertical.
    Uses a small absolute tolerance so hairline rules with sub-point thickness
    still classify.
    """
    h = v = 0
    for chunk in line_art:
        w, ht = chunk.width, chunk.height
        if ht <= 2.0 and w > ht:
            h += 1
        elif w <= 2.0 and ht > w:
            v += 1
    return h, v
