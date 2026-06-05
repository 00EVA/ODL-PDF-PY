# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Tests for advanced console-logging and debug utilities.

Covers:
* Colour auto-disable when not a TTY (and when NO_COLOR is set).
* :func:`timed_stage` logs at DEBUG only (not at INFO).
* :func:`log_kv` output format.
* :func:`configure` idempotency (no double-handler install).
* :func:`run_summary` happy path.
* TRACE level wiring.
* ``ODL_LOG_TIMING`` env var triggers timing output regardless of level.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from unittest.mock import patch

import pytest

from odl_pdf.logging_config import (
    TRACE,
    _ColourFormatter,
    _want_colour,
    configure,
    get_logger,
    reset_for_testing,
)
from odl_pdf.debug import log_kv, run_summary, timed_stage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_logging(monkeypatch):
    """Reset odl_pdf logging state before and after every test."""
    reset_for_testing()
    # Clean environment knobs so tests don't bleed into each other.
    for var in ("ODL_LOG_LEVEL", "ODL_LOG_COLOR", "ODL_LOG_TIMING",
                "ODL_LOG_FORMAT", "ODL_TRACE", "NO_COLOR"):
        monkeypatch.delenv(var, raising=False)
    yield
    reset_for_testing()


def _install_capture() -> tuple[logging.Logger, io.StringIO]:
    """Configure odl_pdf logging into a StringIO and return (root, stream)."""
    buf = io.StringIO()
    root = logging.getLogger("odl_pdf")
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.propagate = False
    return root, buf


# ---------------------------------------------------------------------------
# 1. Colour auto-disable
# ---------------------------------------------------------------------------

class TestColourAutoDisable:
    def test_disabled_when_not_a_tty(self, monkeypatch):
        """_want_colour must return False when stderr is not a TTY."""
        monkeypatch.delenv("ODL_LOG_COLOR", raising=False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        with patch.object(sys.stderr, "isatty", return_value=False):
            assert _want_colour() is False

    def test_disabled_by_no_color_env(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("ODL_LOG_COLOR", raising=False)
        with patch.object(sys.stderr, "isatty", return_value=True):
            assert _want_colour() is False

    def test_forced_off_by_odl_log_color_0(self, monkeypatch):
        monkeypatch.setenv("ODL_LOG_COLOR", "0")
        with patch.object(sys.stderr, "isatty", return_value=True):
            assert _want_colour() is False

    def test_forced_on_by_odl_log_color_1(self, monkeypatch):
        monkeypatch.setenv("ODL_LOG_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        with patch.object(sys.stderr, "isatty", return_value=False):
            assert _want_colour() is True

    def test_enabled_when_tty_and_no_overrides(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("ODL_LOG_COLOR", raising=False)
        with patch.object(sys.stderr, "isatty", return_value=True):
            assert _want_colour() is True

    def test_colour_formatter_strips_ansi_when_not_tty(self, monkeypatch):
        """_ColourFormatter must produce a plain string when colour is off."""
        monkeypatch.setenv("ODL_LOG_COLOR", "0")
        fmt = _ColourFormatter()
        record = logging.LogRecord(
            name="odl_pdf.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        text = fmt.format(record)
        assert "\033[" not in text, "ANSI codes leaked when colour disabled"

    def test_colour_formatter_includes_ansi_when_forced_on(self, monkeypatch):
        """_ColourFormatter must include ANSI codes when ODL_LOG_COLOR=1."""
        monkeypatch.setenv("ODL_LOG_COLOR", "1")
        fmt = _ColourFormatter()
        record = logging.LogRecord(
            name="odl_pdf.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        text = fmt.format(record)
        assert "\033[" in text, "ANSI codes missing when colour forced on"


# ---------------------------------------------------------------------------
# 2. timed_stage: logs at DEBUG only
# ---------------------------------------------------------------------------

class TestTimedStage:
    def test_no_log_at_info_level(self, monkeypatch):
        """timed_stage must be silent when logger is at INFO."""
        monkeypatch.delenv("ODL_LOG_TIMING", raising=False)
        root, buf = _install_capture()
        root.setLevel(logging.INFO)

        with timed_stage("my_stage"):
            pass

        assert "my_stage" not in buf.getvalue()

    def test_logs_at_debug_level(self, monkeypatch):
        """timed_stage must emit a 'stage took' line at DEBUG."""
        monkeypatch.delenv("ODL_LOG_TIMING", raising=False)
        root, buf = _install_capture()
        root.setLevel(logging.DEBUG)

        with timed_stage("my_stage"):
            pass

        assert "my_stage" in buf.getvalue()
        assert "ms" in buf.getvalue()

    def test_decorator_form_logs_at_debug(self, monkeypatch):
        monkeypatch.delenv("ODL_LOG_TIMING", raising=False)
        root, buf = _install_capture()
        root.setLevel(logging.DEBUG)

        @timed_stage("wrapped")
        def _inner():
            return 99

        result = _inner()
        assert result == 99
        assert "wrapped" in buf.getvalue()

    def test_odl_log_timing_forces_output_at_info(self, monkeypatch):
        """ODL_LOG_TIMING=1 forces timing even when logger is at INFO."""
        monkeypatch.setenv("ODL_LOG_TIMING", "1")
        root, buf = _install_capture()
        root.setLevel(logging.INFO)

        with timed_stage("forced_stage"):
            pass

        assert "forced_stage" in buf.getvalue()

    def test_elapsed_is_plausible(self, monkeypatch):
        """The elapsed time in the log line must be non-negative."""
        monkeypatch.delenv("ODL_LOG_TIMING", raising=False)
        root, buf = _install_capture()
        root.setLevel(logging.DEBUG)

        with timed_stage("slow"):
            time.sleep(0.01)

        text = buf.getvalue()
        # Extract the number before 'ms'
        import re
        m = re.search(r"([\d.]+) ms", text)
        assert m is not None, f"No 'N ms' found in: {text!r}"
        ms = float(m.group(1))
        assert ms >= 5, f"Expected >=5 ms sleep, got {ms}"


# ---------------------------------------------------------------------------
# 3. log_kv formatting
# ---------------------------------------------------------------------------

class TestLogKv:
    def test_emits_key_value_pairs(self, monkeypatch):
        root, buf = _install_capture()
        root.setLevel(logging.DEBUG)
        logger = get_logger("log_kv_test")

        log_kv(logger, "parse", page=3, chunks=71)

        text = buf.getvalue()
        assert "parse" in text
        assert "page=3" in text
        assert "chunks=71" in text

    def test_silent_at_info_level(self, monkeypatch):
        root, buf = _install_capture()
        root.setLevel(logging.INFO)
        logger = get_logger("log_kv_test")

        log_kv(logger, "parse", page=3, chunks=71)

        assert buf.getvalue() == ""

    def test_string_values_are_quoted(self, monkeypatch):
        root, buf = _install_capture()
        root.setLevel(logging.DEBUG)
        logger = get_logger("log_kv_test")

        log_kv(logger, "event", path="/tmp/x.pdf")

        text = buf.getvalue()
        assert "path='/tmp/x.pdf'" in text or 'path="/tmp/x.pdf"' in text


# ---------------------------------------------------------------------------
# 4. configure() idempotency
# ---------------------------------------------------------------------------

class TestConfigureIdempotent:
    def test_single_handler_after_multiple_calls(self):
        """Calling configure() multiple times must not add duplicate handlers."""
        configure(logging.INFO)
        configure(logging.DEBUG)
        configure(logging.INFO)

        root = logging.getLogger("odl_pdf")
        assert len(root.handlers) == 1, (
            f"Expected 1 handler after 3 configure() calls, got {len(root.handlers)}"
        )

    def test_output_not_duplicated(self):
        """A single log line must appear exactly once even after two configure calls."""
        buf = io.StringIO()
        configure(logging.INFO)
        configure(logging.INFO)  # second call — must be no-op

        root = logging.getLogger("odl_pdf")
        # Replace the installed handler's stream with our buffer.
        for h in root.handlers:
            if hasattr(h, "stream"):
                h.stream = buf

        logger = get_logger("idempotent_test")
        logger.info("sentinel message")

        lines = [l for l in buf.getvalue().splitlines() if "sentinel message" in l]
        assert len(lines) == 1, f"Expected 1 log line, got {len(lines)}: {buf.getvalue()!r}"


# ---------------------------------------------------------------------------
# 5. TRACE level
# ---------------------------------------------------------------------------

class TestTraceLevel:
    def test_trace_level_is_5(self):
        assert TRACE == 5

    def test_configure_trace_sets_odl_trace(self, monkeypatch):
        """configure('TRACE') must set ODL_TRACE=1 in the environment."""
        monkeypatch.delenv("ODL_TRACE", raising=False)
        configure("TRACE")
        assert os.environ.get("ODL_TRACE") == "1"

    def test_configure_trace_sets_root_level_to_5(self, monkeypatch):
        monkeypatch.delenv("ODL_TRACE", raising=False)
        configure("TRACE")
        root = logging.getLogger("odl_pdf")
        assert root.level == TRACE


# ---------------------------------------------------------------------------
# 6. run_summary
# ---------------------------------------------------------------------------

class TestRunSummary:
    def _make_doc(self):
        """Build a minimal Document with one page and two _kids."""
        from odl_pdf.entities.document import Document, DocumentMetadata, Page
        from odl_pdf.entities.chunk import TextChunk
        from odl_pdf.entities.bounding_box import BoundingBox
        from odl_pdf.entities.semantic import SemanticTextNode, SemanticType
        from odl_pdf.entities.text import TextColumn

        doc = Document(metadata=DocumentMetadata(file_name="test.pdf"))
        page = Page(page_number=0, width=595.0, height=842.0)
        # Add two text chunks so total_chunks == 2.
        # BoundingBox.of() is the canonical single-page constructor.
        chunk = TextChunk(
            value="Hello",
            font_size=12.0,
            bounding_box=BoundingBox.of(0, 0.0, 0.0, 100.0, 20.0),
        )
        page.push_text(chunk)
        page.push_text(chunk)

        # Add a paragraph kid
        col = TextColumn()
        kid = SemanticTextNode(semantic_type=SemanticType.PARAGRAPH, columns=[col])
        page._kids = [kid]  # type: ignore[attr-defined]
        doc.push_page(page)
        return doc

    def test_run_summary_logs_info(self):
        root, buf = _install_capture()
        root.setLevel(logging.INFO)
        doc = self._make_doc()

        run_summary(doc, elapsed_s=0.42)

        text = buf.getvalue()
        assert "[summary]" in text
        assert "test.pdf" in text
        assert "pages=1" in text
        assert "elapsed=0.42s" in text

    def test_run_summary_without_elapsed(self):
        root, buf = _install_capture()
        root.setLevel(logging.INFO)
        doc = self._make_doc()

        run_summary(doc)

        text = buf.getvalue()
        assert "[summary]" in text
        assert "elapsed" not in text
