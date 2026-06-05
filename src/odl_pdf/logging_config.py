# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Centralized logging for the pipeline.

One logger per module, all under the ``odl_pdf`` root, so a caller can dial the
whole pipeline up or down with a single call to :func:`configure`. Critical
points (document open, per-page parse, chunk counts, recoverable errors) log at
INFO/DEBUG; anything that drops data logs at WARNING so a smoke test surfaces it.

Advanced console features (opt-in via env vars):

``ODL_LOG_LEVEL``
    Numeric or symbolic level string, e.g. ``DEBUG``, ``10``.  Defaults to
    ``INFO``.  The special value ``TRACE`` maps to ``DEBUG`` and also enables the
    per-page stat dumps in :mod:`odl_pdf.debug`.

``ODL_LOG_COLOR``
    ``0`` to disable ANSI colour codes; ``1`` to force them on even when not a
    TTY.  By default colour is enabled when stdout is a TTY and ``NO_COLOR`` is
    unset.

``ODL_LOG_TIMING``
    ``1`` to emit elapsed-time annotations from :func:`odl_pdf.debug.timed_stage`
    even when not at DEBUG level.

``ODL_LOG_FORMAT``
    ``plain`` (default) for the ANSI-colourised stream handler, or ``rich`` to
    use the ``rich`` library's ``RichHandler`` if available (falls back to
    ``plain`` when ``rich`` is not installed).
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = "odl_pdf"
_configured = False

# ---------------------------------------------------------------------------
# TRACE level: sits below DEBUG (5).
# ---------------------------------------------------------------------------
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


# ---------------------------------------------------------------------------
# ANSI colour helpers (stdlib only, no hard dependency).
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_LEVEL_COLOURS: dict[int, str] = {
    TRACE: "\033[36m",          # cyan (same as DEBUG — sub-debug)
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",   # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[35m",  # magenta
}


def _want_colour() -> bool:
    """Return True when ANSI colour codes should be emitted.

    Colour is on by default when stderr is a real TTY and ``NO_COLOR`` is not
    set.  ``ODL_LOG_COLOR=0`` forces it off; ``ODL_LOG_COLOR=1`` forces it on.
    """
    env = os.environ.get("ODL_LOG_COLOR", "").strip()
    if env == "0":
        return False
    if env == "1":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stderr.isatty()


class _ColourFormatter(logging.Formatter):
    """A :class:`logging.Formatter` that prepends ANSI level colours.

    Colour is applied only when :func:`_want_colour` returns True at the time
    the record is formatted (so the decision is deferred and testable).
    """

    _FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    _DATE = "%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(self._FMT, datefmt=self._DATE)

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        text = super().format(record)
        if _want_colour():
            colour = _LEVEL_COLOURS.get(record.levelno, "")
            if colour:
                text = f"{colour}{text}{_ANSI_RESET}"
        return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """A module logger under the ``odl_pdf`` root.

    Pass ``__name__``; it is normalized so every logger sits beneath the shared
    root regardless of how the package was imported.
    """
    leaf = name.split(".")[-1] if name != "__main__" else "main"
    return logging.getLogger(f"{_ROOT}.{leaf}")


def configure(level: int | str | None = None) -> None:
    """Install a single stream handler on the ``odl_pdf`` root logger.

    Idempotent — calling it twice will not double-log.  ``level`` defaults to
    the ``ODL_LOG_LEVEL`` env var, then INFO.  Set ``ODL_LOG_LEVEL=DEBUG`` (or
    pass ``logging.DEBUG``) for the full per-chunk trace used during smoke
    tests.

    The special level value ``"TRACE"`` maps to :data:`TRACE` (5) and sets the
    ``ODL_TRACE`` env flag so :mod:`odl_pdf.debug` page-stat dumps are active.

    New env-var knobs (all optional; existing call sites are unaffected):

    * ``ODL_LOG_COLOR``   — ``0``/``1`` to force colour off/on.
    * ``ODL_LOG_TIMING``  — ``1`` to enable timing log lines from
      :func:`odl_pdf.debug.timed_stage`.
    * ``ODL_LOG_FORMAT``  — ``plain`` (default) or ``rich`` (requires the
      ``rich`` package; silently falls back to ``plain`` when absent).
    """
    global _configured
    root = logging.getLogger(_ROOT)

    if level is None:
        level = os.environ.get("ODL_LOG_LEVEL", "INFO")

    # TRACE is a special alias: map to numeric 5 and set the trace flag.
    if isinstance(level, str) and level.upper() == "TRACE":
        os.environ.setdefault("ODL_TRACE", "1")
        level = TRACE

    root.setLevel(level)

    if not _configured:
        fmt_choice = os.environ.get("ODL_LOG_FORMAT", "plain").strip().lower()
        if fmt_choice == "rich":
            _try_install_rich_handler(root)
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(_ColourFormatter())
            root.addHandler(handler)

        root.propagate = False
        _configured = True


def _try_install_rich_handler(root: logging.Logger) -> None:
    """Attempt to attach a ``rich.logging.RichHandler``; fall back to plain."""
    try:
        from rich.logging import RichHandler  # type: ignore[import-untyped]

        handler = RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
        )
        root.addHandler(handler)
    except ImportError:
        # rich not installed — degrade gracefully to plain colour handler.
        handler = logging.StreamHandler()
        handler.setFormatter(_ColourFormatter())
        root.addHandler(handler)


def reset_for_testing() -> None:
    """Remove all handlers and reset the configured flag.

    Intended **only** for test isolation — production code must never call this.
    """
    global _configured
    root = logging.getLogger(_ROOT)
    for h in root.handlers[:]:
        root.removeHandler(h)
    _configured = False
