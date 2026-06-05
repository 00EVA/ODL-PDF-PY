# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of label-based list detection from ``ListProcessor.java``.
# Behavior is reconstructed from the architecture doc
# ``05-tables-lists.md`` §8 and from the accessor/constant table in §7.
# No veraPDF source was copied.
#
# The following ListProcessor phases are NOT implemented here (future phases):
#   - §8.2  processListsFromTextNodes  (paragraph-node detection)
#   - §8.4  checkNeighborLists         (cross-page list linking)
#   - §8.5  Nesting                    (recursive sub-list detection)
#   - Korean "붙임" prefix stripping
"""Label-based list detection.

:func:`detect_lists` scans a flat sequence of :class:`TextLine` objects for
items that begin with a recognised list label (bullets, numbered, lettered,
roman) and groups consecutive same-style items into :class:`PDFList` objects.

Label patterns recognised (from §8.3 ``NumberingStyleNames`` table):

  DECIMAL / ARABIC_NUMBERS     ``1.``  ``1)``  ``1.`` ...
  ENGLISH_LETTERS_LOWER_CASE   ``a.``  ``a)``
  ENGLISH_LETTERS_UPPER_CASE   ``A.``  ``A)``
  ROMAN_NUMBERS_LOWER_CASE     ``i.``  ``ii.``
  ROMAN_NUMBERS_UPPER_CASE     ``I.``  ``II.``
  BULLET / UNORDERED           ``•``   ``-``   ``*``

Out of scope (future phases):
  - Paragraph-node detection (``processListsFromTextNodes``)
  - Neighbor list linking (``checkNeighborLists``)
  - Nesting
  - Korean "붙임" prefix handling
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from odl_pdf.entities.list_entity import ListItem, PDFList
from odl_pdf.entities.text import TextLine
from odl_pdf.logging_config import get_logger

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (from ListProcessor.java and architecture doc §8 / §9)
# ---------------------------------------------------------------------------

# Validation guard: an interval where every label matches this regex is
# rejected (it is a table of decimal-dotted floats, not a list).
_DECIMAL_FLOAT_PATTERN: re.Pattern[str] = re.compile(r"^\d+\.\d+$")

# Minimum items in an interval to form a list (from ListProcessor logic:
# "only intervals with more than 1 item are kept", §8.1).
_MIN_ITEMS: int = 2

# Maximum backward scan distance when searching for a matching interval.
_MAX_LIST_INTERVAL_LOOKBACK: int = 500


# ---------------------------------------------------------------------------
# Numbering styles (from §8.3)
# ---------------------------------------------------------------------------

class _Style:
    DECIMAL = "DECIMAL"
    ENGLISH_LOWER = "ENGLISH_LETTERS_LOWER_CASE"
    ENGLISH_UPPER = "ENGLISH_LETTERS_UPPER_CASE"
    ROMAN_LOWER = "ROMAN_NUMBERS_LOWER_CASE"
    ROMAN_UPPER = "ROMAN_NUMBERS_UPPER_CASE"
    BULLET = "BULLET"


# ---------------------------------------------------------------------------
# Label patterns
# ---------------------------------------------------------------------------

# Each entry: (style_name, compiled_pattern)
# The pattern must match the ENTIRE label token (the label + separator).
# We extract the label as group(1) and the separator as group(2).

_ROMAN_LOWER_VALUES = (
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
)

_LABEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Decimal: "1." or "1)" followed by whitespace
    (_Style.DECIMAL, re.compile(r"^(\d+)([.)]\s)")),
    # Lower alpha: "a." or "a)" followed by whitespace
    (_Style.ENGLISH_LOWER, re.compile(r"^([a-z])([.)]\s)")),
    # Upper alpha: "A." or "A)" followed by whitespace
    (_Style.ENGLISH_UPPER, re.compile(r"^([A-Z])([.)]\s)")),
    # Lower roman: "i." "ii." etc. followed by whitespace — must test before alpha
    (
        _Style.ROMAN_LOWER,
        re.compile(
            r"^(" + "|".join(_ROMAN_LOWER_VALUES[::-1]) + r")([.)]\s)",
            re.IGNORECASE,
        ),
    ),
    # Upper roman: same tokens in upper case
    (
        _Style.ROMAN_UPPER,
        re.compile(
            r"^(" + "|".join(v.upper() for v in _ROMAN_LOWER_VALUES[::-1]) + r")([.)]\s)",
        ),
    ),
    # Bullet: •, -, *, · followed by whitespace
    (_Style.BULLET, re.compile(r"^([•\-*·])(\s)")),
]


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

@dataclass
class _LabelInfo:
    """The label found at the start of a text line."""
    style: str
    label: str
    label_length: int  # total length of label + separator in source string


def _extract_label(text: str) -> _LabelInfo | None:
    """Try to match a list label at the start of *text*.

    Returns a :class:`_LabelInfo` on success, or ``None`` if the line does
    not start with a known label token.

    Lower-roman patterns are tested before lower-alpha to avoid misclassifying
    "i." (roman numeral 1) as the letter "i".
    """
    stripped = text.lstrip()  # ignore leading spaces
    offset = len(text) - len(stripped)

    # Test roman lower/upper before alpha to avoid misclassification.
    # The _LABEL_PATTERNS list is ordered: decimal → lower alpha → upper alpha
    # → lower roman → upper roman → bullet.
    # We need roman before alpha; reorder test sequence:
    priority_order = [
        _Style.ROMAN_LOWER,
        _Style.ROMAN_UPPER,
        _Style.DECIMAL,
        _Style.ENGLISH_LOWER,
        _Style.ENGLISH_UPPER,
        _Style.BULLET,
    ]
    patterns_by_style = {style: pat for style, pat in _LABEL_PATTERNS}

    for style in priority_order:
        pat = patterns_by_style[style]
        m = pat.match(stripped)
        if m:
            full_match = m.group(0)
            return _LabelInfo(
                style=style,
                label=m.group(1),
                label_length=offset + len(full_match),
            )
    return None


# ---------------------------------------------------------------------------
# Sequence validation helpers
# ---------------------------------------------------------------------------

_ROMAN_LOWER_MAP: dict[str, int] = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
    "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
    "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18,
    "xix": 19, "xx": 20,
}


def _is_consecutive(style: str, prev_label: str, curr_label: str) -> bool:
    """Return True when *curr_label* is the natural successor of *prev_label*
    for the given numbering *style*.

    For BULLET style any bullet following another bullet is valid.
    """
    if style == _Style.BULLET:
        return True  # bullets don't need strict ordering

    if style == _Style.DECIMAL:
        try:
            return int(curr_label) == int(prev_label) + 1
        except ValueError:
            return False

    if style in (_Style.ENGLISH_LOWER, _Style.ENGLISH_UPPER):
        return (
            len(curr_label) == 1
            and len(prev_label) == 1
            and ord(curr_label) == ord(prev_label) + 1
        )

    if style in (_Style.ROMAN_LOWER, _Style.ROMAN_UPPER):
        pl = prev_label.lower()
        cl = curr_label.lower()
        pv = _ROMAN_LOWER_MAP.get(pl)
        cv = _ROMAN_LOWER_MAP.get(cl)
        if pv is None or cv is None:
            return False
        return cv == pv + 1

    return False


def _is_valid_interval_label(label: str, style: str) -> bool:
    """Guard: reject decimal-dotted floats (e.g. "1.5") as list labels."""
    if _DECIMAL_FLOAT_PATTERN.match(label):
        logger.debug(
            "_is_valid_interval_label: rejected decimal-float label %r", label
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Interval accumulator
# ---------------------------------------------------------------------------

@dataclass
class _Interval:
    """A run of lines that form one candidate list."""
    style: str
    items: list[tuple[int, TextLine, _LabelInfo]] = field(default_factory=list)
    # items: list of (original_index, TextLine, LabelInfo)

    @property
    def last_label(self) -> str | None:
        return self.items[-1][2].label if self.items else None

    def can_accept(self, info: _LabelInfo) -> bool:
        """Return True when this interval can accept *info* as its next item."""
        if info.style != self.style:
            return False
        last = self.last_label
        if last is None:
            return True
        return _is_consecutive(self.style, last, info.label)


# ---------------------------------------------------------------------------
# List-builder helper
# ---------------------------------------------------------------------------

def _build_pdf_list(interval: _Interval) -> PDFList:
    """Convert a completed :class:`_Interval` into a :class:`PDFList`."""
    lst = PDFList()
    lst.numbering_style = interval.style
    # label_length: use the label_length from the first item's info
    lst.label_length = interval.items[0][2].label_length if interval.items else 0

    for _idx, line, info in interval.items:
        item = ListItem()
        item.add_content(line)
        lst.push(item)

    logger.debug(
        "_build_pdf_list: style=%s, %d items", interval.style, len(interval.items)
    )
    return lst


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_lists(
    lines: Sequence[TextLine],
) -> list[PDFList]:
    """Detect label-based lists from a sequence of :class:`TextLine` objects.

    Implements the Step-1 phase of ``ListProcessor.processLists`` (§8.1):
    scan every non-blank line for a recognised list label, accumulate
    consecutive same-style items into intervals, then emit a :class:`PDFList`
    for each interval that has at least two items.

    Parameters
    ----------
    lines:
        TextLine objects in reading order.  Typically from a single page, but
        multi-page input is handled transparently.

    Returns
    -------
    list[PDFList]
        Detected lists in document order.

    Notes
    -----
    - Minimum list size is 2 items (§8.1: "only intervals with more than 1
      item are kept").
    - Decimal-dotted-float labels (e.g. "1.5") are rejected (§8.1
      ``isCorrectList`` guard).
    - Neighbor-list linking, nesting, and paragraph-node detection are
      deferred to future phases.
    """
    logger.info("detect_lists: entry — %d input lines", len(lines))

    if not lines:
        logger.debug("detect_lists: empty input → no lists")
        return []

    # Collect (index, line, label_info) for every labeled line.
    labeled: list[tuple[int, TextLine, _LabelInfo]] = []
    for i, line in enumerate(lines):
        if line.is_blank or line.is_space_line:
            continue
        text = line.value
        info = _extract_label(text)
        if info is not None and _is_valid_interval_label(info.label, info.style):
            labeled.append((i, line, info))
            logger.debug(
                "detect_lists: line %d label=%r style=%s", i, info.label, info.style
            )

    logger.info("detect_lists: found %d labeled lines out of %d total", len(labeled), len(lines))

    if not labeled:
        return []

    # Build intervals by backward scanning (§8.1 processListItem).
    # For simplicity we do a single forward pass: each labeled line either
    # extends the most-recently open interval of the same style or starts a
    # new one.  This matches the effective behavior for the common case of
    # non-interleaved styles.
    intervals: list[_Interval] = []
    # Map style → the most recently open interval of that style.
    open_by_style: dict[str, _Interval] = {}

    for idx, line, info in labeled:
        style = info.style
        # Look for an open interval of this style that can accept this item.
        existing = open_by_style.get(style)
        if existing is not None and existing.can_accept(info):
            existing.items.append((idx, line, info))
            logger.debug(
                "detect_lists: extend interval style=%s label=%r (now %d items)",
                style,
                info.label,
                len(existing.items),
            )
        else:
            # Close any existing open interval for this style (it is broken).
            if existing is not None:
                logger.debug(
                    "detect_lists: close broken interval style=%s (%d items)",
                    style,
                    len(existing.items),
                )
                # Leave it in `intervals`; it was already appended when opened.
            new_iv = _Interval(style=style, items=[(idx, line, info)])
            intervals.append(new_iv)
            open_by_style[style] = new_iv
            logger.debug(
                "detect_lists: start new interval style=%s label=%r",
                style,
                info.label,
            )

    # Emit PDFList for every interval with >= MIN_ITEMS items.
    result: list[PDFList] = []
    for iv in intervals:
        if len(iv.items) < _MIN_ITEMS:
            logger.debug(
                "detect_lists: skip interval style=%s — only %d item(s) (need %d)",
                iv.style,
                len(iv.items),
                _MIN_ITEMS,
            )
            continue
        lst = _build_pdf_list(iv)
        result.append(lst)
        logger.info(
            "detect_lists: emitted PDFList style=%s items=%d",
            iv.style,
            iv.number_of_items if hasattr(iv, "number_of_items") else len(iv.items),
        )

    logger.info("detect_lists: produced %d list(s)", len(result))
    return result
