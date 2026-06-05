# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Text grouping and classification processors.

Clean-room port of the grouping pipeline described in
``docs/architecture/03-processors.md``.  No veraPDF/Java source was copied;
behavior is reconstructed from the architecture doc's algorithms and constants.

Public API
----------
group_lines(chunks)            -> list[TextLine]          (TextLineProcessor)
group_paragraphs(lines)        -> list[SemanticTextNode]  (ParagraphProcessor)
detect_headings(nodes, body)   -> None                    (HeadingProcessor, per-page)
assign_heading_levels(nodes)   -> None                    (HeadingProcessor, sequential)
detect_header_footer(pages_lines, page_heights)
                               -> list[list[SemanticTextNode]]  (HeaderFooterProcessor)

Constants (from architecture doc)
---------------------------------
ONE_LINE_PROBABILITY           = 0.75   (TextLineProcessor threshold)
TEXT_LINE_SPACE_RATIO          = 0.2    (space injection gap ratio)
SAME_BASELINE_TOLERANCE_RATIO  = 0.15  (baseline tolerance = ratio * font_size)
DIFFERENT_LINES_PROBABILITY    = 0.75   (ParagraphProcessor merge threshold)
HEADING_PROBABILITY            = 0.75   (HeadingProcessor promotion threshold)
HEADING_FONT_SIZE_RATIO        = 1.15  (font size must exceed body by this factor)
HEADING_WEIGHT_THRESHOLD       = 600.0 (font weight >= this considered bold)
MAX_HEADER_FOOTER_GAP          = 30.0  (max gap between adjacent header/footer elements)
HEADER_ZONE_RATIO              = 2/3   (header candidate bottom >= zone * page_height)
FOOTER_ZONE_RATIO              = 1/3   (footer candidate top <= zone * page_height)
LEADING_RATIO                  = 1.5   (paragraph merge: gap <= leading_ratio * line_height)
"""

from __future__ import annotations

import re

from odl_pdf.entities import (
    BoundingBox,
    SemanticTextNode,
    SemanticType,
    TextBlock,
    TextChunk,
    TextColumn,
    TextLine,
)
from odl_pdf.entities.text import _SHORT_LINE_WIDTH_RATIO, TextAlignment
from odl_pdf.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (architecture doc §03-processors.md)
# ---------------------------------------------------------------------------

ONE_LINE_PROBABILITY: float = 0.75
TEXT_LINE_SPACE_RATIO: float = 0.2
SAME_BASELINE_TOLERANCE_RATIO: float = 0.15
DIFFERENT_LINES_PROBABILITY: float = 0.75
HEADING_PROBABILITY: float = 0.75
HEADING_FONT_SIZE_RATIO: float = 1.15
HEADING_WEIGHT_THRESHOLD: float = 600.0
MAX_HEADER_FOOTER_GAP: float = 30.0
HEADER_ZONE_RATIO: float = 2.0 / 3.0
FOOTER_ZONE_RATIO: float = 1.0 / 3.0
LEADING_RATIO: float = 1.5
CLOSE_FONT_SIZE_EPSILON: float = 0.1  # from ParagraphProcessor.areCloseStyle
CLOSE_FONT_WEIGHT_EPSILON: float = 0.1

# --- Alignment + size-hierarchy heading signals (task #40) -------------------
# Boost applied when the first line of a node is CENTERED (e.g. a paper title
# or chapter heading in a document without section numbers).  Only applied to
# SHORT lines (narrower than _SHORT_LINE_WIDTH_RATIO * page_width) so that full-
# width justified body text is not falsely boosted.
HEADING_ALIGNMENT_CENTER_BOOST: float = 0.35

# Additional boost when BOTH centered AND larger than body font size.
# (Each signal already contributes on its own; this is the *combined* extra.)
HEADING_CENTERED_AND_LARGE_EXTRA: float = 0.15

# Outline section-number heading signal. Technical/technical procedure documents number their
# sections (``6``, ``6.1``, ``6.1.1`` …) in the SAME font size as body text, so
# the font-size/weight heading signals never fire on them. A node whose first
# line begins with an outline number followed by a short title is a strong
# heading regardless of font size. Matches "6", "6.1", "6.8 Out of Tolerance",
# "3.2.1 Die Ansätze" — but NOT a decimal value mid-sentence (anchored at start)
# nor a long paragraph that merely opens with a number.
_SECTION_NUMBER_RE = re.compile(r"^\s*([1-9]\d*(?:\.\d+){0,4})\.?(?:\s+\S|\s*$)")
# (First number >= 1: outline sections start at 1, so a bare "0" — a revision
#  number or list value — is not a section heading.)
# A numbered heading's first line must be short (a title, not a paragraph).
_SECTION_HEADING_MAX_WORDS: int = 12
# Pattern to reject "X of Y" page-number lines (e.g. "5 of 11", "Page 3 of 10").
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Section-number line helpers (used in both grouping and heading detection)
# ---------------------------------------------------------------------------

def _line_is_section_number_heading(line: TextLine) -> bool:
    """Whether *line* is a short outline-numbered section heading.

    A qualifying line:
    - begins with an outline number pattern (``6``, ``6.1``, ``6.1.1.2`` …),
    - contains at most _SECTION_HEADING_MAX_WORDS words (title, not body),
    - is NOT a page-number expression like "5 of 11",
    - does NOT contain a colon in the title words (colons indicate metadata
      field labels like "8 Date: 02/09/2026" or "Revision: 8", not headings).

    Used by group_paragraphs to split section-number lines into their own
    SemanticTextNode so that detect_headings can promote them correctly.
    """
    txt = line.value.strip()
    if not txt:
        return False
    # Reject "X of Y" page-number pattern before running the heavier regex
    if _PAGE_NUMBER_RE.match(txt):
        return False
    if not _SECTION_NUMBER_RE.match(txt):
        return False
    if len(txt.split()) > _SECTION_HEADING_MAX_WORDS:
        return False
    # Reject lines that contain a colon — these are metadata field labels
    # (e.g. "8 Date: 02/09/2026", "Revision: 8"), not section titles.
    if ":" in txt:
        return False
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _baseline_tolerance(font_size: float) -> float:
    """Same-baseline tolerance: fraction of font size."""
    return SAME_BASELINE_TOLERANCE_RATIO * max(font_size, 1.0)


def _are_same_baseline(a: TextChunk, b: TextChunk) -> bool:
    """Whether two chunks share a baseline within tolerance."""
    if a.base_line is None or b.base_line is None:
        return False
    fs = max(a.font_size, b.font_size, 1.0)
    tol = _baseline_tolerance(fs)
    return abs(a.base_line - b.base_line) <= tol


def _count_one_line_probability(prev_line: TextLine, chunk: TextChunk) -> float:
    """Estimate the probability that *chunk* continues *prev_line*.

    Heuristic: combines baseline Y proximity and font-size consistency.
    Threshold ONE_LINE_PROBABILITY = 0.75 (from TextLineProcessor).

    Algorithm (clean-room reconstruction):
    1. If the baselines are within tolerance and font sizes close => high score.
    2. If baselines differ => low score.
    """
    if prev_line.is_empty:
        return 0.0
    prev_chunk = prev_line.chunks[-1]
    if prev_chunk.base_line is None or chunk.base_line is None:
        return 0.0

    fs = max(prev_chunk.font_size, chunk.font_size, 1.0)
    tol = _baseline_tolerance(fs)
    baseline_diff = abs(prev_chunk.base_line - chunk.base_line)

    if baseline_diff > tol:
        return 0.0

    # Check horizontal adjacency: gap between prev right edge and current left
    prev_bb = prev_line.bounding_box
    if prev_bb is None:
        return 0.0
    gap = chunk.left - prev_bb.right

    # Gap must be non-negative (no extreme overlap) and within reasonable limit
    # Negative gap (overlap) is OK if baselines match (kern/ligature scenario)
    max_gap = fs * 2.0  # generous; space injection handles the visual gap later
    if gap > max_gap:
        return 0.0

    # Font size must be within CLOSE_FONT_SIZE_EPSILON * 10 of the prev font size
    # (looser than style-match; we just want gross consistency)
    size_diff = abs(prev_chunk.font_size - chunk.font_size)
    if size_diff > fs * 0.3:
        return 0.9  # different size but same baseline -> allow merge with reduced score

    return 1.0


def _inject_spaces(line: TextLine, chunks_after_whitespace: set[int]) -> TextLine:
    """Phase 2: sort a line's chunks by left X, then inject synthetic space chunks
    where the gap exceeds font_size * TEXT_LINE_SPACE_RATIO or where a whitespace
    token was seen before the chunk in PDF stream order.

    The space bounding box spans min(prevEnd, curStart)..max(prevEnd, curStart)
    (may have zero width — intentional per the architecture doc).
    """
    if not line.chunks:
        return line

    sorted_chunks = sorted(line.chunks, key=lambda c: c.left)
    result_line = TextLine()
    result_line.push(sorted_chunks[0])

    for chunk in sorted_chunks[1:]:
        prev = result_line.chunks[-1]
        fs = max(prev.font_size, chunk.font_size, 1.0)
        gap = chunk.left - prev.right

        # Default: two separate non-whitespace chunks on one line almost always
        # sit on a word boundary (the PDF emitted them as distinct text runs),
        # so INJECT a space. Only SUPPRESS when they clearly abut as a single
        # word — a small positive gap below the mid-word threshold (e.g. a font
        # switch mid-word, or "qua"+"lity"). This is robust to unreliable
        # right-edge widths (some PDFs yield width=0 / overlapping boxes, which
        # made the old "gap > threshold" test silently drop spaces, producing
        # "ProductionPartApproval"). A whitespace token seen between them in
        # stream order always forces the space.
        midword_gap = fs * TEXT_LINE_SPACE_RATIO  # below this = same word
        abuts_as_one_word = 0.0 <= gap < midword_gap
        insert_space = (not abuts_as_one_word) or (id(chunk) in chunks_after_whitespace)
        # Don't double-space when the text already carries one at the boundary
        # (e.g. "Lorem " + "Ipsum" -> "Lorem  Ipsum").
        if prev.value.endswith((" ", "\t", "\n")) or chunk.value.startswith((" ", "\t", "\n")):
            insert_space = False
        if insert_space:
            space_left = min(prev.right, chunk.left)
            space_right = max(prev.right, chunk.left)
            page = prev.page_number
            space_bb = BoundingBox.of(
                page, space_left, prev.bottom, space_right, prev.top
            )
            space_chunk = TextChunk(
                bounding_box=space_bb,
                value=" ",
                font_size=fs,
                font_name=prev.font_name,
            )
            space_chunk.base_line = prev.base_line
            result_line.push(space_chunk)

        result_line.push(chunk)

    return result_line


# ---------------------------------------------------------------------------
# TextLineProcessor
# ---------------------------------------------------------------------------

def group_lines(chunks: list[TextChunk]) -> list[TextLine]:
    """Phase-1 + Phase-2 of TextLineProcessor.

    Converts a page's flat TextChunk list into TextLines:
    1. Stream-order grouping by baseline proximity + ONE_LINE_PROBABILITY.
    2. Per-line sort by left-X + synthetic space injection.

    Whitespace-only and empty chunks are skipped but whitespace chunks set a
    flag so the immediately following real chunk is marked as needing a space
    (tracked by object identity, not string value).

    Args:
        chunks: Raw TextChunk list for one page (PDF-stream order).

    Returns:
        List of TextLine objects (non-empty; spaces may be injected).
    """
    logger.info(
        "group_lines: entry, input_chunks=%d", len(chunks)
    )

    lines: list[TextLine] = []
    current_line: TextLine = TextLine()
    chunks_after_whitespace: set[int] = set()  # identity set by id()
    saw_whitespace = False

    for chunk in chunks:
        if chunk.is_empty:
            logger.debug("group_lines: skipping empty chunk value=%r", chunk.value)
            continue
        if chunk.is_whitespace:
            saw_whitespace = True
            logger.debug("group_lines: skipping whitespace chunk, flagging next")
            continue

        # Real chunk — check if following whitespace
        if saw_whitespace:
            chunks_after_whitespace.add(id(chunk))
        saw_whitespace = False

        if current_line.is_empty:
            current_line.push(chunk)
            continue

        prob = _count_one_line_probability(current_line, chunk)
        logger.debug(
            "group_lines: one_line_prob=%.3f for chunk value=%r", prob, chunk.value
        )

        if prob < ONE_LINE_PROBABILITY:
            # Start a new line
            logger.debug(
                "group_lines: line break (prob=%.3f < %.2f), chunks_in_line=%d",
                prob, ONE_LINE_PROBABILITY, len(current_line.chunks),
            )
            lines.append(current_line)
            current_line = TextLine()
        current_line.push(chunk)

    if not current_line.is_empty:
        lines.append(current_line)

    logger.info(
        "group_lines: phase-1 done, raw_lines=%d", len(lines)
    )

    # Phase 2: sort chunks per line + space injection
    final_lines: list[TextLine] = []
    for line in lines:
        enriched = _inject_spaces(line, chunks_after_whitespace)
        final_lines.append(enriched)
        logger.debug(
            "group_lines: line text=%r, chunks=%d",
            enriched.value, len(enriched.chunks),
        )

    logger.info(
        "group_lines: phase-2 done, final_lines=%d", len(final_lines)
    )
    return final_lines


# ---------------------------------------------------------------------------
# ParagraphProcessor helpers
# ---------------------------------------------------------------------------

def _get_line_height(line: TextLine) -> float:
    """Return the line's visual height, falling back to font_size."""
    bb = line.bounding_box
    if bb is not None and bb.height > 0:
        return bb.height
    return line.font_size or 12.0


def _leading_for_line(line: TextLine) -> float:
    """Expected inter-line gap for this line's font size."""
    return _get_line_height(line) * LEADING_RATIO


def _vertical_gap(upper_line: TextLine, lower_line: TextLine) -> float:
    """Vertical gap between the bottom of the upper line and top of the lower line.

    In PDF y-up space the upper line has a higher bottom coordinate.
    """
    upper_bb = upper_line.bounding_box
    lower_bb = lower_line.bounding_box
    if upper_bb is None or lower_bb is None:
        return float("inf")
    # upper_line should be above lower_line (higher Y values in y-up space)
    return upper_bb.bottom - lower_bb.top


def _merge_leading_probability(prev_block: TextBlock, next_line: TextLine) -> float:
    """Estimate the probability that next_line continues prev_block.

    Mirrors ChunksMergeUtils.mergeLeadingProbability logic (clean-room):
    - Vertical gap must be within LEADING_RATIO * line_height.
    - Font sizes should be close.
    """
    if prev_block.is_empty:
        return 0.0
    prev_line = prev_block.lines[-1]

    gap = _vertical_gap(prev_line, next_line)

    # In y-up coordinates, if prev_line is above (higher Y), the gap is positive
    # We also handle the inverted case (lines given bottom-to-top order)
    if gap < 0:
        # Try the other direction — lines may be in top-to-bottom y-down order
        lower_bb = prev_line.bounding_box
        upper_bb = next_line.bounding_box
        if lower_bb is None or upper_bb is None:
            return 0.0
        gap = upper_bb.bottom - lower_bb.top
        if gap < 0:
            # Overlapping or unknown — allow merge if font consistent
            gap = 0.0

    leading = _leading_for_line(prev_line)
    if gap > leading:
        logger.debug(
            "_merge_leading_probability: gap=%.2f > leading=%.2f => 0.0",
            gap, leading,
        )
        return 0.0

    # Font size proximity check
    prev_fs = prev_line.font_size or 12.0
    next_fs = next_line.font_size or 12.0
    if abs(prev_fs - next_fs) > CLOSE_FONT_SIZE_EPSILON * 10:
        return 0.5  # different sizes reduce probability

    return 1.0


def _are_blocks_same_text_size(a: TextBlock, b: TextBlock) -> bool:
    """Cross-product check: any font size from A close to any font size from B.

    Implements ParagraphProcessor.areTextBlocksHaveSameTextSize exactly:
    nested loop, not set intersection.
    """
    sizes_a = {c.font_size for line in a.lines for c in line.chunks}
    sizes_b = {c.font_size for line in b.lines for c in line.chunks}
    for sa in sizes_a:
        for sb in sizes_b:
            if abs(sa - sb) <= CLOSE_FONT_SIZE_EPSILON * 10:
                return True
    return False


def _line_to_block(line: TextLine) -> TextBlock:
    block = TextBlock()
    block.push(line)
    return block


def _block_to_node(block: TextBlock) -> SemanticTextNode:
    col = TextColumn()
    col.push(block)
    return SemanticTextNode.paragraph([col])


# ---------------------------------------------------------------------------
# ParagraphProcessor
# ---------------------------------------------------------------------------

def group_paragraphs(lines: list[TextLine]) -> list[SemanticTextNode]:
    """Group TextLines into SemanticTextNode paragraphs.

    Implements ParagraphProcessor.processParagraphs (simplified single-pass
    reconstruction): wraps each TextLine in a single-line TextBlock, then
    merges consecutive blocks where the merge-leading probability exceeds
    DIFFERENT_LINES_PROBABILITY AND the text sizes match.

    The full Java version runs 8 ordered passes (justified, left-aligned x2,
    etc.).  This clean-room port implements the core spatial + size proximity
    signal that is common to all passes.  The single forward scan is
    order-sensitive (architecture doc: "Later passes operate on the output of
    earlier ones").

    Args:
        lines: TextLines for one page (in reading order).

    Returns:
        List of SemanticTextNode paragraphs.
    """
    logger.info("group_paragraphs: entry, input_lines=%d", len(lines))

    if not lines:
        return []

    # Wrap each line in a single-line TextBlock
    blocks: list[TextBlock] = [_line_to_block(line) for line in lines]

    # Forward scan: merge consecutive blocks
    merged: list[TextBlock] = [blocks[0]]
    for block in blocks[1:]:
        prev = merged[-1]
        next_line = block.first_line
        if next_line is None:
            merged.append(block)
            continue

        # ----------------------------------------------------------------
        # Section-number split guard (section-heading recall fix):
        #
        # technical procedure documents use the same 12pt font for section headings as
        # body text, so font-size signals alone never fire.  The section
        # number signal in _heading_probability adds 0.8, but ONLY when
        # the node's first line is the number — which requires each
        # numbered line to start its OWN node.
        #
        # Rule A: if *next_line* is a section-number heading, always
        #         start a new block (never absorb into the previous para).
        # Rule B: if the *last line* of the current block is a bare
        #         section-number line (e.g. "6.1" alone), allow merging
        #         the immediately following SHORT title line so that the
        #         heading node reads "6.1\nGeneral" — but reject long body
        #         lines (they become a separate paragraph).
        # ----------------------------------------------------------------
        next_is_section = _line_is_section_number_heading(next_line)
        prev_last_line = prev.lines[-1] if prev.lines else None
        prev_last_is_section = (
            prev_last_line is not None
            and _line_is_section_number_heading(prev_last_line)
        )

        if next_is_section:
            # Rule A: always start a new block on a section-number line
            logger.debug(
                "group_paragraphs: section-number split (Rule A) at line=%r",
                next_line.value,
            )
            merged.append(block)
            continue

        if prev_last_is_section:
            # Rule B: prev block ends with a bare section-number line.
            # Allow merge only if the following line is a short title
            # (no more than _SECTION_HEADING_MAX_WORDS words) AND the line
            # does not contain TOC dot-leaders ("....").  Dot-leader lines
            # are Table of Contents entries, not section title lines.
            next_txt = next_line.value.strip()
            next_words = len(next_txt.split())
            has_dot_leaders = "...." in next_txt
            if next_words <= _SECTION_HEADING_MAX_WORDS and not has_dot_leaders:
                # Short title without dot-leaders — merge into the section-number block
                prev.push(next_line)
                logger.debug(
                    "group_paragraphs: section-number title merge (Rule B)"
                    " bare_section=%r title=%r",
                    prev_last_line.value, next_line.value,
                )
            else:
                # Long body line or TOC dot-leader — force a new block
                logger.debug(
                    "group_paragraphs: section-number body split (Rule B)"
                    " bare_section=%r next=%r (dot_leaders=%s)",
                    prev_last_line.value, next_line.value, has_dot_leaders,
                )
                merged.append(block)
            continue

        prob = _merge_leading_probability(prev, next_line)
        same_size = _are_blocks_same_text_size(prev, block)

        logger.debug(
            "group_paragraphs: merge_prob=%.3f same_size=%s for line=%r",
            prob, same_size, next_line.value,
        )

        if prob >= DIFFERENT_LINES_PROBABILITY and same_size:
            # Merge block's lines into prev
            for line in block.lines:
                prev.push(line)
            logger.debug(
                "group_paragraphs: merged line into existing block, total_lines=%d",
                len(prev.lines),
            )
        else:
            merged.append(block)

    # ----------------------------------------------------------------
    # TOC de-splitting pass:
    # Rule A may have split out a bare section-number block whose
    # immediately following block contains TOC dot-leaders ("....").
    # In that case the "section number" is really a TOC entry number,
    # not a standalone heading — re-merge it into the following block
    # so the combined block (e.g. "1\nScope.....") does NOT get the
    # section-number heading boost (its first line would no longer be
    # alone and the dot-leader content disqualifies it as a heading).
    #
    # Additionally, bare page-number blocks (a single digit or decimal
    # like "4", "10") that immediately follow a dot-leader block are
    # TOC page numbers, not section headings — they also get re-merged.
    # ----------------------------------------------------------------

    def _block_has_dot_leaders(blk: TextBlock) -> bool:
        """True if any line in the block contains TOC dot-leaders."""
        for ln in blk.lines:
            if "...." in ln.value:
                return True
        return False

    def _block_is_bare_section_num(blk: TextBlock) -> bool:
        """True if block is a single section-number line."""
        if len(blk.lines) != 1:
            return False
        first = blk.first_line
        return first is not None and _line_is_section_number_heading(first)

    cleaned: list[TextBlock] = []
    i = 0
    while i < len(merged):
        blk = merged[i]
        first = blk.first_line

        # Case 1: single section-number block followed by a dot-leader block
        # → merge section-num INTO the dot-leader block (section_num first)
        if (
            _block_is_bare_section_num(blk)
            and i + 1 < len(merged)
            and _block_has_dot_leaders(merged[i + 1])
        ):
            nxt = merged[i + 1]
            logger.debug(
                "group_paragraphs: TOC de-split (case 1): section-num %r"
                " + dot-leader block",
                first.value if first else "",
            )
            new_block = TextBlock()
            for ln in blk.lines:
                new_block.push(ln)
            for ln in nxt.lines:
                new_block.push(ln)
            cleaned.append(new_block)
            i += 2
            continue

        # Case 2: single section-number block immediately preceded by
        # a dot-leader block — this is a TOC page number (e.g. "4" after
        # ".........." lines).  Re-attach it to the preceding block.
        if (
            _block_is_bare_section_num(blk)
            and cleaned
            and _block_has_dot_leaders(cleaned[-1])
        ):
            logger.debug(
                "group_paragraphs: TOC de-split (case 2): bare page-num %r"
                " after dot-leader block",
                first.value if first else "",
            )
            for ln in blk.lines:
                cleaned[-1].push(ln)
            i += 1
            continue

        cleaned.append(blk)
        i += 1

    if len(cleaned) != len(merged):
        logger.info(
            "group_paragraphs: TOC de-split removed %d section-num blocks",
            len(merged) - len(cleaned),
        )

    logger.info(
        "group_paragraphs: produced %d paragraph blocks from %d lines",
        len(cleaned), len(lines),
    )

    # Convert each TextBlock into a SemanticTextNode paragraph
    nodes = [_block_to_node(block) for block in cleaned]
    return nodes


# ---------------------------------------------------------------------------
# HeadingProcessor — per-page detection
# ---------------------------------------------------------------------------

def _node_font_size(node: SemanticTextNode) -> float | None:
    """Extract the dominant font size from the first line of a node."""
    first_line = node.first_line
    if first_line is None or not first_line.chunks:
        return None
    return first_line.font_size


def _node_font_weight(node: SemanticTextNode) -> float | None:
    """Extract the font weight from the first non-space chunk of the node."""
    first_line = node.first_line
    if first_line is None:
        return None
    for chunk in first_line.chunks:
        if not chunk.is_whitespace and chunk.font_weight is not None:
            return chunk.font_weight
    return None


def _node_line_count(node: SemanticTextNode) -> int:
    """Total number of lines across all columns of a node."""
    count = 0
    for col in node.columns:
        for block in col.blocks:
            count += len(block.lines)
    return count


def _heading_probability(
    node: SemanticTextNode,
    body_font_size: float,
    all_font_sizes: list[float],
    all_font_weights: list[float],
    page_width: float = 0.0,
) -> float:
    """Estimate the probability that *node* is a heading.

    Clean-room reconstruction of NodeUtils.headingProbability +
    textNodeStatistics rarity boosts (HeadingProcessor.java), extended with
    two additional signals for generalisation beyond section numbers
    (task #40):

    Signals:
    1. Font-size ratio vs body font size (primary signal).
    2. Font-size rarity boost: if this font size is rare among all nodes,
       it is more likely a heading.
    3. Font-weight rarity boost: if this weight (bold) appears rarely, boost.
    4. Line count penalty: multi-line nodes are less likely to be headings.
    5. Outline section number (font-size-independent, section-number path).
    6. Alignment: a short centered line is a strong title/heading signal
       (research papers, slides, documents without section numbers).
    7. (See detect_headings / assign_heading_levels for size-rank path.)
    """
    fs = _node_font_size(node)
    if fs is None:
        return 0.0

    score = 0.0

    # --- Signal 1: font-size ratio ---
    ratio = fs / max(body_font_size, 1.0)
    if ratio >= HEADING_FONT_SIZE_RATIO:
        # Strong heading signal: score proportional to ratio overshoot
        score += min(0.6 + (ratio - HEADING_FONT_SIZE_RATIO) * 0.5, 0.85)
    elif ratio > 1.0:
        # Slightly larger than body — weak signal
        score += 0.3 * (ratio - 1.0) / (HEADING_FONT_SIZE_RATIO - 1.0)

    # --- Signal 2: font-size rarity boost ---
    if all_font_sizes:
        count_same_size = sum(
            1 for s in all_font_sizes
            if abs(s - fs) <= CLOSE_FONT_SIZE_EPSILON * 10
        )
        rarity = 1.0 - (count_same_size / len(all_font_sizes))
        score += rarity * 0.15  # rarity boost scaled to 0..0.15

    # --- Signal 3: font-weight rarity boost ---
    fw = _node_font_weight(node)
    if fw is not None and fw >= HEADING_WEIGHT_THRESHOLD:
        if all_font_weights:
            count_bold = sum(
                1 for w in all_font_weights
                if w >= HEADING_WEIGHT_THRESHOLD
            )
            rarity = 1.0 - (count_bold / len(all_font_weights))
            score += rarity * 0.15
        else:
            score += 0.1  # bold with no context

    # --- Signal 5: outline section number (font-size-independent) ---
    # A node whose first line is a short title beginning with an outline number
    # ("6", "6.1", "6.8 Out of Tolerance") is a section heading even when it is
    # the same font size as body text — the common case in technical procedure documents.
    # This is a strong structural signal; on its own it clears the promotion
    # threshold (a bare numbered title is a heading regardless of font).
    has_section_number = _starts_with_section_number(node)
    if has_section_number:
        score += 0.8

    # --- Signal 4: line count penalty ---
    # Multi-line nodes are less likely to be headings — UNLESS the section-number
    # signal fired, in which case the extra lines are the title text intentionally
    # merged with the section number by group_paragraphs (Rule B).  Do not penalise
    # a 2-line "6.1\nGeneral" node — that IS a heading.
    line_count = _node_line_count(node)
    if line_count > 1 and not has_section_number:
        score -= 0.1 * (line_count - 1)

    # --- Signal 6: alignment (centered short line => heading boost) ---
    # Applies only when page_width is known and the node is a single-line node
    # (multi-line centered blocks are usually body paragraphs, not headings).
    # Only boost if not already covered by the section-number path.
    if page_width > 0.0 and not has_section_number:
        first_line = node.first_line
        if first_line is not None and first_line.is_centered(page_width, require_short=True):
            score += HEADING_ALIGNMENT_CENTER_BOOST
            logger.debug(
                "_heading_probability: centered-line boost +%.2f, text=%r",
                HEADING_ALIGNMENT_CENTER_BOOST, node.value[:40],
            )
            # Extra combined boost when the line is ALSO larger than body
            if ratio >= HEADING_FONT_SIZE_RATIO:
                score += HEADING_CENTERED_AND_LARGE_EXTRA
                logger.debug(
                    "_heading_probability: centered+large extra boost +%.2f",
                    HEADING_CENTERED_AND_LARGE_EXTRA,
                )

    logger.debug(
        "_heading_probability: fs=%.1f body=%.1f ratio=%.2f score=%.3f page_width=%.1f",
        fs, body_font_size, ratio, score, page_width,
    )
    return score


def _first_line_text(node: SemanticTextNode) -> str:
    """Text of the node's first line (the candidate heading line)."""
    for col in node.columns:
        for block in col.blocks:
            for line in block.lines:
                return line.value
    return node.value.split("\n", 1)[0] if node.value else ""


def _starts_with_section_number(node: SemanticTextNode) -> bool:
    """Whether the node's first line is a short outline-numbered heading.

    Returns False for Table-of-Contents nodes: if any line in the node
    contains TOC dot-leaders ("...."), the section number is a TOC entry
    rather than a real section heading.
    """
    first = _first_line_text(node).strip()
    if not first:
        return False
    m = _SECTION_NUMBER_RE.match(first)
    if not m:
        return False
    # Reject "X of Y" page-number expressions.
    if _PAGE_NUMBER_RE.match(first):
        return False
    # Reject decimal *values* that are really data (e.g. "6.4.1" alone is fine —
    # it's a section id — but a line like "12.5 mm tolerance band ..." with many
    # words is body text). Require the whole first line to be short.
    if len(first.split()) > _SECTION_HEADING_MAX_WORDS:
        return False
    # Reject lines that contain a colon — metadata field labels like
    # "8 Date: 02/09/2026" are not section headings.
    if ":" in first:
        return False
    # Reject TOC nodes: if the node body contains dot-leaders, this section
    # number is part of a Table of Contents entry, not a standalone heading.
    full_text = node.value
    if "...." in full_text:
        return False
    return True


def detect_headings(
    nodes: list[SemanticTextNode],
    body_font_size: float,
    page_width: float = 0.0,
) -> None:
    """Classify SemanticTextNodes as HEADING or PARAGRAPH in place.

    Implements HeadingProcessor.processHeadings (architecture doc §03).
    Nodes already classified as HEADING are left unchanged.

    The heading probability combines:
    - font-size ratio vs body_font_size (threshold HEADING_FONT_SIZE_RATIO = 1.15)
    - font-size rarity boost across the page's font-size distribution
    - font-weight rarity boost (bold nodes)
    - line-count penalty (multi-line nodes less likely to be headings)
    - alignment boost: short centered lines get a strong heading boost
      (task #40; requires page_width > 0.0 to activate).

    A node is promoted to HEADING if probability > HEADING_PROBABILITY (0.75).

    Args:
        nodes: SemanticTextNodes for one page (modified in place).
        body_font_size: Dominant body font size for the page (or document).
        page_width: Page width in PDF points (0.0 to skip alignment signal).
    """
    logger.info(
        "detect_headings: entry, nodes=%d, body_font_size=%.1f, page_width=%.1f",
        len(nodes), body_font_size, page_width,
    )

    if not nodes:
        return

    # Collect font-size and font-weight distribution for rarity boosts
    all_font_sizes: list[float] = []
    all_font_weights: list[float] = []
    for node in nodes:
        fs = _node_font_size(node)
        if fs is not None:
            all_font_sizes.append(fs)
        fw = _node_font_weight(node)
        if fw is not None:
            all_font_weights.append(fw)

    promoted = 0
    for node in nodes:
        if node.semantic_type == SemanticType.HEADING:
            continue  # already classified

        prob = _heading_probability(
            node, body_font_size, all_font_sizes, all_font_weights, page_width
        )

        if prob > HEADING_PROBABILITY:
            node.set_semantic_type(SemanticType.HEADING)
            promoted += 1
            logger.debug(
                "detect_headings: promoted node to HEADING, prob=%.3f, text=%r",
                prob, node.value[:40],
            )
        else:
            logger.debug(
                "detect_headings: node stays PARAGRAPH, prob=%.3f, text=%r",
                prob, node.value[:40],
            )

    logger.info(
        "detect_headings: done, promoted=%d/%d nodes to HEADING",
        promoted, len(nodes),
    )


# ---------------------------------------------------------------------------
# HeadingProcessor — sequential level assignment
# ---------------------------------------------------------------------------

def assign_heading_levels(nodes: list[SemanticTextNode]) -> None:
    """Assign heading_level (1..6) to HEADING nodes by descending font size.

    Implements HeadingProcessor.detectHeadingsLevels (architecture doc §03).
    Non-HEADING nodes are ignored.

    Algorithm (clean-room reconstruction from the TreeMap/TextStyle comparator
    description):
    1. Collect all unique font sizes from HEADING nodes.
    2. Sort them in descending order (largest first => H1).
    3. Assign levels 1..6; sizes beyond rank 6 are capped at 6.
    4. Walk all HEADING nodes and set heading_level from the rank map.

    Args:
        nodes: All SemanticTextNodes for the document (modified in place).
    """
    logger.info("assign_heading_levels: entry, total_nodes=%d", len(nodes))

    heading_nodes = [
        n for n in nodes if n.semantic_type == SemanticType.HEADING
    ]

    if not heading_nodes:
        logger.info("assign_heading_levels: no headings found, nothing to do")
        return

    # Collect unique font sizes
    size_to_level: dict[float, int] = {}
    unique_sizes: list[float] = []
    for node in heading_nodes:
        fs = _node_font_size(node)
        if fs is not None and not any(
            abs(fs - s) <= CLOSE_FONT_SIZE_EPSILON * 10 for s in unique_sizes
        ):
            unique_sizes.append(fs)

    # Sort descending: largest font = H1
    unique_sizes.sort(reverse=True)
    logger.debug(
        "assign_heading_levels: unique_sizes=%s", unique_sizes
    )

    for rank, size in enumerate(unique_sizes, start=1):
        level = min(rank, 6)
        size_to_level[size] = level
        logger.debug(
            "assign_heading_levels: size=%.1f -> H%d", size, level
        )

    # Assign levels to nodes
    assigned = 0
    for node in heading_nodes:
        fs = _node_font_size(node)
        if fs is None:
            logger.warning(
                "assign_heading_levels: HEADING node has no font size, skipping"
            )
            continue
        # Find matching size in map (with tolerance)
        matched_level: int | None = None
        for size, level in size_to_level.items():
            if abs(fs - size) <= CLOSE_FONT_SIZE_EPSILON * 10:
                matched_level = level
                break
        if matched_level is None:
            # Font size not in map — use level 6 as fallback
            logger.warning(
                "assign_heading_levels: HEADING fs=%.1f not in level map, using 6", fs
            )
            matched_level = 6
        node.heading_level = matched_level
        assigned += 1
        logger.debug(
            "assign_heading_levels: node text=%r -> H%d",
            node.value[:30], matched_level,
        )

    logger.info(
        "assign_heading_levels: done, assigned levels to %d/%d heading nodes",
        assigned, len(heading_nodes),
    )


# ---------------------------------------------------------------------------
# HeaderFooterProcessor
# ---------------------------------------------------------------------------

def _classify_line_position(
    line: TextLine,
    page_height: float,
) -> SemanticType | None:
    """Classify a line as HEADER, FOOTER, or PARAGRAPH by position.

    Returns:
        SemanticType.HEADER if line bottom >= HEADER_ZONE_RATIO * page_height
        SemanticType.FOOTER if line top <= FOOTER_ZONE_RATIO * page_height
        SemanticType.PARAGRAPH otherwise
    """
    bb = line.bounding_box
    if bb is None:
        return SemanticType.PARAGRAPH

    header_threshold = HEADER_ZONE_RATIO * page_height
    footer_threshold = FOOTER_ZONE_RATIO * page_height

    if bb.bottom >= header_threshold:
        return SemanticType.HEADER
    if bb.top <= footer_threshold:
        return SemanticType.FOOTER
    return SemanticType.PARAGRAPH


def detect_header_footer(
    pages_lines: list[list[TextLine]],
    page_heights: list[float],
) -> list[list[SemanticTextNode]]:
    """Classify TextLines as HEADER, FOOTER, or PARAGRAPH across pages.

    Implements HeaderFooterProcessor (architecture doc §03):
    1. Per-line spatial filter: lines in the header zone (bottom >= 2/3 * H)
       or footer zone (top <= 1/3 * H) are candidates.
    2. Cross-page repetition: lines with the same text at the same position
       on adjacent pages are confirmed as HEADER/FOOTER.  (For single-page
       documents, position-only classification is used.)

    The output is a list of per-page lists of SemanticTextNode where each node
    carries the classified SemanticType.

    Args:
        pages_lines: Per-page lists of TextLines (one list per page).
        page_heights: Height of each page in PDF points (same length).

    Returns:
        Per-page list of SemanticTextNodes with HEADER/FOOTER/PARAGRAPH types.
    """
    logger.info(
        "detect_header_footer: entry, pages=%d", len(pages_lines)
    )

    if not pages_lines:
        return []

    num_pages = len(pages_lines)
    page_heights_padded = list(page_heights) + [792.0] * max(0, num_pages - len(page_heights))

    # Step 1: classify each line by position
    page_types: list[list[SemanticType]] = []
    for page_idx, lines in enumerate(pages_lines):
        ph = page_heights_padded[page_idx]
        types: list[SemanticType] = []
        for line in lines:
            stype = _classify_line_position(line, ph) or SemanticType.PARAGRAPH
            types.append(stype)
            logger.debug(
                "detect_header_footer: page=%d line=%r type=%s",
                page_idx, line.value[:30], stype.name,
            )
        page_types.append(types)

    # Step 2: cross-page confirmation for multi-page documents
    # If the same text appears at header/footer position on consecutive pages,
    # it is confirmed; otherwise revert to PARAGRAPH.
    if num_pages > 1:
        page_types = _cross_page_confirm(pages_lines, page_types)

    # Build output SemanticTextNodes
    result: list[list[SemanticTextNode]] = []
    for page_idx, (lines, types) in enumerate(zip(pages_lines, page_types)):
        page_nodes: list[SemanticTextNode] = []
        for line, stype in zip(lines, types):
            node = _wrap_line_as_semantic_node(line, stype)
            page_nodes.append(node)
            logger.debug(
                "detect_header_footer: page=%d node type=%s text=%r",
                page_idx, stype.name, node.value[:30],
            )
        result.append(page_nodes)

    header_count = sum(
        1 for page in result for n in page if n.semantic_type == SemanticType.HEADER
    )
    footer_count = sum(
        1 for page in result for n in page if n.semantic_type == SemanticType.FOOTER
    )
    logger.info(
        "detect_header_footer: done, headers=%d footers=%d across %d pages",
        header_count, footer_count, num_pages,
    )

    return result


def _cross_page_confirm(
    pages_lines: list[list[TextLine]],
    page_types: list[list[SemanticType]],
) -> list[list[SemanticType]]:
    """Confirm header/footer candidates by cross-page text matching.

    Architecture doc: adjacent page pairs (stride 1) AND stride-2 pairs.
    A candidate is confirmed if a matching text+position candidate exists
    on at least one adjacent page.  Unconfirmed header/footer candidates
    revert to PARAGRAPH.
    """
    num_pages = len(pages_lines)
    # Track which (page, line_idx) positions have at least one cross-page match
    confirmed: set[tuple[int, int]] = set()

    def _text_matches(a: TextLine, b: TextLine) -> bool:
        av = a.value.strip()
        bv = b.value.strip()
        # Exact match OR one is a page-number variant of the other
        if av == bv:
            return True
        # Allow numeric page numbers (different values but same position => footer page num)
        if av.isdigit() and bv.isdigit():
            return True
        return False

    def _position_close(a: TextLine, b: TextLine) -> bool:
        """Bounding boxes at roughly the same vertical position (ignore page num)."""
        ba = a.bounding_box
        bb = b.bounding_box
        if ba is None or bb is None:
            return False
        return abs(ba.bottom - bb.bottom) <= MAX_HEADER_FOOTER_GAP

    # Check strides 1 and 2
    for stride in (1, 2):
        for pg in range(num_pages - stride):
            pg2 = pg + stride
            for li, (line_a, type_a) in enumerate(zip(pages_lines[pg], page_types[pg])):
                if type_a == SemanticType.PARAGRAPH:
                    continue
                for lj, (line_b, type_b) in enumerate(
                    zip(pages_lines[pg2], page_types[pg2])
                ):
                    if type_b != type_a:
                        continue
                    if _text_matches(line_a, line_b) and _position_close(line_a, line_b):
                        confirmed.add((pg, li))
                        confirmed.add((pg2, lj))
                        logger.debug(
                            "_cross_page_confirm: matched pages %d/%d lines %d/%d as %s",
                            pg, pg2, li, lj, type_a.name,
                        )

    # Revert un-confirmed non-PARAGRAPH candidates to PARAGRAPH
    confirmed_types = [list(t) for t in page_types]
    for pg, types in enumerate(confirmed_types):
        for li, stype in enumerate(types):
            if stype != SemanticType.PARAGRAPH and (pg, li) not in confirmed:
                logger.debug(
                    "_cross_page_confirm: page=%d line=%d type=%s not confirmed, reverting to PARAGRAPH",
                    pg, li, stype.name,
                )
                confirmed_types[pg][li] = SemanticType.PARAGRAPH

    return confirmed_types


def _wrap_line_as_semantic_node(
    line: TextLine, stype: SemanticType
) -> SemanticTextNode:
    """Wrap a TextLine into a SemanticTextNode with the given type."""
    block = TextBlock()
    block.push(line)
    col = TextColumn()
    col.push(block)
    return SemanticTextNode(semantic_type=stype, columns=[col])
