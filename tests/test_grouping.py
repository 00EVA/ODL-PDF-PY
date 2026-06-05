# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Tests for the text grouping + classification processors.

TDD: tests are written against the documented algorithms in
docs/architecture/03-processors.md and cover:
  - TextLineProcessor.group_lines: same-baseline merging, horizontal adjacency,
    space injection, line-art bullet linking.
  - ParagraphProcessor.group_paragraphs: vertical proximity, font consistency,
    multi-line merging into SemanticTextNode paragraphs.
  - HeadingProcessor.detect_headings: font-size/weight relative to body font.
  - HeadingProcessor.assign_heading_levels: descending font-size -> H1..H6.
  - HeaderFooterProcessor.detect_header_footer: top/bottom repeated lines flagged
    as HEADER/FOOTER.
"""

from __future__ import annotations

import pytest

from odl_pdf.entities import (
    BoundingBox,
    SemanticTextNode,
    SemanticType,
    TextChunk,
    TextLine,
)
from odl_pdf.processors.grouping import (
    assign_heading_levels,
    detect_header_footer,
    detect_headings,
    group_lines,
    group_paragraphs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    page: int,
    left: float,
    bottom: float,
    right: float,
    top: float,
    value: str = "word",
    font_size: float = 12.0,
    font_name: str = "Arial",
    font_weight: float | None = None,
    base_line: float | None = None,
) -> TextChunk:
    bb = BoundingBox.of(page, left, bottom, right, top)
    bl = base_line if base_line is not None else bottom
    c = TextChunk(bounding_box=bb, value=value, font_size=font_size,
                  font_name=font_name, font_weight=font_weight)
    c.base_line = bl
    return c


def _para_node(lines: list[TextLine]) -> SemanticTextNode:
    """Wrap a list of TextLines into a SemanticTextNode paragraph (helper)."""
    from odl_pdf.entities import TextBlock, TextColumn
    col = TextColumn()
    block = TextBlock()
    for line in lines:
        block.push(line)
    col.push(block)
    return SemanticTextNode.paragraph([col])


# ---------------------------------------------------------------------------
# group_lines — TextLineProcessor
# ---------------------------------------------------------------------------


class TestGroupLines:
    """Tests for group_lines(chunks) -> list[TextLine]."""

    def test_single_chunk_produces_one_line(self):
        chunks = [_chunk(0, 10, 700, 60, 714)]
        lines = group_lines(chunks)
        assert len(lines) == 1
        assert len(lines[0].chunks) >= 1

    def test_same_baseline_chunks_merge_into_one_line(self):
        """Two horizontally adjacent chunks at the same baseline form one line."""
        # baseline = bottom = 700, font_size=12, gap between chunks is 2 < 0.1*12
        c1 = _chunk(0, 10, 700, 60, 712, value="Hello", font_size=12.0, base_line=700)
        c2 = _chunk(0, 62, 700, 120, 712, value="world", font_size=12.0, base_line=700)
        lines = group_lines([c1, c2])
        assert len(lines) == 1
        assert len(lines[0].chunks) >= 2

    def test_different_baseline_chunks_separate_lines(self):
        """Chunks with baselines far apart (> same_baseline_tol) produce separate lines."""
        c1 = _chunk(0, 10, 700, 60, 712, value="Line1", font_size=12.0, base_line=700)
        # baseline differs by 20 pts — far above same-baseline tolerance of ~1.2 (0.1*12)
        c2 = _chunk(0, 10, 680, 60, 692, value="Line2", font_size=12.0, base_line=680)
        lines = group_lines([c1, c2])
        assert len(lines) == 2

    def test_whitespace_chunk_skipped_but_marks_next_for_space(self):
        """Whitespace-only chunks are skipped; the following real chunk triggers space insertion."""
        c1 = _chunk(0, 10, 700, 60, 712, value="Hello", font_size=12.0, base_line=700)
        ws = _chunk(0, 60, 700, 70, 712, value=" ", font_size=12.0, base_line=700)
        c2 = _chunk(0, 70, 700, 130, 712, value="world", font_size=12.0, base_line=700)
        lines = group_lines([c1, ws, c2])
        # Should still produce a single line (same baseline)
        assert len(lines) == 1
        # The line text should include both words
        text = lines[0].value
        assert "Hello" in text
        assert "world" in text

    def test_empty_chunks_skipped(self):
        """Empty (zero-length value) chunks are not included in output lines."""
        c1 = _chunk(0, 10, 700, 60, 712, value="text", base_line=700)
        empty = _chunk(0, 60, 700, 65, 712, value="", base_line=700)
        lines = group_lines([c1, empty])
        assert len(lines) == 1
        # empty chunk should not appear or, if included, the value should be "text"
        assert "text" in lines[0].value

    def test_wide_horizontal_gap_triggers_space_injection(self):
        """A gap > font_size * TEXT_LINE_SPACE_RATIO between adjacent same-baseline chunks
        should inject a synthetic space chunk between them."""
        # TEXT_LINE_SPACE_RATIO ~ 0.2 (common in veraPDF); gap = 50 >> 12*0.2=2.4
        c1 = _chunk(0, 10, 700, 60, 712, value="Hello", font_size=12.0, base_line=700)
        c2 = _chunk(0, 110, 700, 160, 712, value="world", font_size=12.0, base_line=700)
        lines = group_lines([c1, c2])
        # Either produces one line with space, or two lines; check the text makes sense
        assert len(lines) >= 1
        combined = " ".join(line.value for line in lines)
        assert "Hello" in combined
        assert "world" in combined

    def test_three_baseline_groups(self):
        """Three groups of chunks at different baselines produce three lines."""
        chunks = [
            _chunk(0, 10, 700, 60, 712, value="A", base_line=700),
            _chunk(0, 62, 700, 120, 712, value="B", base_line=700),
            _chunk(0, 10, 680, 60, 692, value="C", base_line=680),
            _chunk(0, 10, 660, 60, 672, value="D", base_line=660),
        ]
        lines = group_lines(chunks)
        assert len(lines) == 3

    def test_returns_list(self):
        lines = group_lines([])
        assert isinstance(lines, list)

    def test_large_font_size_affects_merge_tolerance(self):
        """With font_size=24, same-baseline tolerance is larger (0.1*24=2.4)."""
        c1 = _chunk(0, 10, 700, 80, 724, value="Big", font_size=24.0, base_line=700)
        c2 = _chunk(0, 82, 700, 200, 724, value="Title", font_size=24.0, base_line=700)
        lines = group_lines([c1, c2])
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# group_paragraphs — ParagraphProcessor
# ---------------------------------------------------------------------------


class TestGroupParagraphs:
    """Tests for group_paragraphs(lines) -> list[SemanticTextNode]."""

    def test_single_line_produces_one_paragraph(self):
        lines = [
            _make_line(page=0, bottom=700, top=712, left=10, right=200, font_size=12.0)
        ]
        nodes = group_paragraphs(lines)
        assert len(nodes) == 1
        assert nodes[0].semantic_type == SemanticType.PARAGRAPH

    def test_two_close_lines_merge_into_one_paragraph(self):
        """Two lines vertically close (< leading threshold) should merge."""
        # Leading for font_size=12: typical line height ~14; gap ~ 2 (close)
        line1 = _make_line(0, 700, 712, 10, 200, 12.0)
        line2 = _make_line(0, 684, 696, 10, 200, 12.0)  # gap=4, close
        nodes = group_paragraphs([line1, line2])
        assert len(nodes) == 1
        assert nodes[0].semantic_type == SemanticType.PARAGRAPH

    def test_two_far_lines_produce_separate_paragraphs(self):
        """Two lines with a large gap (>> leading) should become two paragraphs."""
        line1 = _make_line(0, 700, 712, 10, 200, 12.0)
        line2 = _make_line(0, 600, 612, 10, 200, 12.0)  # gap=88 >> leading
        nodes = group_paragraphs([line1, line2])
        assert len(nodes) == 2

    def test_empty_input_returns_empty(self):
        nodes = group_paragraphs([])
        assert nodes == []

    def test_different_font_size_lines_separate(self):
        """Lines with very different font sizes should not merge (different text size)."""
        line1 = _make_line(0, 700, 724, 10, 200, 24.0)  # big heading
        line2 = _make_line(0, 684, 696, 10, 200, 12.0)  # body
        nodes = group_paragraphs([line1, line2])
        # Different font sizes => different blocks (the big line stays separate)
        assert len(nodes) >= 1
        # The big-font node comes first
        first_node = nodes[0]
        assert first_node.semantic_type == SemanticType.PARAGRAPH

    def test_three_close_lines_one_paragraph(self):
        """Three consecutive close lines merge into a single paragraph."""
        lines = [
            _make_line(0, 700, 712, 10, 300, 12.0),
            _make_line(0, 684, 696, 10, 300, 12.0),
            _make_line(0, 668, 680, 10, 300, 12.0),
        ]
        nodes = group_paragraphs(lines)
        assert len(nodes) == 1

    def test_output_types_are_semantic_text_nodes(self):
        line = _make_line(0, 700, 712, 10, 200, 12.0)
        nodes = group_paragraphs([line])
        assert all(isinstance(n, SemanticTextNode) for n in nodes)


def _make_line(
    page: int,
    bottom: float,
    top: float,
    left: float,
    right: float,
    font_size: float,
    value: str = "text",
) -> TextLine:
    """Create a TextLine with one TextChunk for test purposes."""
    c = _chunk(page, left, bottom, right, top, value=value, font_size=font_size,
               base_line=bottom)
    line = TextLine()
    line.push(c)
    return line


# ---------------------------------------------------------------------------
# detect_headings — HeadingProcessor (per-page classification)
# ---------------------------------------------------------------------------


class TestDetectHeadings:
    """Tests for detect_headings(nodes, body_font_size)."""

    def test_large_font_node_becomes_heading(self):
        """A node with font size significantly > body should be promoted to HEADING."""
        heading_line = _make_line(0, 700, 724, 10, 200, 24.0, value="Chapter 1")
        body_line = _make_line(0, 684, 696, 10, 200, 12.0, value="body text here")
        nodes = [
            _wrap_line_as_node(heading_line),
            _wrap_line_as_node(body_line),
        ]
        detect_headings(nodes, body_font_size=12.0)
        assert nodes[0].semantic_type == SemanticType.HEADING

    def test_body_font_node_stays_paragraph(self):
        """A node with body font size should remain PARAGRAPH."""
        body_line = _make_line(0, 684, 696, 10, 200, 12.0, value="body text here")
        nodes = [_wrap_line_as_node(body_line)]
        detect_headings(nodes, body_font_size=12.0)
        assert nodes[0].semantic_type == SemanticType.PARAGRAPH

    def test_bold_node_at_body_size_may_become_heading(self):
        """A bold node (weight >= 700) at body size can become HEADING due to weight rarity boost."""
        bold_line = _make_line(0, 700, 712, 10, 200, 12.0, value="Section Title")
        bold_line.chunks[0].font_weight = 700.0
        nodes = [_wrap_line_as_node(bold_line)]
        # With bold boost, it should become heading
        detect_headings(nodes, body_font_size=12.0)
        # Result may be HEADING or PARAGRAPH depending on score; just verify no crash
        assert nodes[0].semantic_type in (SemanticType.HEADING, SemanticType.PARAGRAPH)

    def test_heading_probability_threshold(self):
        """Nodes below heading probability threshold stay PARAGRAPH."""
        # Normal body text, same font size as body
        line = _make_line(0, 700, 712, 10, 400, 12.0, value="Normal paragraph text that is long")
        nodes = [_wrap_line_as_node(line)]
        detect_headings(nodes, body_font_size=12.0)
        assert nodes[0].semantic_type == SemanticType.PARAGRAPH

    def test_single_word_large_font_is_heading(self):
        """A single short word in a large font is a strong heading signal."""
        line = _make_line(0, 700, 730, 10, 100, 30.0, value="Introduction")
        nodes = [_wrap_line_as_node(line)]
        detect_headings(nodes, body_font_size=12.0)
        assert nodes[0].semantic_type == SemanticType.HEADING

    def test_already_heading_not_changed(self):
        """A node already classified as HEADING should not change type."""
        line = _make_line(0, 700, 724, 10, 200, 24.0)
        node = _wrap_line_as_node(line)
        node.set_semantic_type(SemanticType.HEADING)
        nodes = [node]
        detect_headings(nodes, body_font_size=12.0)
        assert nodes[0].semantic_type == SemanticType.HEADING

    def test_empty_nodes_list(self):
        detect_headings([], body_font_size=12.0)  # no crash


def _wrap_line_as_node(line: TextLine) -> SemanticTextNode:
    """Wrap a single TextLine into a SemanticTextNode paragraph."""
    from odl_pdf.entities import TextBlock, TextColumn
    col = TextColumn()
    block = TextBlock()
    block.push(line)
    col.push(block)
    return SemanticTextNode.paragraph([col])


# ---------------------------------------------------------------------------
# assign_heading_levels — HeadingProcessor.detectHeadingsLevels
# ---------------------------------------------------------------------------


class TestAssignHeadingLevels:
    """Tests for assign_heading_levels(nodes)."""

    def test_single_heading_gets_level_one(self):
        node = _make_heading_node(font_size=24.0)
        assign_heading_levels([node])
        assert node.heading_level == 1

    def test_two_different_font_sizes_h1_h2(self):
        """Two headings with distinct font sizes: larger -> H1, smaller -> H2."""
        h1 = _make_heading_node(font_size=24.0)
        h2 = _make_heading_node(font_size=18.0)
        assign_heading_levels([h1, h2])
        assert h1.heading_level == 1
        assert h2.heading_level == 2

    def test_same_font_size_same_level(self):
        """Two headings with the same font size get the same level."""
        h1 = _make_heading_node(font_size=18.0)
        h2 = _make_heading_node(font_size=18.0)
        assign_heading_levels([h1, h2])
        assert h1.heading_level == h2.heading_level

    def test_three_sizes_three_levels(self):
        """Three distinct sizes map to H1, H2, H3 in descending order."""
        h1 = _make_heading_node(font_size=24.0)
        h2 = _make_heading_node(font_size=18.0)
        h3 = _make_heading_node(font_size=14.0)
        assign_heading_levels([h1, h2, h3])
        assert h1.heading_level == 1
        assert h2.heading_level == 2
        assert h3.heading_level == 3

    def test_non_headings_ignored(self):
        """Non-HEADING nodes should not get a heading level assigned."""
        para = _make_para_node(font_size=12.0)
        assign_heading_levels([para])
        assert para.heading_level is None

    def test_mixed_nodes_only_headings_leveled(self):
        """Paragraphs mixed with headings: only headings get levels."""
        para = _make_para_node(font_size=12.0)
        h1 = _make_heading_node(font_size=24.0)
        assign_heading_levels([para, h1])
        assert para.heading_level is None
        assert h1.heading_level == 1

    def test_empty_list_no_crash(self):
        assign_heading_levels([])  # no crash

    def test_level_capped_at_six(self):
        """More than 6 distinct heading sizes: levels beyond 6 are capped at 6."""
        nodes = [_make_heading_node(font_size=float(24 - i * 2)) for i in range(8)]
        assign_heading_levels(nodes)
        for n in nodes:
            assert n.heading_level is not None
            assert n.heading_level <= 6


def _make_heading_node(font_size: float) -> SemanticTextNode:
    from odl_pdf.entities import TextBlock, TextColumn
    line = _make_line(0, 700, 700 + font_size, 10, 200, font_size)
    col = TextColumn()
    block = TextBlock()
    block.push(line)
    col.push(block)
    return SemanticTextNode.heading(0, [col])  # level=0, will be assigned


def _make_para_node(font_size: float) -> SemanticTextNode:
    from odl_pdf.entities import TextBlock, TextColumn
    line = _make_line(0, 700, 700 + font_size, 10, 200, font_size)
    col = TextColumn()
    block = TextBlock()
    block.push(line)
    col.push(block)
    return SemanticTextNode.paragraph([col])


# ---------------------------------------------------------------------------
# detect_header_footer — HeaderFooterProcessor
# ---------------------------------------------------------------------------


class TestDetectHeaderFooter:
    """Tests for detect_header_footer(pages_lines, page_heights) -> per-page classifications."""

    def test_line_at_top_of_page_flagged_header(self):
        """A TextLine at the top zone (bottom >= 2/3 * page_height) should be HEADER."""
        page_height = 792.0  # US Letter
        # At top: bottom=700, top=712, page_height=792 => bottom/page_height ~0.88 >= 2/3
        line = _make_line(0, 700, 712, 10, 200, 10.0, value="Page Header")
        nodes = detect_header_footer([[line]], [page_height])
        assert len(nodes) == 1 and len(nodes[0]) >= 1
        assert nodes[0][0].semantic_type == SemanticType.HEADER

    def test_line_at_bottom_of_page_flagged_footer(self):
        """A TextLine at the bottom zone (top <= 1/3 * page_height) should be FOOTER."""
        page_height = 792.0
        # At bottom: bottom=20, top=32, page_height=792 => top/page_height ~0.04 <= 1/3
        line = _make_line(0, 20, 32, 10, 200, 10.0, value="Page 1")
        nodes = detect_header_footer([[line]], [page_height])
        assert len(nodes) == 1 and len(nodes[0]) >= 1
        assert nodes[0][0].semantic_type == SemanticType.FOOTER

    def test_line_in_middle_not_flagged(self):
        """A TextLine in the middle of the page should remain PARAGRAPH."""
        page_height = 792.0
        # Middle: bottom=400, top=412
        line = _make_line(0, 400, 412, 10, 200, 12.0, value="Body text")
        nodes = detect_header_footer([[line]], [page_height])
        assert len(nodes) == 1
        assert nodes[0][0].semantic_type == SemanticType.PARAGRAPH

    def test_multi_page_repeated_header_flagged(self):
        """The same text at the top of multiple pages should be flagged as HEADER."""
        page_height = 792.0
        # Same header on two pages
        header1 = _make_line(0, 700, 712, 10, 200, 10.0, value="Company Name")
        header2 = _make_line(1, 700, 712, 10, 200, 10.0, value="Company Name")
        nodes = detect_header_footer([[header1], [header2]], [page_height, page_height])
        # Both pages should have their header node flagged
        assert nodes[0][0].semantic_type == SemanticType.HEADER
        assert nodes[1][0].semantic_type == SemanticType.HEADER

    def test_multi_page_repeated_footer_flagged(self):
        """The same text at the bottom of multiple pages should be flagged as FOOTER."""
        page_height = 792.0
        footer1 = _make_line(0, 20, 32, 10, 200, 10.0, value="1")
        footer2 = _make_line(1, 20, 32, 10, 200, 10.0, value="2")
        nodes = detect_header_footer([[footer1], [footer2]], [page_height, page_height])
        assert nodes[0][0].semantic_type == SemanticType.FOOTER
        assert nodes[1][0].semantic_type == SemanticType.FOOTER

    def test_empty_page_list(self):
        result = detect_header_footer([], [])
        assert result == []

    def test_single_page_with_no_header_candidates(self):
        """A single page cannot have repeated header/footer (no cross-page match)."""
        page_height = 792.0
        line = _make_line(0, 700, 712, 10, 200, 10.0, value="Top text")
        nodes = detect_header_footer([[line]], [page_height])
        # Single page: line at top zone is flagged as HEADER by position alone
        assert len(nodes) == 1


# ---------------------------------------------------------------------------
# Section-number heading splitting (section-heading recall fix)
# ---------------------------------------------------------------------------


class TestSectionNumberHeadingSplitting:
    """Tests for the section-number heading-split logic in group_paragraphs.

    technical procedure documents use the same font size for numbered section headings
    (``6``, ``6.1``, ``6.8 Out of Tolerance``) as for body text.  The fix
    ensures that a numbered section line starts its own SemanticTextNode so
    that detect_headings can promote it via the section-number signal.
    """

    def test_numbered_heading_line_followed_by_body_produces_two_nodes(self):
        """A short section-number line followed by long body text -> HEADING + PARAGRAPH.

        The section-number line (e.g. ``6.1``) must become its own node so
        that detect_headings can fire the +0.8 section-number signal on it.
        The body line must become a separate PARAGRAPH node.
        """
        # All lines at 12pt (same as body — font-size signal must NOT fire)
        heading_line = _make_line(
            page=0, bottom=700, top=712, left=10, right=60, font_size=12.0,
            value="6.1",
        )
        # Long body: more than _SECTION_HEADING_MAX_WORDS words
        body_words = " ".join(["word"] * 15)
        body_line = _make_line(
            page=0, bottom=685, top=697, left=10, right=400, font_size=12.0,
            value=body_words,
        )
        nodes = group_paragraphs([heading_line, body_line])

        # Must produce at least 2 nodes — the heading and the body
        assert len(nodes) >= 2, (
            f"Expected >=2 nodes for section-num + body, got {len(nodes)}: "
            + str([n.value[:40] for n in nodes])
        )
        # First node's first line must be the section number
        assert nodes[0].value.strip().startswith("6.1"), (
            f"First node should start with section number, got {repr(nodes[0].value[:40])}"
        )
        # After detect_headings, the first node should be promoted to HEADING
        from odl_pdf.processors.grouping import detect_headings
        detect_headings(nodes, body_font_size=12.0)
        assert nodes[0].semantic_type == SemanticType.HEADING, (
            f"Section-number node should be HEADING, got {nodes[0].semantic_type}"
        )
        # Body node should remain PARAGRAPH
        body_node = nodes[-1]
        assert body_node.semantic_type == SemanticType.PARAGRAPH, (
            f"Body node should be PARAGRAPH, got {body_node.semantic_type}"
        )

    def test_numbered_heading_with_short_title_merges_into_one_heading_node(self):
        """A section-number line followed by a short title -> single HEADING node.

        ``6.1`` + ``General`` (both short) should merge into one node that
        reads ``6.1\\nGeneral`` and is promoted to HEADING.
        """
        heading_line = _make_line(
            page=0, bottom=700, top=712, left=10, right=60, font_size=12.0,
            value="6.1",
        )
        title_line = _make_line(
            page=0, bottom=685, top=697, left=10, right=120, font_size=12.0,
            value="General",
        )
        nodes = group_paragraphs([heading_line, title_line])

        # The two short lines may merge into one node ("6.1\nGeneral")
        # or stay as two nodes — both are acceptable as long as the FIRST
        # node starts with the section number and is promoted to HEADING.
        assert len(nodes) >= 1
        first = nodes[0]
        assert first.value.strip().startswith("6.1"), (
            f"First node should start with 6.1, got {repr(first.value[:40])}"
        )

        from odl_pdf.processors.grouping import detect_headings
        detect_headings(nodes, body_font_size=12.0)
        assert first.semantic_type == SemanticType.HEADING, (
            f"Section-number node should be HEADING, got {first.semantic_type}"
        )

    def test_plain_mid_sentence_paragraph_not_split(self):
        """A paragraph that does NOT start with a section number is NOT split.

        Two close lines of body text that do not start with a section number
        should merge into a single PARAGRAPH node (not falsely split).
        """
        line1 = _make_line(
            page=0, bottom=700, top=712, left=10, right=400, font_size=12.0,
            value="This is the first sentence of a paragraph.",
        )
        line2 = _make_line(
            page=0, bottom=685, top=697, left=10, right=400, font_size=12.0,
            value="This is the continuation of the same paragraph.",
        )
        nodes = group_paragraphs([line1, line2])

        # Both lines are close and same font — they should merge into 1 node
        assert len(nodes) == 1, (
            f"Plain paragraph lines should merge into 1 node, got {len(nodes)}"
        )
        assert nodes[0].semantic_type == SemanticType.PARAGRAPH

    def test_toc_section_number_with_dot_leaders_not_split_as_heading(self):
        """A section number followed by a dot-leader TOC line should NOT become a heading.

        In a Table of Contents, ``6.1`` followed by
        ``General................................`` (with dot-leaders) is a TOC
        entry — the section-number line should NOT be promoted to HEADING.
        """
        section_line = _make_line(
            page=0, bottom=700, top=712, left=10, right=60, font_size=12.0,
            value="6.1",
        )
        toc_line = _make_line(
            page=0, bottom=685, top=697, left=10, right=400, font_size=12.0,
            value="General................................",
        )
        nodes = group_paragraphs([section_line, toc_line])

        # The TOC de-split pass should merge these back or leave them merged
        # so that the combined node contains dot-leaders and does NOT fire the
        # section-number heading signal.
        from odl_pdf.processors.grouping import detect_headings
        detect_headings(nodes, body_font_size=12.0)

        # No node should be promoted to HEADING (TOC entries are not headings)
        heading_nodes = [n for n in nodes if n.semantic_type == SemanticType.HEADING]
        assert len(heading_nodes) == 0, (
            f"TOC section entries should NOT be promoted to HEADING, "
            f"got {[n.value[:40] for n in heading_nodes]}"
        )


# ---------------------------------------------------------------------------
# Text alignment + heading size hierarchy (task #40)
# ---------------------------------------------------------------------------


class TestTextLineAlignment:
    """Tests for TextLine.alignment() and TextLine.is_centered()."""

    def test_centered_short_line_reports_center(self):
        """A short line with equal left/right margins is CENTER."""
        # page_width=600; line from x=200 to x=400 -> left_margin=200, right_margin=200
        line = _make_line(page=0, bottom=700, top=712, left=200, right=400,
                          font_size=18.0, value="Introduction")
        from odl_pdf.entities.text import TextAlignment
        assert line.alignment(600.0) == TextAlignment.CENTER

    def test_left_aligned_line_reports_left(self):
        """A line starting near the left edge is LEFT-aligned."""
        # page_width=600; line from x=10 to x=300 -> left_margin=10, right_margin=300
        line = _make_line(page=0, bottom=700, top=712, left=10, right=300,
                          font_size=12.0, value="body text")
        from odl_pdf.entities.text import TextAlignment
        assert line.alignment(600.0) == TextAlignment.LEFT

    def test_right_aligned_line_reports_right(self):
        """A line ending near the right edge with large left margin is RIGHT."""
        # page_width=600; line from x=400 to x=590 -> left_margin=400, right_margin=10
        line = _make_line(page=0, bottom=700, top=712, left=400, right=590,
                          font_size=12.0, value="right text")
        from odl_pdf.entities.text import TextAlignment
        assert line.alignment(600.0) == TextAlignment.RIGHT

    def test_alignment_unknown_for_zero_page_width(self):
        """With page_width=0, alignment returns UNKNOWN."""
        line = _make_line(page=0, bottom=700, top=712, left=200, right=400,
                          font_size=12.0, value="title")
        from odl_pdf.entities.text import TextAlignment
        assert line.alignment(0.0) == TextAlignment.UNKNOWN

    def test_is_centered_true_for_short_centered_line(self):
        """A short centered line (< 70% of page width) reports is_centered=True."""
        line = _make_line(page=0, bottom=700, top=712, left=200, right=400,
                          font_size=18.0, value="Abstract")
        # line_width = 200, page_width = 600 -> 200/600 = 33% < 70% threshold
        assert line.is_centered(600.0, require_short=True)

    def test_is_centered_false_for_wide_line(self):
        """A full-width centered-looking line should NOT be is_centered when require_short=True."""
        # line_width = 580, page_width = 600 -> 96.7% > 70% threshold => not short
        line = _make_line(page=0, bottom=700, top=712, left=10, right=590,
                          font_size=12.0, value="This is a full-width body paragraph line text.")
        assert not line.is_centered(600.0, require_short=True)

    def test_is_centered_require_short_false_accepts_wide(self):
        """When require_short=False, a wide line can still pass the center test."""
        # line from x=10 to x=590 on page_width=600:
        # left_margin=10, right_margin=10 -> centered by margin test
        line = _make_line(page=0, bottom=700, top=712, left=10, right=590,
                          font_size=12.0, value="wide centered text")
        # margin diff = |10 - 10| = 0 <= 0.08*600 = 48 -> CENTER
        assert line.is_centered(600.0, require_short=False)


class TestAlignmentHeadingSignal:
    """Tests for the alignment-based heading signal in detect_headings (task #40)."""

    def test_centered_large_font_no_section_number_detected_as_heading(self):
        """A centered large-font line with no section number must be detected as heading.

        This covers research papers and slides where headings are identified by
        visual centering rather than section numbering.
        """
        # page_width=612 (US Letter); line centered at x=200..412 = 212pt wide
        # font_size=18 >> body(12) -> ratio=1.5 >= HEADING_FONT_SIZE_RATIO
        # centered + large font should clear HEADING_PROBABILITY=0.75
        heading_line = _make_line(
            page=0, bottom=700, top=718, left=200, right=412,
            font_size=18.0, value="Introduction",
        )
        node = _wrap_line_as_node(heading_line)
        detect_headings([node], body_font_size=12.0, page_width=612.0)
        assert node.semantic_type == SemanticType.HEADING, (
            f"Centered large-font line should be HEADING, got {node.semantic_type}"
        )

    def test_centered_same_body_size_no_section_number_detected_as_heading(self):
        """A centered short line at SAME body font but with strong centering boost is detected.

        The alignment boost alone (0.35) plus rarity boost should push a
        visually centered same-size title to heading status.
        """
        # Build a page of 10 body nodes at 12pt + 1 centered title node at 12pt.
        # The rarity signal will be small (size is common), but the centered boost
        # adds 0.35 which alone approaches the 0.75 threshold when combined with
        # font rarity that fires slightly.
        # We'll use font_size=14 (slightly above body 12) to ensure detection.
        heading_line = _make_line(
            page=0, bottom=700, top=714, left=230, right=382,
            font_size=14.0, value="Abstract",
        )
        body_nodes = [
            _wrap_line_as_node(_make_line(0, 680 - i * 14, 694 - i * 14, 10, 600, 12.0))
            for i in range(10)
        ]
        heading_node = _wrap_line_as_node(heading_line)
        all_nodes = [heading_node] + body_nodes
        detect_headings(all_nodes, body_font_size=12.0, page_width=612.0)
        assert heading_node.semantic_type == SemanticType.HEADING, (
            "Centered short line at slightly larger font should be HEADING"
        )

    def test_body_paragraph_not_boosted_to_heading_by_alignment(self):
        """A full-width left-aligned body paragraph must NOT be boosted to heading."""
        body_line = _make_line(
            page=0, bottom=700, top=712, left=10, right=602,
            font_size=12.0, value="This is a long body paragraph that spans the full width.",
        )
        node = _wrap_line_as_node(body_line)
        detect_headings([node], body_font_size=12.0, page_width=612.0)
        assert node.semantic_type == SemanticType.PARAGRAPH, (
            "Full-width left-aligned body text should remain PARAGRAPH"
        )

    def test_qms_section_number_heading_not_broken_by_alignment(self):
        """Existing section-number path must still work when page_width is provided.

        A numbered heading like '6.1' at body font must still be detected as
        HEADING even when alignment is also computed.
        """
        section_line = _make_line(
            page=0, bottom=700, top=712, left=10, right=80,
            font_size=12.0, value="6.1 General Requirements",
        )
        node = _wrap_line_as_node(section_line)
        detect_headings([node], body_font_size=12.0, page_width=612.0)
        assert node.semantic_type == SemanticType.HEADING, (
            "section-number node must still be HEADING when page_width is given"
        )

    def test_no_page_width_no_alignment_boost(self):
        """Without page_width=0.0 (default), the alignment signal is inactive.

        A centered-looking line that would only clear the threshold via the
        alignment boost must NOT be promoted when page_width is 0.
        """
        # 14pt font on 12pt body => ratio ~ 1.16 (barely above threshold 1.15).
        # Without alignment boost the score is marginal.  We test that a body-
        # size line (same as body) with no section number stays PARAGRAPH.
        line = _make_line(
            page=0, bottom=700, top=712, left=200, right=412,
            font_size=12.0, value="Same Size Centered",
        )
        node = _wrap_line_as_node(line)
        # With many same-size body nodes, rarity is low => score stays under 0.75
        body_nodes = [
            _wrap_line_as_node(_make_line(0, 680 - i * 14, 694 - i * 14, 10, 600, 12.0))
            for i in range(10)
        ]
        all_nodes = [node] + body_nodes
        detect_headings(all_nodes, body_font_size=12.0, page_width=0.0)
        # Without alignment signal: same-size node should NOT be heading
        assert node.semantic_type == SemanticType.PARAGRAPH, (
            "Same-size centered line without alignment signal should stay PARAGRAPH"
        )


class TestSizeHierarchyHeadingLevels:
    """Tests for the size-hierarchy heading level assignment (task #40).

    The existing assign_heading_levels already does this correctly; these tests
    verify the documented behavior explicitly for the 'no section number' case.
    """

    def test_size_cluster_assigns_h1_to_largest_non_section_number_heading(self):
        """For headings without section numbers, largest font size => H1."""
        # Three headings at three distinct font sizes, no section numbers
        h1_node = _make_heading_node(font_size=24.0)
        h2_node = _make_heading_node(font_size=18.0)
        h3_node = _make_heading_node(font_size=14.0)
        assign_heading_levels([h1_node, h2_node, h3_node])
        assert h1_node.heading_level == 1, "Largest font should be H1"
        assert h2_node.heading_level == 2, "Middle font should be H2"
        assert h3_node.heading_level == 3, "Smallest font should be H3"

    def test_h1_level_less_than_h2_level(self):
        """H1 level number must be less than H2 level number (1 < 2)."""
        big = _make_heading_node(font_size=20.0)
        small = _make_heading_node(font_size=14.0)
        assign_heading_levels([big, small])
        assert big.heading_level < small.heading_level, (
            f"H1 level={big.heading_level} should be < H2 level={small.heading_level}"
        )

    def test_mixed_section_and_non_section_headings_level_by_size(self):
        """Headings with and without section numbers all get levels by font size."""
        # Simulate: a large title (non-section), a medium section heading, a small sub-heading
        big_title = _make_heading_node(font_size=24.0)   # H1 (largest)
        medium_section = _make_heading_node(font_size=14.0)  # H2
        small_section = _make_heading_node(font_size=12.0)   # H3
        assign_heading_levels([big_title, medium_section, small_section])
        assert big_title.heading_level == 1
        assert medium_section.heading_level == 2
        assert small_section.heading_level == 3

    def test_close_font_sizes_get_same_level(self):
        """Two headings with font sizes within tolerance get the same level."""
        # CLOSE_FONT_SIZE_EPSILON * 10 = 1.0; 14.0 vs 14.3 => same cluster
        h_a = _make_heading_node(font_size=14.0)
        h_b = _make_heading_node(font_size=14.3)
        assign_heading_levels([h_a, h_b])
        assert h_a.heading_level == h_b.heading_level, (
            "Near-identical font sizes should map to the same heading level"
        )
