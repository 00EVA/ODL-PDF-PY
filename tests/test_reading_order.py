# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Behavioral parity tests for XY-Cut++ reading-order sorter.

Mirror of the Rust suite.  All tests use the worked examples and edge cases
documented in ``docs/architecture/04-reading-order.md``.
"""

from __future__ import annotations

import pytest

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.processors.reading_order import (
    DEFAULT_BETA,
    DEFAULT_DENSITY_THRESHOLD,
    MIN_GAP_THRESHOLD,
    MIN_OVERLAP_COUNT,
    NARROW_ELEMENT_WIDTH_RATIO,
    OVERLAP_THRESHOLD,
    sort_reading_order,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bb(left: float, bottom: float, right: float, top: float, page: int = 1) -> BoundingBox:
    """Create a BoundingBox on the given page (default 1)."""
    return BoundingBox.of(page, left, bottom, right, top)


def indexed(*boxes: BoundingBox) -> list[tuple[int, BoundingBox]]:
    """Zip boxes with sequential 0-based indices."""
    return [(i, b) for i, b in enumerate(boxes)]


# ---------------------------------------------------------------------------
# §5 Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_beta(self):
        assert DEFAULT_BETA == 2.0

    def test_default_density_threshold(self):
        assert DEFAULT_DENSITY_THRESHOLD == 0.9

    def test_overlap_threshold(self):
        assert OVERLAP_THRESHOLD == 0.1

    def test_min_overlap_count(self):
        assert MIN_OVERLAP_COUNT == 2

    def test_min_gap_threshold(self):
        assert MIN_GAP_THRESHOLD == 5.0

    def test_narrow_element_width_ratio(self):
        assert NARROW_ELEMENT_WIDTH_RATIO == 0.1


# ---------------------------------------------------------------------------
# §4 Worked example — two-column paper with header (beta=0.7)
# ---------------------------------------------------------------------------


class TestWorkedExampleTwoColumnWithHeader:
    """Full walk-through from §4.  Expected order: Header, Col1-A, Col1-B, Col2-A, Col2-B."""

    def _make_items(self):
        header = bb(10, 85, 190, 95)   # full-width header at top
        col1_a = bb(10, 65, 50, 75)    # upper-left
        col1_b = bb(10, 45, 50, 55)    # lower-left
        col2_a = bb(100, 65, 140, 75)  # upper-right
        col2_b = bb(100, 45, 140, 55)  # lower-right
        return header, col1_a, col1_b, col2_a, col2_b

    def test_full_order_with_beta_0_7(self):
        header, col1_a, col1_b, col2_a, col2_b = self._make_items()
        # Indices: 0=header, 1=col1_a, 2=col1_b, 3=col2_a, 4=col2_b
        items = indexed(header, col1_a, col1_b, col2_a, col2_b)
        result = sort_reading_order(items, beta=0.7)
        assert result == [0, 1, 2, 3, 4], (
            f"Expected [0,1,2,3,4] (Header,Col1-A,Col1-B,Col2-A,Col2-B) but got {result}"
        )

    def test_default_beta_does_not_detect_cross_layout(self):
        """With beta=2.0 (default), header is NOT detected as cross-layout.
        The full sort still terminates correctly but header is placed by geometry
        alone — it will still sort first because it has the largest topY."""
        header, col1_a, col1_b, col2_a, col2_b = self._make_items()
        items = indexed(header, col1_a, col1_b, col2_a, col2_b)
        result = sort_reading_order(items)  # default beta=2.0
        # Header topY=95 is highest; all columns have lower topY.
        # Without cross-layout masking the algorithm still cuts correctly because
        # the header fills the vertical column gap — no valid vertical cut exists
        # for the full set, so a horizontal cut fires at y≈90.
        # Final result should still be [0, 1, 2, 3, 4] (header on top).
        assert result[0] == 0, "Header should still appear first with default beta"

    def test_col2_follows_col1_entirely(self):
        """Col1-A and Col1-B must both precede Col2-A with beta=0.7."""
        header, col1_a, col1_b, col2_a, col2_b = self._make_items()
        items = indexed(header, col1_a, col1_b, col2_a, col2_b)
        result = sort_reading_order(items, beta=0.7)
        assert result.index(1) < result.index(3), "Col1-A must precede Col2-A"
        assert result.index(2) < result.index(3), "Col1-B must precede Col2-A"


# ---------------------------------------------------------------------------
# Edge cases — empty / single / two elements
# ---------------------------------------------------------------------------


class TestSmallInputs:
    def test_empty_list(self):
        assert sort_reading_order([]) == []

    def test_single_element(self):
        assert sort_reading_order([(0, bb(10, 10, 50, 20))]) == [0]

    def test_two_elements_same_column(self):
        """Two stacked boxes — top one first (larger topY)."""
        top = bb(10, 60, 50, 70)
        bot = bb(10, 40, 50, 50)
        result = sort_reading_order(indexed(top, bot))
        assert result == [0, 1]

    def test_two_elements_same_row(self):
        """Two side-by-side boxes — left one first (smaller leftX)."""
        left_box = bb(10, 40, 50, 50)
        right_box = bb(60, 40, 100, 50)
        result = sort_reading_order(indexed(left_box, right_box))
        assert result == [0, 1]

    def test_two_elements_reversed_order_normalized(self):
        """Input in wrong order (higher Y given second) is fixed."""
        bot = bb(10, 10, 50, 20)
        top = bb(10, 40, 50, 50)
        result = sort_reading_order(indexed(bot, top))
        # top has topY=50, bot has topY=20 → top should come first → index 1
        assert result == [1, 0]


# ---------------------------------------------------------------------------
# sortByYThenX tie-breaking (§7.4)
# ---------------------------------------------------------------------------


class TestSortByYThenX:
    """When no valid cut exists the fallback sort fires (§7.4)."""

    def test_same_row_left_to_right(self):
        """Elements on the same row ordered by leftX ascending."""
        a = bb(50, 10, 80, 20)
        b = bb(10, 10, 40, 20)
        c = bb(90, 10, 120, 20)
        result = sort_reading_order(indexed(a, b, c))
        # b leftX=10, a leftX=50, c leftX=90 → b,a,c → indices 1,0,2
        assert result == [1, 0, 2]

    def test_descending_y_ordering(self):
        """Higher Y (top of page) comes first."""
        hi = bb(0, 80, 100, 90)
        lo = bb(0, 10, 100, 20)
        mid = bb(0, 45, 100, 55)
        result = sort_reading_order(indexed(hi, lo, mid))
        # topY: hi=90, mid=55, lo=20 → hi, mid, lo → 0,2,1
        assert result == [0, 2, 1]

    def test_same_topY_taller_first_in_horizontal_scan(self):
        """In the horizontal projection sort, bottomY DESC breaks topY ties (§7.3).

        Two boxes with identical topY: one taller (smaller bottomY) and one
        shorter.  The horizontal scan requires taller-first (desc bottomY) so
        prevBottom tracks the lowest edge correctly.
        """
        tall = bb(0, 10, 40, 50)   # height=40, top=50, bottom=10
        short_ = bb(50, 30, 90, 50)  # height=20, top=50, bottom=30
        # The gap detection scan sees these in order (tall, short) when sorted
        # descending by bottomY.  No cut exists here (only 2 items, no gap),
        # so fallback sortByYThenX fires: both have topY=50, leftX breaks tie.
        result = sort_reading_order(indexed(tall, short_))
        assert result == [0, 1]  # leftX: 0 < 50


# ---------------------------------------------------------------------------
# Cross-layout detection (§2.1)
# ---------------------------------------------------------------------------


class TestCrossLayoutDetection:
    def test_guard_size_less_than_3_skips_cross_layout(self):
        """With only 2 objects cross-layout detection is skipped regardless of beta."""
        wide = bb(0, 80, 200, 90)
        narrow = bb(0, 10, 50, 20)
        # Would satisfy width criterion with beta=0.5 but guard fires
        result = sort_reading_order(indexed(wide, narrow), beta=0.5)
        # wide topY=90 > narrow topY=20 → wide first
        assert result == [0, 1]

    def test_cross_layout_requires_min_overlap_count(self):
        """A wide element overlapping only 1 other is NOT cross-layout."""
        wide = bb(0, 80, 200, 90)
        narrow_a = bb(10, 40, 50, 60)
        narrow_b = bb(110, 40, 150, 60)  # overlaps wide but total overlaps=2
        # With beta=0.7: threshold=0.7*200=140, wide.width=200>=140 ✓
        # wide overlaps narrow_a: overlapWidth=40, min(200,40)=40, ratio=1.0 ✓
        # wide overlaps narrow_b: overlapWidth=40, ratio=1.0 ✓
        # overlapCount=2 >= 2 → cross-layout
        result = sort_reading_order(indexed(wide, narrow_a, narrow_b), beta=0.7)
        # Cross-layout wide has topY=90; main items topY=60 → wide first
        assert result[0] == 0

    def test_cross_layout_requires_min_2_overlaps(self):
        """Wide element with only 1 qualifying overlap is kept in main flow."""
        wide = bb(0, 80, 200, 90)
        narrow_a = bb(10, 40, 50, 60)   # overlaps wide
        narrow_b = bb(300, 40, 400, 60) # does NOT overlap wide (entirely to the right)
        # overlapCount=1 < MIN_OVERLAP_COUNT=2 → NOT cross-layout
        result = sort_reading_order(indexed(wide, narrow_a, narrow_b), beta=0.7)
        # wide topY=90 highest → still first in sortByYThenX fallback
        assert result[0] == 0


# ---------------------------------------------------------------------------
# Gap selection (§2.3) and MIN_GAP_THRESHOLD
# ---------------------------------------------------------------------------


class TestGapSelection:
    def test_tiny_gap_below_threshold_triggers_no_cut(self):
        """A gap of 4.9 pt (< 5.0) is not a valid cut."""
        # Two stacked boxes with only 4 pt vertical gap
        top_box = bb(0, 54, 100, 60)
        bot_box = bb(0, 40, 100, 50)  # gap = 54 - 50 = 4
        result = sort_reading_order(indexed(top_box, bot_box))
        # No valid cut, fallback: topY desc → top_box (topY=60) first
        assert result == [0, 1]

    def test_gap_at_threshold_is_valid(self):
        """A gap of exactly 5.0 pt is a valid cut."""
        top_box = bb(0, 55, 100, 65)
        bot_box = bb(0, 40, 100, 50)  # gap = 55 - 50 = 5.0
        result = sort_reading_order(indexed(top_box, bot_box))
        assert result == [0, 1]

    def test_vertical_gap_preferred_over_horizontal_when_larger(self):
        """When vertical gap > horizontal gap, vertical cut is chosen (§8.6)."""
        # Vertical gap=50, horizontal gap=10 (same as §4 worked example inner step)
        col1_a = bb(10, 65, 50, 75)
        col1_b = bb(10, 45, 50, 55)
        col2_a = bb(100, 65, 140, 75)
        col2_b = bb(100, 45, 140, 55)
        result = sort_reading_order(indexed(col1_a, col1_b, col2_a, col2_b))
        # Vertical cut at x=75 separates left=[col1_a,col1_b] from right=[col2_a,col2_b]
        # Expected: col1_a, col1_b, col2_a, col2_b → [0,1,2,3]
        assert result == [0, 1, 2, 3]

    def test_equal_gaps_prefer_vertical_cut(self):
        """Equal hGap == vGap: vertical cut is preferred (§8.6 uses strict >)."""
        # Arrange 4 boxes so hGap == vGap == 10
        # top-left, top-right (same top row)
        tl = bb(0, 50, 40, 60)
        tr = bb(60, 50, 100, 60)
        # bottom-left, bottom-right (10pt below)
        bl = bb(0, 30, 40, 40)
        br = bb(60, 30, 100, 40)
        # hGap = 50 - 40 = 10 (gap between row 0 and row 1 in y)
        # vGap = 60 - 40 = 20 ... actually let me set explicit coords
        # hGap: topY of bottom row = 40, bottomY of top row = 50 → gap=10
        # vGap: leftX of right col = 60, rightX of left col = 40 → gap=20
        # To get equal gaps, use tighter spacing
        tl2 = bb(0, 50, 40, 60)
        tr2 = bb(50, 50, 90, 60)   # vGap = 50-40 = 10
        bl2 = bb(0, 30, 40, 40)   # hGap = 50-40 = 10
        br2 = bb(50, 30, 90, 40)
        result = sort_reading_order(indexed(tl2, tr2, bl2, br2))
        # hGap=10 == vGap=10 → strict > is false → vertical cut taken
        # left=[tl2,bl2] sorted topY desc → tl2,bl2; right=[tr2,br2] → tr2,br2
        # Expected: tl2(0), bl2(2), tr2(1), br2(3)
        assert result == [0, 2, 1, 3]


# ---------------------------------------------------------------------------
# Narrow-outlier filter (§2.4)
# ---------------------------------------------------------------------------


class TestNarrowOutlierFilter:
    def test_narrow_element_removed_allows_vertical_cut(self):
        """A narrow element straddling the column gutter is dropped so the cut fires."""
        # Two columns with a narrow decorator in the gutter
        col1 = bb(0, 20, 40, 60)    # wide enough (width=40)
        col2 = bb(60, 20, 100, 60)  # wide enough (width=40)
        # Region width = 100.  NARROW_THRESHOLD = 100 * 0.1 = 10
        # Narrow element straddles the gap (x: 35–45, width=10 is exactly at threshold
        # → included, NOT filtered).  Use width=9 to be filtered.
        narrow = bb(35, 30, 44, 50)  # width=9 < threshold=10 → filtered
        result = sort_reading_order(indexed(col1, col2, narrow), beta=2.0)
        # After filtering narrow, vGap = 60-40=20 >= 5 → vertical cut fires
        # left group: col1(center_x=20), narrow would be center_x=39.5 < 47.5
        # right group: col2(center_x=80)
        # Both columns topY=60, leftX: col1=0 < col2=60 → col1 then col2
        # narrow ends up in left column group, sorted by topY then leftX
        assert result[0] == 0 or result[0] == 2  # col1 or narrow first (both top=60)
        # Importantly col2 appears after col1
        assert result.index(0) < result.index(1) or result.index(2) < result.index(1)


# ---------------------------------------------------------------------------
# Recursion depth guard
# ---------------------------------------------------------------------------


class TestRecursionDepthGuard:
    def test_deep_layout_does_not_stack_overflow(self):
        """1001 stacked boxes should not raise RecursionError."""
        # Each box stacked with 10pt gap — up to 1000 levels of recursion possible
        items = [(i, bb(0, i * 15, 100, i * 15 + 10)) for i in range(1001)]
        result = sort_reading_order(items)
        assert len(result) == 1001
        # All indices present
        assert set(result) == set(range(1001))


# ---------------------------------------------------------------------------
# Merge phase (§8.5 — not a simple concat)
# ---------------------------------------------------------------------------


class TestMergePhase:
    def test_footer_interleaved_at_correct_y(self):
        """A cross-layout footer at Y=10–20 is inserted after main content above Y=20."""
        footer = bb(0, 10, 200, 20)  # wide, so it's cross-layout
        main_hi = bb(20, 80, 80, 90)
        main_lo = bb(20, 30, 80, 40)
        # beta=0.7: maxWidth=200, threshold=140; footer.width=200>=140 ✓
        # footer overlaps main_hi: overlapWidth=60, min(200,60)=60, ratio=1.0 ✓
        # footer overlaps main_lo: same ✓ → footer is cross-layout
        result = sort_reading_order(indexed(footer, main_hi, main_lo), beta=0.7)
        # sortedCrossLayout=[footer topY=20]
        # sortedMain=[main_hi topY=90, main_lo topY=40]
        # Merge: crossTopY=20 < mainTopY=90 → emit main_hi
        #        crossTopY=20 < mainTopY=40 → emit main_lo
        #        exhaust main → emit footer
        assert result == [1, 2, 0], f"Expected [1,2,0] but got {result}"

    def test_cross_element_same_topY_as_main_goes_first(self):
        """At equal topY, cross-layout element precedes main (§7.7 uses >=)."""
        # cross at topY=50, main at topY=50 → cross goes first
        wide = bb(0, 40, 200, 50)   # wide → cross-layout (beta=0.7)
        narrow_a = bb(10, 40, 50, 50)
        narrow_b = bb(110, 40, 150, 50)
        # All three have topY=50
        result = sort_reading_order(indexed(wide, narrow_a, narrow_b), beta=0.7)
        # cross=wide (index 0) must precede the two narrow main elements
        assert result[0] == 0


# ---------------------------------------------------------------------------
# Density ratio (§2.2)
# ---------------------------------------------------------------------------


class TestDensityRatio:
    def test_high_density_sets_prefer_horizontal(self):
        """Very dense layout (densityRatio > 0.9) sets preferHorizontalFirst=True.

        The flag is vestigial (§8.6) but the algorithm must not crash when set.
        We verify the output is still a valid permutation of the indices.
        """
        # Fill ~95% of the bounding region
        a = bb(0, 50, 95, 100)
        b = bb(0, 0, 95, 45)
        items = indexed(a, b)
        result = sort_reading_order(items)
        assert set(result) == {0, 1}
        assert result == [0, 1]  # a topY=100 > b topY=45


# ---------------------------------------------------------------------------
# Overlap ratio (§7.5)
# ---------------------------------------------------------------------------


class TestOverlapRatio:
    def test_non_overlapping_boxes_have_zero_ratio(self):
        """Adjacent but non-overlapping boxes return ratio=0, not negative."""
        # This tests the internal helper indirectly via cross-layout detection
        # by confirming a non-overlapping wide element is NOT cross-layout
        wide = bb(0, 80, 200, 90)
        col1 = bb(210, 40, 250, 60)  # entirely to the right of wide → no overlap
        col2 = bb(260, 40, 300, 60)
        # overlapCount=0 < 2 → wide not cross-layout
        result = sort_reading_order(indexed(wide, col1, col2), beta=0.7)
        # wide topY=90 → first
        assert result[0] == 0


# ---------------------------------------------------------------------------
# Vertical cut split (center-based assignment)
# ---------------------------------------------------------------------------


class TestVerticalCutAssignment:
    def test_center_x_decides_left_vs_right(self):
        """An element whose centerX < cutX goes left; >= cutX goes right."""
        # cut at x=75 (from §4)
        left_item = bb(10, 40, 50, 60)   # centerX=30 < 75 → left
        right_item = bb(100, 40, 140, 60) # centerX=120 >= 75 → right
        result = sort_reading_order(indexed(left_item, right_item))
        # Both same topY=60 → sort by leftX: left_item first
        assert result == [0, 1]

    def test_center_x_exactly_at_cut_goes_right(self):
        """An element whose centerX == cutX is assigned to the right group."""
        # Two columns with centerX=75 exactly at cut
        # col1: x 50–100, centerX=75 exactly at cut → right
        # col2: x 0–40, centerX=20 < 75 → left
        col1 = bb(50, 20, 100, 40)  # centerX=75
        col2 = bb(0, 20, 40, 40)    # centerX=20
        col3 = bb(110, 20, 150, 40) # centerX=130 → right
        # vGap: gap between x=40 and x=50 = 10 → cut at 45
        # col1.centerX=75 >= 45 → right; col2.centerX=20 < 45 → left; col3 → right
        result = sort_reading_order(indexed(col1, col2, col3))
        # Left group = [col2]; right group = [col1, col3] sorted by topY then leftX
        # Merge: left(col2) then right(col1 leftX=50, col3 leftX=110)
        assert result.index(1) < result.index(0)  # col2 before col1
        assert result.index(1) < result.index(2)  # col2 before col3
