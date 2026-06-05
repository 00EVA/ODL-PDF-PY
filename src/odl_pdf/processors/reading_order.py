# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of XYCutPlusPlusSorter (651 LOC Java, package
# org.opendataloader.pdf.processors.readingorder).  No Java source was
# copied.  Behaviour reconstructed from the documented algorithm + constants in
# docs/architecture/04-reading-order.md and the paper arXiv:2504.10258.
"""XY-Cut++ reading-order sorter.

Public entry point::

    result_indices = sort_reading_order(items, beta=2.0, density_threshold=0.9)

where ``items`` is ``list[tuple[int, BoundingBox]]`` and the return value is
the input indices in visual reading order (top-to-bottom, left-to-right for
Y-up PDF coordinate space).
"""

from __future__ import annotations

from typing import Optional

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# §5 Constants — replicated exactly from XYCutPlusPlusSorter.java field list
# ---------------------------------------------------------------------------

DEFAULT_BETA: float = 2.0
"""Cross-layout width multiplier.  Value 2.0 effectively disables detection."""

DEFAULT_DENSITY_THRESHOLD: float = 0.9
"""Density ratio above which preferHorizontalFirst=True (currently vestigial)."""

OVERLAP_THRESHOLD: float = 0.1
"""Minimum horizontal overlap ratio (relative to smaller box) to count."""

MIN_OVERLAP_COUNT: int = 2
"""Cross-layout element must overlap with at least this many other elements."""

MIN_GAP_THRESHOLD: float = 5.0
"""Minimum gap in points to treat a cut as valid."""

NARROW_ELEMENT_WIDTH_RATIO: float = 0.1
"""Elements narrower than 10% of the region width are potential outliers."""

_MAX_RECURSION_DEPTH: int = 1000
"""Stack depth limit to prevent overflow on pathological inputs."""

# ---------------------------------------------------------------------------
# Internal type alias
# ---------------------------------------------------------------------------

# Each item is a (user_index, BoundingBox) pair.
_Item = tuple[int, BoundingBox]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sort_reading_order(
    items: list[_Item],
    beta: float = DEFAULT_BETA,
    density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
) -> list[int]:
    """Sort *items* into visual reading order and return their indices.

    Parameters
    ----------
    items:
        ``[(index, BoundingBox), ...]`` — the bounding boxes to sort together
        with caller-supplied opaque indices (typically 0-based).
    beta:
        Cross-layout width multiplier.  ``2.0`` (default) disables cross-layout
        detection.  Typical useful value is ``0.7`` for two-column papers.
    density_threshold:
        Density ratio above which *preferHorizontalFirst* is set.  The flag is
        vestigial in the current algorithm (gap magnitude decides axis), but the
        parameter is exposed so callers can pass the Java default exactly.

    Returns
    -------
    list[int]
        The *index* component of *items* in visual reading order.
    """
    logger.info(
        "sort_reading_order entry: n=%d beta=%.2f density_threshold=%.2f",
        len(items),
        beta,
        density_threshold,
    )
    if not items:
        logger.debug("sort_reading_order: empty input, returning []")
        return []

    # Filter null-bbox objects (mirroring Java lines 99–105 which skip null bbox)
    valid: list[_Item] = [(idx, box) for (idx, box) in items if box is not None]
    dropped = len(items) - len(valid)
    if dropped:
        logger.warning(
            "sort_reading_order: dropped %d item(s) with None bounding_box", dropped
        )
    if not valid:
        logger.warning("sort_reading_order: no valid items after bbox filter")
        return []

    # Phase 1 — identify cross-layout elements
    cross_layout, remaining = _identify_cross_layout_elements(valid, beta)
    logger.debug(
        "Phase 1: cross_layout=%d remaining=%d", len(cross_layout), len(remaining)
    )

    if not remaining:
        logger.debug("Phase 1: remaining empty → sortByYThenX on all items")
        return _sort_by_y_then_x(valid)

    # Phase 2 — density ratio (vestigial tiebreaker, computed for parity)
    prefer_horizontal = _compute_density_ratio(remaining) > density_threshold
    logger.debug("Phase 2: prefer_horizontal_first=%s", prefer_horizontal)

    # Phase 3 — recursive segmentation (returns _Item list in order, not just indices)
    sorted_main_items = _recursive_segment_items(remaining, prefer_horizontal, depth=0)
    logger.debug("Phase 3: sorted_main count=%d", len(sorted_main_items))

    # Phase 4 — merge cross-layout elements back by topY descending
    result_items = _merge_cross_layout_elements(sorted_main_items, cross_layout)
    logger.info("sort_reading_order done: output %d items", len(result_items))
    return [idx for idx, _ in result_items]


# ---------------------------------------------------------------------------
# Phase 1 — cross-layout detection
# ---------------------------------------------------------------------------


def _identify_cross_layout_elements(
    items: list[_Item], beta: float
) -> tuple[list[_Item], list[_Item]]:
    """Split *items* into (cross_layout, remaining).

    A cross-layout element satisfies:
    1. width >= beta * maxWidth
    2. horizontally overlaps >= MIN_OVERLAP_COUNT other elements with
       overlap ratio >= OVERLAP_THRESHOLD

    Guard: if len(items) < 3, always return ([], items).
    """
    if len(items) < 3:
        logger.debug("_identify_cross_layout_elements: guard size < 3, skip")
        return [], list(items)

    max_width = max(box.width for _, box in items)
    threshold = beta * max_width
    logger.debug(
        "_identify_cross_layout_elements: n=%d maxWidth=%.2f threshold=%.2f",
        len(items),
        max_width,
        threshold,
    )

    cross: list[_Item] = []
    remaining: list[_Item] = []

    for item in items:
        idx, box = item
        if box.width >= threshold:
            # Count qualifying overlaps with all other elements
            overlap_count = 0
            for other_idx, other_box in items:
                if other_idx == idx:
                    continue
                ratio = _calculate_horizontal_overlap_ratio(box, other_box)
                if ratio >= OVERLAP_THRESHOLD:
                    overlap_count += 1
            logger.debug(
                "_identify_cross_layout_elements: item=%d width=%.2f overlap_count=%d",
                idx,
                box.width,
                overlap_count,
            )
            if overlap_count >= MIN_OVERLAP_COUNT:
                cross.append(item)
                continue
        remaining.append(item)

    logger.debug(
        "_identify_cross_layout_elements: cross=%d remaining=%d",
        len(cross),
        len(remaining),
    )
    return cross, remaining


def _calculate_horizontal_overlap_ratio(box1: BoundingBox, box2: BoundingBox) -> float:
    """Ratio of horizontal overlap width to the width of the smaller box.

    Mirrors Java calculateHorizontalOverlapRatio (lines 233–247).
    Uses hard zero for non-positive overlap (no epsilon — §7.5).
    """
    overlap_left = max(box1.left, box2.left)
    overlap_right = min(box1.right, box2.right)
    overlap_width = max(0.0, overlap_right - overlap_left)
    smaller_width = min(box1.width, box2.width)
    if smaller_width <= 0.0:
        return 0.0
    return overlap_width / smaller_width


# ---------------------------------------------------------------------------
# Phase 2 — density ratio
# ---------------------------------------------------------------------------


def _compute_density_ratio(items: list[_Item]) -> float:
    """Compute content density relative to the bounding region.

    ``densityRatio = min(1.0, sum(area) / bounding_region_area)``
    Returns 0.0 if the bounding region has zero area.
    """
    if not items:
        return 0.0

    # Bounding region of all items
    region = _bounding_region(items)
    region_area = region.width * region.height
    if region_area <= 0.0:
        logger.debug("_compute_density_ratio: zero region area, returning 0.0")
        return 0.0

    total_area = sum(box.width * box.height for _, box in items)
    ratio = min(1.0, total_area / region_area)
    logger.debug(
        "_compute_density_ratio: total_area=%.2f region_area=%.2f ratio=%.4f",
        total_area,
        region_area,
        ratio,
    )
    return ratio


def _bounding_region(items: list[_Item]) -> BoundingBox:
    """Union bounding box of all items."""
    boxes = [box for _, box in items]
    result = BoundingBox(
        page_number=boxes[0].page_number,
        last_page_number=boxes[0].last_page_number,
        left=boxes[0].left,
        bottom=boxes[0].bottom,
        right=boxes[0].right,
        top=boxes[0].top,
    )
    for box in boxes[1:]:
        result.union(box)
    return result


# ---------------------------------------------------------------------------
# Phase 3 — recursive segmentation (iterative with explicit stack)
# Returns _Item list (index, box) in sorted order so Phase 4 can compare topY.
# ---------------------------------------------------------------------------


def _recursive_segment_items(
    items: list[_Item],
    prefer_horizontal: bool,
    depth: int,
) -> list[_Item]:
    """Segment *items* recursively and return them in reading order.

    Uses an explicit stack to avoid Python's recursion limit on large inputs.
    The implicit XY-Cut tree is traversed depth-first, top/left group first.
    """
    if not items:
        return []
    if len(items) == 1:
        return list(items)

    output: list[_Item] = []

    # Stack holds (group, prefer_h, depth)
    stack: list[tuple[list[_Item], bool, int]] = [(items, prefer_horizontal, 0)]

    while stack:
        group, ph, d = stack.pop()

        if not group:
            continue
        if len(group) == 1:
            output.append(group[0])
            continue

        if d >= _MAX_RECURSION_DEPTH:
            logger.warning(
                "_recursive_segment: depth %d >= limit %d, fallback sortByYThenX for %d items",
                d,
                _MAX_RECURSION_DEPTH,
                len(group),
            )
            output.extend(_sort_items_by_y_then_x(group))
            continue

        h_cut = _find_best_horizontal_cut(group)
        v_cut = _find_best_vertical_cut(group)

        has_h = h_cut is not None
        has_v = v_cut is not None

        logger.debug(
            "_recursive_segment depth=%d n=%d h_cut=%s v_cut=%s",
            d,
            len(group),
            f"gap={h_cut[0]:.2f} pos={h_cut[1]:.2f}" if h_cut else "None",
            f"gap={v_cut[0]:.2f} pos={v_cut[1]:.2f}" if v_cut else "None",
        )

        if has_h and has_v:
            h_gap, _ = h_cut  # type: ignore[misc]
            v_gap, _ = v_cut  # type: ignore[misc]
            # Strict >: equal gaps → vertical cut (§8.6, replicates Java's >)
            use_h = h_gap > v_gap
            logger.debug(
                "_recursive_segment depth=%d: both valid h_gap=%.2f v_gap=%.2f → use_h=%s",
                d,
                h_gap,
                v_gap,
                use_h,
            )
        elif has_h:
            use_h = True
        elif has_v:
            use_h = False
        else:
            logger.debug(
                "_recursive_segment depth=%d: no valid cut, sortByYThenX on %d items",
                d,
                len(group),
            )
            output.extend(_sort_items_by_y_then_x(group))
            continue

        if use_h:
            _, h_pos = h_cut  # type: ignore[misc]
            above, below = _split_by_horizontal_cut(group, h_pos)
            logger.debug(
                "_recursive_segment depth=%d: horizontal cut y=%.2f → above=%d below=%d",
                d,
                h_pos,
                len(above),
                len(below),
            )
            if not above or not below:
                logger.warning(
                    "_recursive_segment depth=%d: horizontal cut produced empty group, fallback",
                    d,
                )
                output.extend(_sort_items_by_y_then_x(group))
                continue
            # Push below first so above is processed first (LIFO)
            stack.append((below, ph, d + 1))
            stack.append((above, ph, d + 1))
        else:
            _, v_pos = v_cut  # type: ignore[misc]
            left_g, right_g = _split_by_vertical_cut(group, v_pos)
            logger.debug(
                "_recursive_segment depth=%d: vertical cut x=%.2f → left=%d right=%d",
                d,
                v_pos,
                len(left_g),
                len(right_g),
            )
            if not left_g or not right_g:
                logger.warning(
                    "_recursive_segment depth=%d: vertical cut produced empty group, fallback",
                    d,
                )
                output.extend(_sort_items_by_y_then_x(group))
                continue
            # Push right first so left is processed first (LIFO)
            stack.append((right_g, ph, d + 1))
            stack.append((left_g, ph, d + 1))

    return output


# ---------------------------------------------------------------------------
# Cut finders — return (gap, cut_position) or None
# ---------------------------------------------------------------------------


def _find_best_horizontal_cut(
    items: list[_Item],
) -> Optional[tuple[float, float]]:
    """Find the largest horizontal gap (whitespace strip across full width).

    Scan elements sorted by topY DESC (then bottomY DESC for ties, §7.3).
    Tracks ``prev_bottom`` as the minimum bottomY seen so far — the lowest
    floor of all previously processed elements.  A gap opens when the next
    element's topY is below that floor.

    Returns ``(gap, cut_position)`` if gap >= MIN_GAP_THRESHOLD, else None.
    """
    if len(items) < 2:
        return None

    # Stable sort: primary topY DESC, secondary bottomY DESC (§7.3 / §8.3)
    sorted_items = sorted(items, key=lambda x: (-x[1].top, -x[1].bottom))

    largest_gap = 0.0
    cut_position = 0.0
    prev_bottom: Optional[float] = None

    for _, box in sorted_items:
        top = box.top
        bottom = box.bottom
        if prev_bottom is not None:
            if top < prev_bottom:
                gap = prev_bottom - top
                if gap > largest_gap:
                    largest_gap = gap
                    cut_position = (prev_bottom + top) / 2.0
            # prev_bottom tracks the minimum bottom Y seen (furthest down the page)
            prev_bottom = min(prev_bottom, bottom)
        else:
            prev_bottom = bottom

    if largest_gap >= MIN_GAP_THRESHOLD:
        return largest_gap, cut_position
    return None


def _find_best_vertical_cut(
    items: list[_Item],
) -> Optional[tuple[float, float]]:
    """Find the largest vertical gap (whitespace strip across full height).

    Tries full item set first.  If below threshold, retries after filtering
    narrow-outlier elements (§2.4).

    Returns ``(gap, cut_position)`` or None.
    """
    result = _find_vertical_cut_with_projection(items)
    if result is not None:
        return result

    # §2.4 narrow-outlier retry
    if len(items) < 3:
        return None

    region = _bounding_region(items)
    region_width = region.width
    narrow_threshold = region_width * NARROW_ELEMENT_WIDTH_RATIO

    filtered = [(idx, box) for idx, box in items if box.width >= narrow_threshold]
    if len(filtered) >= len(items):
        logger.debug("_find_best_vertical_cut: no narrow elements to filter")
        return None
    if len(filtered) < 2:
        logger.debug(
            "_find_best_vertical_cut: too few items (%d) after narrow filter",
            len(filtered),
        )
        return None

    logger.debug(
        "_find_best_vertical_cut: narrow-outlier retry: filtered %d → %d items",
        len(items),
        len(filtered),
    )
    retry = _find_vertical_cut_with_projection(filtered)
    if retry is None:
        return None

    # The unfiltered result was None (gap < threshold), so any valid retry
    # is strictly better by definition.  Return it.
    return retry


def _find_vertical_cut_with_projection(
    items: list[_Item],
) -> Optional[tuple[float, float]]:
    """Core vertical gap scanner.

    Stable sort by (leftX ASC, rightX ASC) (§7.2 / §8.3).
    Tracks ``prev_right`` as a running maximum (§8.4) — the furthest rightX
    seen so far across all elements processed left-to-right.

    Returns ``(gap, cut_position)`` if gap >= MIN_GAP_THRESHOLD else None.
    """
    if len(items) < 2:
        return None

    # Stable sort: primary leftX ASC, secondary rightX ASC (§8.3)
    sorted_items = sorted(items, key=lambda x: (x[1].left, x[1].right))

    largest_gap = 0.0
    cut_position = 0.0
    prev_right: Optional[float] = None

    for _, box in sorted_items:
        left = box.left
        right = box.right
        if prev_right is not None:
            if left > prev_right:
                gap = left - prev_right
                if gap > largest_gap:
                    largest_gap = gap
                    cut_position = (prev_right + left) / 2.0
            prev_right = max(prev_right, right)
        else:
            prev_right = right

    if largest_gap >= MIN_GAP_THRESHOLD:
        return largest_gap, cut_position
    return None


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------


def _split_by_horizontal_cut(
    items: list[_Item], cut_y: float
) -> tuple[list[_Item], list[_Item]]:
    """Split into above (centerY > cutY) and below (centerY <= cutY)."""
    above: list[_Item] = []
    below: list[_Item] = []
    for item in items:
        _, box = item
        center_y = (box.top + box.bottom) / 2.0
        if center_y > cut_y:
            above.append(item)
        else:
            below.append(item)
    return above, below


def _split_by_vertical_cut(
    items: list[_Item], cut_x: float
) -> tuple[list[_Item], list[_Item]]:
    """Split into left (centerX < cutX) and right (centerX >= cutX)."""
    left_g: list[_Item] = []
    right_g: list[_Item] = []
    for item in items:
        _, box = item
        center_x = (box.left + box.right) / 2.0
        if center_x < cut_x:
            left_g.append(item)
        else:
            right_g.append(item)
    return left_g, right_g


# ---------------------------------------------------------------------------
# sortByYThenX fallback (§7.4 / §8.3)
# ---------------------------------------------------------------------------


def _sort_by_y_then_x(items: list[_Item]) -> list[int]:
    """Stable sort by (-topY, leftX) — largest Y first, then leftmost first.

    Returns the sorted list of user indices (not _Item pairs).
    """
    sorted_items = sorted(items, key=lambda x: (-x[1].top, x[1].left))
    return [idx for idx, _ in sorted_items]


def _sort_items_by_y_then_x(items: list[_Item]) -> list[_Item]:
    """Same as _sort_by_y_then_x but returns _Item pairs (needed inside Phase 3)."""
    return sorted(items, key=lambda x: (-x[1].top, x[1].left))


# ---------------------------------------------------------------------------
# Phase 4 — merge cross-layout elements (§8.5)
# ---------------------------------------------------------------------------


def _merge_cross_layout_elements(
    sorted_main: list[_Item],
    cross_layout: list[_Item],
) -> list[_Item]:
    """Merge-sort *cross_layout* into *sorted_main* by topY descending.

    Both sequences are already sorted by topY descending (highest first).
    Uses ``>=`` at the comparison: a cross-layout element at the same topY
    as a main element goes first (§7.7).

    This is a full merge, not a simple append — a wide footer at Y=15
    will be interleaved after all main content above Y=15.
    """
    if not cross_layout:
        return sorted_main
    if not sorted_main:
        return _sort_items_by_y_then_x(cross_layout)

    # Pre-sort the cross-layout sequence (sortByYThenX)
    sorted_cross = _sort_items_by_y_then_x(cross_layout)

    logger.debug(
        "_merge_cross_layout_elements: merging sorted_main=%d cross=%d",
        len(sorted_main),
        len(sorted_cross),
    )

    result: list[_Item] = []
    ci = 0  # pointer into sorted_cross
    mi = 0  # pointer into sorted_main

    while ci < len(sorted_cross) and mi < len(sorted_main):
        cross_idx, cross_box = sorted_cross[ci]
        _main_idx, main_box = sorted_main[mi]
        cross_top_y = cross_box.top
        main_top_y = main_box.top

        # >= means: equal topY → cross-layout element goes first (§7.7)
        if cross_top_y >= main_top_y:
            result.append(sorted_cross[ci])
            ci += 1
        else:
            result.append(sorted_main[mi])
            mi += 1

    # Exhaust remaining
    while ci < len(sorted_cross):
        result.append(sorted_cross[ci])
        ci += 1
    while mi < len(sorted_main):
        result.append(sorted_main[mi])
        mi += 1

    logger.debug("_merge_cross_layout_elements: result count=%d", len(result))
    return result
