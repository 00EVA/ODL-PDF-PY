# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of veraPDF's ``IObject`` root: the polymorphic content element
# that OpenDataLoader's pipeline is typed as (``List<List<IObject>>``) and that
# every writer/processor type-switches on (see
# ``docs/architecture/02-pdf-parsing-layer.md`` §4). No veraPDF source copied.
"""The polymorphic content element (``IObject``).

In the Java code every content item implements the ``IObject`` interface and
the processors/writers dispatch with ``instanceof``. In Python every entity
exposes a ``bounding_box`` property, so ``IObject`` is the structural set of
content types and :func:`bounding_box_of` / :func:`union_objects` operate on any
of them by duck typing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from odl_pdf.entities.bounding_box import BoundingBox


@runtime_checkable
class IObject(Protocol):
    """Any content element: exposes a (possibly ``None``) bounding box."""

    @property
    def bounding_box(self) -> BoundingBox | None: ...


def bounding_box_of(obj: IObject) -> BoundingBox | None:
    """The bounding box of a content element, or ``None`` if it has none."""
    return obj.bounding_box


def union_objects(objects: list[IObject]) -> BoundingBox | None:
    """Union the bounding boxes of a list of content elements."""
    acc: BoundingBox | None = None
    for obj in objects:
        box = obj.bounding_box
        if box is None:
            continue
        if acc is None:
            acc = BoundingBox(**vars(box))
        else:
            acc.union(box)
    return acc
