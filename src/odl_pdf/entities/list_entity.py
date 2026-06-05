# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Clean-room port of veraPDF's list model: ``PDFList`` and ``ListItem``.
# Accessors are reconstructed from the OpenDataLoader processor/writer call
# sites (``ListProcessor``, ``PDFWriter``, ``MarkdownGenerator``) and the
# ``ListSerializer`` JSON field names. The numbering style is veraPDF's
# ``NumberingStyleNames`` string, kept opaque. No veraPDF source copied.
#
# Module is ``list_entity`` (not ``list``) to avoid shadowing the builtin.
"""List entities."""

from __future__ import annotations

from dataclasses import dataclass, field

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.object import IObject, union_objects


@dataclass
class ListItem:
    """One item (bullet) of a list."""

    contents: list[IObject] = field(default_factory=list)

    def add_content(self, content: IObject) -> None:
        self.contents.append(content)

    @property
    def bounding_box(self) -> BoundingBox | None:
        return union_objects(self.contents)


@dataclass
class PDFList:
    """A detected list.

    ``numbering_style`` is the veraPDF ``NumberingStyleNames`` token (e.g.
    ``"ENGLISH_LETTERS"``) or ``None`` for an unordered list. ``label_length``
    is the character width of the bullet/number labels.
    """

    items: list[ListItem] = field(default_factory=list)
    numbering_style: str | None = None
    label_length: int = 0
    recognized_structure_id: int | None = None
    previous_list_id: int | None = None
    next_list_id: int | None = None

    def push(self, item: ListItem) -> None:
        self.items.append(item)

    @property
    def number_of_items(self) -> int:
        return len(self.items)

    @property
    def bounding_box(self) -> BoundingBox | None:
        acc: BoundingBox | None = None
        for item in self.items:
            box = item.bounding_box
            if box is None:
                continue
            if acc is None:
                acc = BoundingBox(**vars(box))
            else:
                acc.union(box)
        return acc
