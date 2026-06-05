# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Tests for the JSON->Markdown adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from odl_pdf.output.markdown_writer import _tree_to_markdown, document_to_markdown
from odl_pdf.pipeline import extract

ROOT = Path(__file__).resolve().parents[2]
LOREM = ROOT / "opendataloader-pdf/samples/pdf/lorem.pdf"


def test_heading_and_paragraph_from_tree():
    tree = {
        "kids": [
            {"type": "heading", "heading level": 1, "content": "Title", "page number": 1},
            {"type": "heading", "heading level": 2, "content": "Sub", "page number": 1},
            {"type": "paragraph", "content": "Body text here.", "page number": 1},
        ]
    }
    md = _tree_to_markdown(tree)
    assert "# Title" in md
    assert "## Sub" in md
    assert "Body text here." in md


def test_ordered_and_unordered_lists():
    ordered = {"kids": [{"type": "list", "numbering style": "decimal", "page number": 1,
                         "list items": [{"kids": [{"content": "one"}]},
                                        {"kids": [{"content": "two"}]}]}]}
    md = _tree_to_markdown(ordered)
    assert "1. one" in md and "2. two" in md

    bullet = {"kids": [{"type": "list", "numbering style": "bullet", "page number": 1,
                        "list items": [{"kids": [{"content": "a"}]}]}]}
    assert "- a" in _tree_to_markdown(bullet)


def test_table_renders_pipe_table():
    tree = {"kids": [{
        "type": "table", "number of columns": 2, "page number": 1,
        "rows": [
            {"type": "table row", "row number": 1, "cells": [
                {"column number": 1, "kids": [{"content": "H1"}]},
                {"column number": 2, "kids": [{"content": "H2"}]}]},
            {"type": "table row", "row number": 2, "cells": [
                {"column number": 1, "kids": [{"content": "a"}]},
                {"column number": 2, "kids": [{"content": "b"}]}]},
        ],
    }]}
    md = _tree_to_markdown(tree)
    lines = md.strip().splitlines()
    assert lines[0] == "| H1 | H2 |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| a | b |"


def test_figure_with_alt():
    tree = {"kids": [{"type": "figure", "alt": "a chart", "source": "img/1.png", "page number": 1}]}
    assert "![a chart](img/1.png)" in _tree_to_markdown(tree)


def test_pipe_escaped_in_cells():
    tree = {"kids": [{"type": "table", "number of columns": 1, "page number": 1,
                      "rows": [{"cells": [{"column number": 1, "kids": [{"content": "a|b"}]}]}]}]}
    assert "a\\|b" in _tree_to_markdown(tree)


@pytest.mark.skipif(not LOREM.exists(), reason="lorem sample not present")
def test_lorem_end_to_end_markdown():
    md = document_to_markdown(extract(LOREM))
    assert md.startswith("# Lorem Ipsum")
    assert "consectetur adipiscing elit" in md
    # No intra-line double-spacing leaked in (block separators are \n\n, fine).
    for line in md.splitlines():
        assert "  " not in line, f"double space within a line: {line!r}"
