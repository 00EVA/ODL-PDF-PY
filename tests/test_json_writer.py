# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""Tests for the JSON writer — TDD-first, checked against the Java JAR oracle.

Behavioral parity contract: these test case names and assertions match the Rust
test suite in ``crates/json-writer/tests/test_json_writer.rs`` so the two ports
provably agree.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP

import pytest

from odl_pdf.entities.bounding_box import BoundingBox
from odl_pdf.entities.chunk import TextChunk, ImageChunk, LineArtChunk
from odl_pdf.entities.document import Document, DocumentMetadata, Page
from odl_pdf.entities.list_entity import ListItem, PDFList
from odl_pdf.entities.semantic import SemanticTextNode, SemanticType
from odl_pdf.entities.table import TableBorder, TableBorderCell, TableBorderRow
from odl_pdf.entities.text import TextBlock, TextColumn, TextLine
from odl_pdf.output.json_writer import write_document_json, round_double


# ---------------------------------------------------------------------------
# Helper: quick bounding box
# ---------------------------------------------------------------------------

def bbox(page: int, l: float, b: float, r: float, t: float) -> BoundingBox:
    return BoundingBox.of(page, l, b, r, t)


def _text_chunk(
    page: int,
    l: float, b: float, r: float, t: float,
    value: str = "",
    font_name: str = "",
    font_size: float = 0.0,
    text_color: list[float] | None = None,
) -> TextChunk:
    box = bbox(page, l, b, r, t)
    return TextChunk(
        bounding_box=box,
        value=value,
        font_name=font_name,
        font_size=font_size,
        text_color=text_color or [0.0],
    )


def _make_semantic_node(
    stype: SemanticType,
    page: int,
    l: float, b: float, r: float, t: float,
    value: str = "Hello",
    font_name: str = "Arial",
    font_size: float = 12.0,
    text_color: list[float] | None = None,
    heading_level: int | None = None,
) -> SemanticTextNode:
    """Build a SemanticTextNode backed by a single TextChunk."""
    chunk = _text_chunk(page, l, b, r, t, value, font_name, font_size, text_color or [0.0])
    line = TextLine(chunks=[chunk])
    block = TextBlock(lines=[line])
    col = TextColumn(blocks=[block])
    node = SemanticTextNode(semantic_type=stype, columns=[col], heading_level=heading_level)
    return node


# ---------------------------------------------------------------------------
# 1. round_double — must replicate Java's BigDecimal(Double.toString(v)).setScale(3, HALF_UP)
# ---------------------------------------------------------------------------

class TestRoundDouble:
    """Half-up rounding via the shortest decimal repr, not the binary float."""

    def test_basic_three_places(self) -> None:
        # Java: BigDecimal("1.2345").setScale(3, HALF_UP) = 1.235
        # The digit at 4th decimal place is 5 — HALF_UP rounds the 3rd decimal up: .234→.235
        assert round_double(1.2345) == pytest.approx(1.235, abs=1e-9)

    def test_half_up_on_string_repr(self) -> None:
        # Java: BigDecimal("1.2345").setScale(3, HALF_UP) = 1.235 (5 rounds up)
        assert round_double(1.2345) == pytest.approx(1.235, abs=1e-9)

    def test_oracle_745_1316(self) -> None:
        # The task spec: 745.1316 → 745.132 (digit dropped is 6 > 5 → round up)
        assert round_double(745.1316) == pytest.approx(745.132, abs=1e-9)

    def test_oracle_0_0005_half_up(self) -> None:
        # The task spec: 0.0005 → 0.001 (half-up)
        assert round_double(0.0005) == pytest.approx(0.001, abs=1e-9)

    def test_round_no_change(self) -> None:
        assert round_double(32.005) == pytest.approx(32.005, abs=1e-9)

    def test_round_exact(self) -> None:
        assert round_double(200.891) == pytest.approx(200.891, abs=1e-9)

    def test_round_zero(self) -> None:
        assert round_double(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_round_negative(self) -> None:
        # Negative: BigDecimal("-1.2345").setScale(3,HALF_UP) = -1.235
        assert round_double(-1.2345) == pytest.approx(-1.235, abs=1e-9)

    def test_round_large(self) -> None:
        assert round_double(9999.9999) == pytest.approx(10000.0, abs=1e-9)

    def test_round_trailing_zeros_not_padded(self) -> None:
        # 200.0 stays 200.0 (not 200.000 as an int — the JSON number is 200.0)
        assert round_double(200.0) == pytest.approx(200.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Document-level envelope
# ---------------------------------------------------------------------------

class TestDocumentEnvelope:
    """Top-level JSON structure matches JsonWriter.writeDocumentInfo."""

    def _minimal_doc(self) -> Document:
        meta = DocumentMetadata(
            file_name="test.pdf",
            author="Alice",
            title="Test Doc",
            creation_date="D:20260101000000Z",
            modification_date="D:20260101000000Z",
        )
        doc = Document(metadata=meta)
        return doc

    def test_top_level_keys_order(self) -> None:
        doc = self._minimal_doc()
        result = json.loads(write_document_json(doc))
        keys = list(result.keys())
        assert keys[:7] == [
            "file name",
            "number of pages",
            "author",
            "title",
            "creation date",
            "modification date",
            "kids",
        ]

    def test_file_name(self) -> None:
        doc = self._minimal_doc()
        result = json.loads(write_document_json(doc))
        assert result["file name"] == "test.pdf"

    def test_number_of_pages(self) -> None:
        doc = self._minimal_doc()
        doc.push_page(Page(page_number=0, width=612, height=792))
        doc.push_page(Page(page_number=1, width=612, height=792))
        result = json.loads(write_document_json(doc))
        assert result["number of pages"] == 2

    def test_null_metadata_fields(self) -> None:
        doc = Document()  # all metadata None
        result = json.loads(write_document_json(doc))
        assert result["file name"] is None
        assert result["author"] is None
        assert result["title"] is None
        assert result["creation date"] is None
        assert result["modification date"] is None

    def test_kids_is_list(self) -> None:
        doc = self._minimal_doc()
        result = json.loads(write_document_json(doc))
        assert isinstance(result["kids"], list)

    def test_empty_document_zero_pages(self) -> None:
        doc = self._minimal_doc()
        result = json.loads(write_document_json(doc))
        assert result["number of pages"] == 0
        assert result["kids"] == []


# ---------------------------------------------------------------------------
# 3. Heading serialization
# ---------------------------------------------------------------------------

class TestHeadingSerialization:
    """HeadingSerializer field order: base + heading level + text fields."""

    def _heading_doc(
        self,
        level: int = 1,
        bbox_coords: tuple = (200.891, 706.938, 394.152, 745.1316),
        font_name: str = "Pretendard-Regular",
        font_size: float = 32.005,
        content: str = "Lorem Ipsum",
        text_color: list[float] | None = None,
        structure_id: int = 1,
    ) -> tuple[Document, dict]:
        """Return (doc, first_kid) after serializing."""
        node = _make_semantic_node(
            SemanticType.HEADING, 0,
            *bbox_coords,
            value=content,
            font_name=font_name,
            font_size=font_size,
            text_color=text_color or [0.0],
            heading_level=level,
        )
        node.recognized_structure_id = structure_id

        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document(metadata=DocumentMetadata(file_name="lorem.pdf"))
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        return doc, result["kids"][0]

    def test_heading_type(self) -> None:
        _, kid = self._heading_doc()
        assert kid["type"] == "heading"

    def test_heading_pdfua_tag_h1(self) -> None:
        _, kid = self._heading_doc(level=1)
        assert kid["pdfua_tag"] == "H1"

    def test_heading_pdfua_tag_h2(self) -> None:
        _, kid = self._heading_doc(level=2, structure_id=2)
        assert kid["pdfua_tag"] == "H2"

    def test_heading_id_present(self) -> None:
        _, kid = self._heading_doc(structure_id=1)
        assert kid["id"] == 1

    def test_heading_id_omitted_when_zero(self) -> None:
        _, kid = self._heading_doc(structure_id=0)
        assert "id" not in kid

    def test_heading_page_number_one_based(self) -> None:
        _, kid = self._heading_doc()
        assert kid["page number"] == 1

    def test_heading_bounding_box_rounded(self) -> None:
        # 745.1316 → 745.132 (half-up on 6 at 4th decimal)
        _, kid = self._heading_doc(bbox_coords=(200.891, 706.938, 394.152, 745.1316))
        assert kid["bounding box"] == [200.891, 706.938, 394.152, 745.132]

    def test_heading_level_field_order(self) -> None:
        """heading level must come before font/content fields."""
        _, kid = self._heading_doc()
        keys = list(kid.keys())
        assert keys.index("heading level") < keys.index("font")
        assert keys.index("heading level") < keys.index("content")

    def test_heading_level_value(self) -> None:
        _, kid = self._heading_doc(level=3, structure_id=3)
        assert kid["heading level"] == 3

    def test_heading_font(self) -> None:
        _, kid = self._heading_doc(font_name="Pretendard-Regular")
        assert kid["font"] == "Pretendard-Regular"

    def test_heading_font_size_rounded(self) -> None:
        _, kid = self._heading_doc(font_size=32.005)
        assert kid["font size"] == pytest.approx(32.005, abs=1e-6)

    def test_heading_text_color(self) -> None:
        _, kid = self._heading_doc(text_color=[0.0])
        assert kid["text color"] == "[0.0]"

    def test_heading_text_color_rgb(self) -> None:
        _, kid = self._heading_doc(text_color=[0.1, 0.2, 0.3], structure_id=5)
        assert kid["text color"] == "[0.1, 0.2, 0.3]"

    def test_heading_text_color_omitted_when_none(self) -> None:
        """text color must be absent when the node has no color info."""
        node = _make_semantic_node(
            SemanticType.HEADING, 0, 0, 0, 100, 20,
            value="Hi", font_name="X", font_size=12.0,
            text_color=None,
            heading_level=1,
        )
        # Null text_color — force it
        node.columns[0].blocks[0].lines[0].chunks[0].text_color = []
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        assert "text color" not in result["kids"][0]

    def test_heading_content(self) -> None:
        _, kid = self._heading_doc(content="Lorem Ipsum")
        assert kid["content"] == "Lorem Ipsum"

    def test_oracle_lorem_heading(self) -> None:
        """Exact match against the lorem.json oracle for the heading element."""
        _, kid = self._heading_doc(
            level=1,
            bbox_coords=(200.891, 706.938, 394.152, 745.1316),
            font_name="Pretendard-Regular",
            font_size=32.005,
            content="Lorem Ipsum",
            text_color=[0.0],
            structure_id=1,
        )
        node = _make_semantic_node(
            SemanticType.HEADING, 0,
            200.891, 706.938, 394.152, 745.1316,
            value="Lorem Ipsum",
            font_name="Pretendard-Regular",
            font_size=32.005,
            text_color=[0.0],
            heading_level=1,
        )
        node.recognized_structure_id = 1
        node.level = "Doctitle"
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document(metadata=DocumentMetadata(
            file_name="lorem.pdf",
            author="leebd-public",
            title=None,
            creation_date="D:20251010112501+09'00'",
            modification_date="D:20251010112501+09'00'",
        ))
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        kid = result["kids"][0]
        assert kid["type"] == "heading"
        assert kid["pdfua_tag"] == "H1"
        assert kid["id"] == 1
        assert kid["level"] == "Doctitle"
        assert kid["page number"] == 1
        assert kid["bounding box"] == [200.891, 706.938, 394.152, 745.132]
        assert kid["heading level"] == 1
        assert kid["font"] == "Pretendard-Regular"
        assert kid["font size"] == pytest.approx(32.005, abs=1e-6)
        assert kid["text color"] == "[0.0]"
        assert kid["content"] == "Lorem Ipsum"


# ---------------------------------------------------------------------------
# 4. Paragraph serialization
# ---------------------------------------------------------------------------

class TestParagraphSerialization:
    """SemanticTextNodeSerializer for paragraph nodes."""

    def _para_doc(self, **kwargs) -> dict:
        node = _make_semantic_node(
            SemanticType.PARAGRAPH, 0,
            85.034, 567.936, 502.306, 659.761,
            value=kwargs.get("content", "Hello world"),
            font_name=kwargs.get("font_name", "Pretendard-Regular"),
            font_size=kwargs.get("font_size", 9.949),
            text_color=kwargs.get("text_color", [0.0]),
        )
        node.recognized_structure_id = kwargs.get("structure_id", 2)
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        return json.loads(write_document_json(doc))["kids"][0]

    def test_paragraph_type(self) -> None:
        assert self._para_doc()["type"] == "paragraph"

    def test_paragraph_pdfua_tag(self) -> None:
        assert self._para_doc()["pdfua_tag"] == "P"

    def test_paragraph_id(self) -> None:
        assert self._para_doc(structure_id=2)["id"] == 2

    def test_paragraph_page_number(self) -> None:
        assert self._para_doc()["page number"] == 1

    def test_paragraph_bounding_box(self) -> None:
        kid = self._para_doc()
        assert kid["bounding box"] == [85.034, 567.936, 502.306, 659.761]

    def test_oracle_lorem_paragraph(self) -> None:
        """Match the lorem.json oracle paragraph."""
        content = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
            "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
            "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
            "consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse "
            "cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
            "non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        )
        kid = self._para_doc(
            content=content,
            font_name="Pretendard-Regular",
            font_size=9.949,
            text_color=[0.0],
            structure_id=2,
        )
        assert kid["type"] == "paragraph"
        assert kid["pdfua_tag"] == "P"
        assert kid["id"] == 2
        assert kid["page number"] == 1
        assert kid["bounding box"] == [85.034, 567.936, 502.306, 659.761]
        assert kid["font"] == "Pretendard-Regular"
        assert kid["font size"] == pytest.approx(9.949, abs=1e-6)
        assert kid["text color"] == "[0.0]"
        assert kid["content"] == content

    def test_paragraph_no_heading_level_field(self) -> None:
        """Paragraphs must NOT emit heading level."""
        kid = self._para_doc()
        assert "heading level" not in kid


# ---------------------------------------------------------------------------
# 5. Table serialization
# ---------------------------------------------------------------------------

class TestTableSerialization:
    def _make_cell(
        self,
        row: int, col: int,
        row_span: int = 1, col_span: int = 1,
    ) -> TableBorderCell:
        chunk = _text_chunk(0, col * 100.0, 0, (col + 1) * 100.0, 20, value=f"r{row}c{col}")
        cell = TableBorderCell(
            row_number=row, col_number=col,
            row_span=row_span, col_span=col_span,
            _bounding_box=bbox(0, col * 100.0, 0, (col + 1) * 100.0, 20),
        )
        # Add a text node as content
        node = _make_semantic_node(
            SemanticType.PARAGRAPH, 0,
            col * 100.0, 0, (col + 1) * 100.0, 20,
            value=f"r{row}c{col}",
        )
        cell.contents.append(node)
        return cell

    def _make_table(self) -> TableBorder:
        """2x2 table, non-text-block."""
        c00 = self._make_cell(0, 0)
        c01 = self._make_cell(0, 1)
        c10 = self._make_cell(1, 0)
        c11 = self._make_cell(1, 1)
        row0 = TableBorderRow(row_number=0, cells=[c00, c01])
        row1 = TableBorderRow(row_number=1, cells=[c10, c11])
        table = TableBorder(rows=[row0, row1])
        table.set_bounding_box(bbox(0, 0, 0, 200, 40))
        return table

    def _table_kid(self) -> dict:
        table = self._make_table()
        page = Page(page_number=0, width=612, height=792)
        page._kids = [table]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        return json.loads(write_document_json(doc))["kids"][0]

    def test_table_type(self) -> None:
        assert self._table_kid()["type"] == "table"

    def test_table_pdfua_tag(self) -> None:
        assert self._table_kid()["pdfua_tag"] == "Table"

    def test_table_number_of_rows(self) -> None:
        assert self._table_kid()["number of rows"] == 2

    def test_table_number_of_columns(self) -> None:
        assert self._table_kid()["number of columns"] == 2

    def test_table_rows_count(self) -> None:
        assert len(self._table_kid()["rows"]) == 2

    def test_table_row_structure(self) -> None:
        row0 = self._table_kid()["rows"][0]
        assert row0["type"] == "table row"
        assert row0["row number"] == 1  # 1-based
        assert isinstance(row0["cells"], list)

    def test_table_cell_structure(self) -> None:
        cell = self._table_kid()["rows"][0]["cells"][0]
        assert cell["type"] == "table cell"
        assert cell["pdfua_tag"] == "TD"
        assert cell["row number"] == 1  # 1-based
        assert cell["column number"] == 1  # 1-based
        assert cell["row span"] == 1
        assert cell["column span"] == 1
        assert "kids" in cell

    def test_text_block_type(self) -> None:
        """1x1 table is serialized as text block."""
        cell = TableBorderCell(
            row_number=0, col_number=0,
            _bounding_box=bbox(0, 0, 0, 100, 20),
        )
        node = _make_semantic_node(SemanticType.PARAGRAPH, 0, 0, 0, 100, 20, value="block text")
        cell.contents.append(node)
        row = TableBorderRow(row_number=0, cells=[cell])
        table = TableBorder(rows=[row])
        table.set_bounding_box(bbox(0, 0, 0, 100, 20))
        page = Page(page_number=0, width=612, height=792)
        page._kids = [table]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        kid = json.loads(write_document_json(doc))["kids"][0]
        assert kid["type"] == "text block"
        assert "number of rows" not in kid
        assert "rows" not in kid
        assert "kids" in kid


# ---------------------------------------------------------------------------
# 6. List serialization
# ---------------------------------------------------------------------------

class TestListSerialization:
    def _make_list_item(self, text: str, page: int = 0) -> ListItem:
        chunk = _text_chunk(page, 0, 0, 100, 12, value=text, font_name="Arial", font_size=10.0, text_color=[0.0])
        item = ListItem()
        node = _make_semantic_node(SemanticType.PARAGRAPH, page, 0, 0, 100, 12, value=text)
        item.contents.append(node)
        # Attach first_chunk for font info (list item serializer uses first chunk)
        item._first_chunk = chunk  # type: ignore[attr-defined]
        return item

    def _make_list(self) -> PDFList:
        i1 = self._make_list_item("Item one")
        i2 = self._make_list_item("Item two")
        pdf_list = PDFList(items=[i1, i2], numbering_style="bullet")
        return pdf_list

    def _list_kid(self) -> dict:
        pdf_list = self._make_list()
        page = Page(page_number=0, width=612, height=792)
        page._kids = [pdf_list]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        return json.loads(write_document_json(doc))["kids"][0]

    def test_list_type(self) -> None:
        assert self._list_kid()["type"] == "list"

    def test_list_pdfua_tag(self) -> None:
        assert self._list_kid()["pdfua_tag"] == "L"

    def test_list_numbering_style(self) -> None:
        assert self._list_kid()["numbering style"] == "bullet"

    def test_list_number_of_list_items(self) -> None:
        assert self._list_kid()["number of list items"] == 2

    def test_list_items_present(self) -> None:
        assert len(self._list_kid()["list items"]) == 2

    def test_list_item_type(self) -> None:
        item = self._list_kid()["list items"][0]
        assert item["type"] == "list item"

    def test_list_item_pdfua_tag(self) -> None:
        item = self._list_kid()["list items"][0]
        assert item["pdfua_tag"] == "LI"

    def test_list_item_kids(self) -> None:
        item = self._list_kid()["list items"][0]
        assert "kids" in item


# ---------------------------------------------------------------------------
# 7. Image serialization
# ---------------------------------------------------------------------------

class TestImageSerialization:
    def _image_kid(self) -> dict:
        img = ImageChunk(bounding_box=bbox(0, 10, 10, 110, 60))
        page = Page(page_number=0, width=612, height=792)
        page._kids = [img]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        return json.loads(write_document_json(doc))["kids"][0]

    def test_image_type(self) -> None:
        assert self._image_kid()["type"] == "image"

    def test_image_pdfua_tag(self) -> None:
        assert self._image_kid()["pdfua_tag"] == "Figure"

    def test_image_alt_source_missing(self) -> None:
        assert self._image_kid()["alt_source"] == "missing"

    def test_image_no_data_by_default(self) -> None:
        kid = self._image_kid()
        assert "data" not in kid
        assert "source" not in kid

    def test_image_no_alt_by_default(self) -> None:
        kid = self._image_kid()
        assert "alt" not in kid


# ---------------------------------------------------------------------------
# 8. LineArtChunk skipping
# ---------------------------------------------------------------------------

class TestLineArtSkipping:
    def test_line_art_skipped_at_top_level(self) -> None:
        """LineArtChunk objects must be silently skipped by the writer."""
        la = LineArtChunk(bounding_box=bbox(0, 0, 0, 100, 10))
        page = Page(page_number=0, width=612, height=792)
        page._kids = [la]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        assert result["kids"] == []

    def test_line_art_skipped_inside_table_cell(self) -> None:
        """LineArtChunk inside a table cell must also be skipped."""
        la = LineArtChunk(bounding_box=bbox(0, 0, 0, 100, 10))
        cell = TableBorderCell(row_number=0, col_number=0, _bounding_box=bbox(0, 0, 0, 100, 20))
        cell.contents.append(la)
        node = _make_semantic_node(SemanticType.PARAGRAPH, 0, 0, 0, 100, 20, value="text")
        cell.contents.append(node)
        row = TableBorderRow(row_number=0, cells=[cell])
        # Force is_text_block = True (1x1) → we only get "kids" array
        table = TableBorder(rows=[row])
        table.set_bounding_box(bbox(0, 0, 0, 100, 20))
        page = Page(page_number=0, width=612, height=792)
        page._kids = [table]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        kids_in_block = result["kids"][0]["kids"]
        # LineArt must not appear
        for child in kids_in_block:
            assert child.get("type") != "line"


# ---------------------------------------------------------------------------
# 9. Multi-page document — "kids" aggregated across pages
# ---------------------------------------------------------------------------

class TestMultiPage:
    def test_kids_from_all_pages(self) -> None:
        node0 = _make_semantic_node(SemanticType.PARAGRAPH, 0, 0, 0, 100, 20, value="Page 1")
        node1 = _make_semantic_node(SemanticType.PARAGRAPH, 1, 0, 0, 100, 20, value="Page 2")

        page0 = Page(page_number=0, width=612, height=792)
        page0._kids = [node0]  # type: ignore[attr-defined]
        page1 = Page(page_number=1, width=612, height=792)
        page1._kids = [node1]  # type: ignore[attr-defined]

        doc = Document()
        doc.push_page(page0)
        doc.push_page(page1)
        result = json.loads(write_document_json(doc))
        assert len(result["kids"]) == 2
        assert result["kids"][0]["page number"] == 1
        assert result["kids"][1]["page number"] == 2

    def test_number_of_pages_matches(self) -> None:
        doc = Document()
        doc.push_page(Page(page_number=0, width=612, height=792))
        doc.push_page(Page(page_number=1, width=612, height=792))
        result = json.loads(write_document_json(doc))
        assert result["number of pages"] == 2


# ---------------------------------------------------------------------------
# 10. pdfua_tag mapping — comprehensive
# ---------------------------------------------------------------------------

class TestPdfuaTagMapping:
    def _tag_for_node(self, stype: SemanticType, heading_level: int | None = None) -> str | None:
        node = _make_semantic_node(stype, 0, 0, 0, 100, 20, heading_level=heading_level)
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        return result["kids"][0].get("pdfua_tag")

    def test_paragraph_tag(self) -> None:
        assert self._tag_for_node(SemanticType.PARAGRAPH) == "P"

    def test_heading_h1(self) -> None:
        assert self._tag_for_node(SemanticType.HEADING, heading_level=1) == "H1"

    def test_heading_h6(self) -> None:
        assert self._tag_for_node(SemanticType.HEADING, heading_level=6) == "H6"

    def test_heading_no_level(self) -> None:
        assert self._tag_for_node(SemanticType.HEADING, heading_level=None) == "H"

    def test_caption_no_tag(self) -> None:
        assert self._tag_for_node(SemanticType.CAPTION) is None

    def test_header_no_tag(self) -> None:
        assert self._tag_for_node(SemanticType.HEADER) is None

    def test_footer_no_tag(self) -> None:
        assert self._tag_for_node(SemanticType.FOOTER) is None


# ---------------------------------------------------------------------------
# 11. Bounding-box rounding
# ---------------------------------------------------------------------------

class TestBoundingBoxRounding:
    """Every coordinate in bounding box must go through round_double."""

    def test_bbox_0_0005_rounds_up(self) -> None:
        node = _make_semantic_node(SemanticType.PARAGRAPH, 0, 0.0005, 0.0005, 0.0015, 0.0015)
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        bb = result["kids"][0]["bounding box"]
        # 0.0005 → 0.001, 0.0015 → 0.002
        assert bb[0] == pytest.approx(0.001, abs=1e-9)
        assert bb[1] == pytest.approx(0.001, abs=1e-9)
        assert bb[2] == pytest.approx(0.002, abs=1e-9)
        assert bb[3] == pytest.approx(0.002, abs=1e-9)

    def test_bbox_745_1316(self) -> None:
        node = _make_semantic_node(SemanticType.HEADING, 0, 0, 0, 100, 745.1316, heading_level=1)
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        bb = result["kids"][0]["bounding box"]
        assert bb[3] == pytest.approx(745.132, abs=1e-9)


# ---------------------------------------------------------------------------
# 12. Field ordering (base fields always first)
# ---------------------------------------------------------------------------

class TestFieldOrdering:
    def test_base_fields_before_type_specific(self) -> None:
        node = _make_semantic_node(SemanticType.HEADING, 0, 0, 0, 100, 20, heading_level=2)
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        kid = result["kids"][0]
        keys = list(kid.keys())
        # type comes first
        assert keys[0] == "type"
        # page number before bounding box
        assert keys.index("page number") < keys.index("bounding box")
        # bounding box before heading level, font, content
        assert keys.index("bounding box") < keys.index("heading level")
        assert keys.index("heading level") < keys.index("font")
        assert keys.index("font") < keys.index("content")

    def test_id_and_level_before_page_number(self) -> None:
        node = _make_semantic_node(SemanticType.HEADING, 0, 0, 0, 100, 20, heading_level=1)
        node.recognized_structure_id = 5
        node.level = "Doctitle"
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        kid = result["kids"][0]
        keys = list(kid.keys())
        assert keys.index("id") < keys.index("page number")
        assert keys.index("level") < keys.index("page number")

    def test_pdfua_tag_before_id(self) -> None:
        node = _make_semantic_node(SemanticType.HEADING, 0, 0, 0, 100, 20, heading_level=1)
        node.recognized_structure_id = 1
        page = Page(page_number=0, width=612, height=792)
        page._kids = [node]  # type: ignore[attr-defined]
        doc = Document()
        doc.push_page(page)
        result = json.loads(write_document_json(doc))
        kid = result["kids"][0]
        keys = list(kid.keys())
        assert keys.index("pdfua_tag") < keys.index("id")
