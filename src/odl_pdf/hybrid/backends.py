# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Backend adapters. MockAdapter is deterministic and offline (for tests + the
# debugging loop). BedrockAdapter is the real AWS Bedrock vision backend
# recommended in PRD §9 — it renders a hard page to PNG and asks a Claude vision
# model for structured JSON, then maps that JSON into our IObject types
# (reusing the Docling-style schema described in
# docs/architecture/08-hybrid-ai-mode.md §15).
"""Mock and AWS Bedrock vision backend adapters."""

from __future__ import annotations

import json
import os

from odl_pdf.entities import (
    BoundingBox,
    IObject,
    SemanticTextNode,
    SemanticType,
    TableBorder,
    TableBorderCell,
    TableBorderRow,
    TextBlock,
    TextChunk,
    TextColumn,
    TextLine,
)
from odl_pdf.hybrid.adapter import BackendAdapter, BackendError, BackendRegistry
from odl_pdf.hybrid.config import HybridConfig
from odl_pdf.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Shared: backend JSON schema -> IObject
# --------------------------------------------------------------------------
def schema_to_objects(elements: list[dict], page_number: int, page_height: float) -> list[IObject]:
    """Map a list of backend element dicts into IObjects.

    Element schema (DoclingDocument-inspired, see §15):
      {"type": "heading"|"paragraph"|"table",
       "bbox": [left, bottom, right, top],   # PDF points, BOTTOMLEFT origin
       "text": "...",                          # for text elements
       "level": 1,                             # headings only
       "cells": [{"row":r,"col":c,"row_span":1,"col_span":1,"text":"..."}]}  # tables
    """
    objects: list[IObject] = []
    for el in elements:
        etype = el.get("type")
        bbox = el.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        box = BoundingBox.of(page_number, *bbox)
        try:
            if etype == "table":
                objects.append(_table_from_schema(el, page_number))
            elif etype in ("heading", "paragraph", "caption", "list_item"):
                objects.append(_text_node_from_schema(etype, el, box))
            else:
                logger.debug("skipping unknown backend element type %r", etype)
        except Exception:  # noqa: BLE001
            logger.warning("failed to map backend element %r", etype, exc_info=True)
    return objects


def _text_node_from_schema(etype: str, el: dict, box: BoundingBox) -> SemanticTextNode:
    chunk = TextChunk(box, value=el.get("text", ""), font_size=float(el.get("font_size", 0.0)))
    column = TextColumn([TextBlock([TextLine([chunk])])])
    if etype == "heading":
        node = SemanticTextNode.heading(int(el.get("level", 1)), [column])
    else:
        node = SemanticTextNode(_TYPE_MAP.get(etype, SemanticType.PARAGRAPH), [column])
    return node


_TYPE_MAP = {
    "paragraph": SemanticType.PARAGRAPH,
    "caption": SemanticType.CAPTION,
    "list_item": SemanticType.LIST_ITEM,
}


def _table_from_schema(el: dict, page_number: int) -> TableBorder:
    rows_by_index: dict[int, TableBorderRow] = {}
    for cell in el.get("cells", []):
        r = int(cell.get("row", 0))
        c = int(cell.get("col", 0))
        tc = TableBorderCell(
            r, c,
            row_span=int(cell.get("row_span", 1)),
            col_span=int(cell.get("col_span", 1)),
        )
        cb = cell.get("bbox")
        if cb:
            tc.set_bounding_box(BoundingBox.of(page_number, *cb))
        if cell.get("text"):
            tc.add_content(TextChunk(
                tc._bounding_box or BoundingBox.of(page_number, 0, 0, 0, 0),
                value=cell["text"],
            ))
        rows_by_index.setdefault(r, TableBorderRow(r)).cells.append(tc)
    rows = [rows_by_index[i] for i in sorted(rows_by_index)]
    return TableBorder(rows)


# --------------------------------------------------------------------------
# MockAdapter — deterministic, offline
# --------------------------------------------------------------------------
class MockAdapter:
    """Deterministic backend for tests and the debugging loop.

    Returns a fixed paragraph + 2x2 table per page, derived only from the page
    geometry — no network, no randomness — so orchestration and merge logic can
    be tested without Bedrock.
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "mock"

    def check_availability(self) -> None:
        return None

    def convert_page(
        self, page_number: int, image_png: bytes, page_width: float, page_height: float
    ) -> list[IObject]:
        logger.debug("mock backend converting page %d (%d bytes img)", page_number + 1, len(image_png))
        elements = [
            {
                "type": "heading", "level": 1, "text": f"Mock Page {page_number + 1}",
                "bbox": [50.0, page_height - 60.0, page_width - 50.0, page_height - 40.0],
            },
            {
                "type": "table",
                "bbox": [50.0, 100.0, page_width - 50.0, page_height - 80.0],
                "cells": [
                    {"row": 0, "col": 0, "text": "A"},
                    {"row": 0, "col": 1, "text": "B"},
                    {"row": 1, "col": 0, "text": "1"},
                    {"row": 1, "col": 1, "text": "2"},
                ],
            },
        ]
        return schema_to_objects(elements, page_number, page_height)


# --------------------------------------------------------------------------
# BedrockAdapter — real AWS Bedrock vision
# --------------------------------------------------------------------------
_VISION_PROMPT = """You are a PDF page structure extractor. Look at this page image and return ONLY a JSON object (no prose, no markdown fences) of the form:
{"elements": [
  {"type": "heading"|"paragraph"|"caption"|"list_item", "text": "...", "level": 1, "bbox": [left, bottom, right, top]},
  {"type": "table", "bbox": [left, bottom, right, top], "cells": [{"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "..."}]}
]}
Coordinates are PDF points with a BOTTOM-LEFT origin; the page is %(w).0f wide and %(h).0f tall. Capture every table cell. Preserve reading order."""


class BedrockAdapter:
    """AWS Bedrock vision backend (Claude) — PRD §9 recommended internal backend.

    Renders are passed in as PNG bytes by the orchestrator; this adapter sends
    them to a Claude vision model via the Bedrock Converse API and maps the
    returned JSON to IObjects. Non-deterministic by nature — the local engine
    stays the conformance oracle (PRD §8).
    """

    def __init__(self, config: HybridConfig) -> None:
        self._config = config
        # Strip the Claude Code "[1m]" context-window suffix from the env model
        # id to get the raw Bedrock model id.
        raw = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "")
        self._model_id = raw.split("[")[0] or "us.anthropic.claude-sonnet-4-6"
        self._region = os.environ.get("AWS_REGION", "us-west-2")
        self._client = None

    @property
    def name(self) -> str:
        return "bedrock-claude"

    def _bedrock(self):
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
            except ImportError as e:  # pragma: no cover
                raise BackendError("bedrock backend needs 'odl-pdf[hybrid]' (boto3)") from e
            # Vision + large structured JSON can take minutes; boto3's 60s
            # default read timeout is far too short. Honor the configured
            # timeout (0 => generous 600s default), with one retry.
            read_timeout = (self._config.timeout_ms / 1000.0) if self._config.timeout_ms else 600.0
            boto_cfg = BotoConfig(
                read_timeout=read_timeout,
                connect_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
            )
            logger.info(
                "bedrock client: model=%s region=%s read_timeout=%.0fs",
                self._model_id, self._region, read_timeout,
            )
            self._client = boto3.client(
                "bedrock-runtime", region_name=self._region, config=boto_cfg
            )
        return self._client

    def check_availability(self) -> None:
        # Construct the client; credential/region errors surface here, fail-fast.
        try:
            self._bedrock()
        except Exception as e:  # noqa: BLE001
            raise BackendError(f"bedrock unavailable: {e}") from e

    def convert_page(
        self, page_number: int, image_png: bytes, page_width: float, page_height: float
    ) -> list[IObject]:
        client = self._bedrock()
        prompt = _VISION_PROMPT % {"w": page_width, "h": page_height}
        logger.info("bedrock converting page %d (%d KB img)", page_number + 1, len(image_png) // 1024)
        try:
            resp = client.converse(
                modelId=self._model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": image_png}}},
                        {"text": prompt},
                    ],
                }],
                inferenceConfig={"maxTokens": 16384, "temperature": 0.0},
            )
        except Exception as e:  # noqa: BLE001
            raise BackendError(f"bedrock converse failed on page {page_number + 1}: {e}") from e

        text = "".join(
            b.get("text", "") for b in resp["output"]["message"]["content"] if "text" in b
        )
        if resp.get("stopReason") == "max_tokens":
            logger.warning(
                "bedrock page %d hit max_tokens; JSON may be truncated, "
                "salvaging complete elements", page_number + 1,
            )
        elements = _parse_json_elements(text, page_number)
        logger.info("bedrock page %d -> %d element(s)", page_number + 1, len(elements))
        return schema_to_objects(elements, page_number, page_height)


def _parse_json_elements(text: str, page_number: int) -> list[dict]:
    """Extract the ``elements`` array from the model's text.

    Tolerant of markdown fences and of truncation: if the whole object fails to
    parse (e.g. the response was cut at max_tokens mid-array), fall back to
    salvaging each complete ``{...}`` element object with a brace-depth scan.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]

    # Fast path: the whole object parses.
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(t[start : end + 1]).get("elements", [])
        except json.JSONDecodeError:
            pass  # fall through to salvage

    # Salvage path: scan the elements array for complete top-level objects.
    salvaged = _salvage_elements(t)
    logger.warning(
        "bedrock page %d: JSON not fully parseable; salvaged %d element(s)",
        page_number + 1, len(salvaged),
    )
    return salvaged


def _salvage_elements(text: str) -> list[dict]:
    """Recover complete element objects from a (possibly truncated) elements array."""
    arr_start = text.find('"elements"')
    if arr_start < 0:
        return []
    i = text.find("[", arr_start)
    if i < 0:
        return []
    out: list[dict] = []
    depth = 0
    obj_start = -1
    in_str = False
    escape = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = j
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                try:
                    out.append(json.loads(text[obj_start : j + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = -1
        elif ch == "]" and depth == 0:
            break
    return out


# Register both backends at import time (the plug-in seam).
BackendRegistry.register("mock", MockAdapter)
BackendRegistry.register("bedrock-claude", BedrockAdapter)
