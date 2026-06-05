# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
#
# Track B parsing bridge for Python: turn a PDF into the entity model using
# pypdf for text extraction (proven viable in spike/py-spike — correct
# CID->Unicode and per-chunk positions via visitor_text) and pikepdf for page
# geometry and image XObjects. The CTM math that recovers page coordinates from
# the text matrix is lifted from spike2.py's INV-4/INV-7 analysis. No veraPDF
# source copied.
"""PDF -> Document parser.

Extraction strategy (Track B):
- **Text**: pypdf's ``extract_text(visitor_text=...)`` yields each text run with
  its current transformation matrix (cm) and text matrix (tm). Applying cm to
  the tm origin gives the run's page coordinates; the chunk box is sized from
  that origin, the run width, and the device font size.
- **Geometry / images**: pikepdf gives the crop box (page dimensions) and the
  ``/XObject`` image list per page.

Every critical step logs: document open, metadata, per-page entry, chunk
counts, and any recoverable per-page error (logged at WARNING and skipped so a
single bad page never aborts the document).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pikepdf
import pypdf

from odl_pdf.entities import (
    BoundingBox,
    Document,
    DocumentMetadata,
    ImageChunk,
    LineArtChunk,
    Page,
    TextChunk,
)
from odl_pdf.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def _build_run_widths(pike_page: pikepdf.Page) -> dict[tuple[float, float], float]:
    """Pre-compute per-run page-space advance widths from the page's content stream.

    Parses the content stream operators to build a mapping from
    ``(tm_x, tm_y)`` text-matrix origin to the run's glyph-advance width in
    **page-space points**.  This uses the CIDFont ``/W`` width table (or ``/DW``
    default) together with the TJ kerning adjustments and the current CTM
    horizontal scale to recover the exact advance that a PDF renderer would use.

    The mapping is consumed by the visitor_text callback in ``_parse_page`` to
    replace the crude ``0.5 * fs * len(text)`` estimate.  Runs whose position
    is not in the map (rare for PDFs that set a new Tm before every TJ, which
    is the common generator pattern) fall back to the estimate, so correctness
    is never worse than before.

    Width method is logged at DEBUG level as ``glyph`` or ``fallback``.
    """
    pos_to_advance: dict[tuple[float, float], float] = {}

    # Per-font CID->width map (1/1000 em units).  Rebuilt on each Tf.
    cid_w_map: dict[int, int] = {}
    default_w: int = 1000
    current_fs_text: float | None = None
    current_tm_pos: tuple[float, float] = (0.0, 0.0)
    # Horizontal scale component of the current CTM (cm[0], the 'a' coefficient).
    ctm_a: float = 1.0

    try:
        font_resources: dict[str, object] = {}
        try:
            res = pike_page["/Resources"]
            fonts = res.get("/Font")
            if fonts is not None:
                font_resources = {str(k): v for k, v in fonts.items()}
        except Exception:  # noqa: BLE001
            pass

        for operands, operator in pikepdf.parse_content_stream(pike_page):
            op = str(operator)

            if op == "cm":
                ctm_a = float(operands[0])

            elif op == "Tf":
                # /FontName fs Tf — reload width table for this font.
                current_fs_text = float(operands[1])
                cid_w_map, default_w = _load_cid_widths(
                    str(operands[0]), font_resources
                )

            elif op == "Tm":
                current_tm_pos = (float(operands[4]), float(operands[5]))

            elif op in ("TJ", "Tj"):
                if current_fs_text is None or current_fs_text == 0:
                    continue

                # Sum glyph advances in 1/1000-em units.
                total_1000em: float = 0.0
                if op == "TJ":
                    tj_array = operands[0]
                    for item in tj_array:
                        if isinstance(item, (Decimal, int, float)):
                            # PDF spec: negative kerning *increases* advance.
                            total_1000em -= float(item)
                        else:
                            total_1000em += _sum_cid_advances(
                                item, cid_w_map, default_w
                            )
                else:
                    # Tj: single string operand, no kerning numbers.
                    total_1000em += _sum_cid_advances(
                        operands[0], cid_w_map, default_w
                    )

                # Convert to page-space points:
                # advance_page = (total_1000em / 1000) * fs_text * |ctm_a|
                advance_pts = (total_1000em / 1000.0) * current_fs_text * abs(ctm_a)
                pos_to_advance[current_tm_pos] = advance_pts

    except Exception:  # noqa: BLE001 — never abort page extraction
        logger.debug("run-width pre-scan failed; will use estimate for all runs")

    return pos_to_advance


def _load_cid_widths(
    font_name: str,
    font_resources: dict[str, object],
) -> tuple[dict[int, int], int]:
    """Return ``(cid_w_map, default_w)`` for *font_name* from the page resources.

    ``cid_w_map`` maps CID integer to advance width in 1/1000-em units.
    ``default_w`` is the fallback when a CID is not in the map (from ``/DW``).

    Supports CIDFont Type0/Type2 fonts with a ``/DescendantFonts`` array.
    Returns an empty map with ``default_w = 1000`` if the font is not found or
    has no parseable width table.
    """
    try:
        font_obj = font_resources.get(font_name)
        if font_obj is None:
            return {}, 1000

        desc_fonts = font_obj.get("/DescendantFonts")  # type: ignore[union-attr]
        if desc_fonts is None or len(desc_fonts) == 0:
            return {}, 1000

        cid_font = desc_fonts[0]
        default_w = int(cid_font.get("/DW", 1000))  # type: ignore[arg-type]
        w_array = cid_font.get("/W")  # type: ignore[union-attr]
        if w_array is None:
            return {}, default_w

        # Parse /W: alternating (start_cid, [w0 w1 ...]) or (c1, c2, w) entries.
        cid_w_map: dict[int, int] = {}
        i = 0
        while i < len(w_array):
            entry = w_array[i]
            # Avoid importing pikepdf array type at module scope — use duck-typing.
            try:
                cid_start = int(entry)
            except (TypeError, ValueError):
                i += 1
                continue
            if i + 1 >= len(w_array):
                break
            next_entry = w_array[i + 1]
            try:
                int(next_entry)  # is it a scalar? then range form: c1 c2 w
                if i + 2 >= len(w_array):
                    break
                cid_end = int(next_entry)
                w = int(w_array[i + 2])
                for cid in range(cid_start, cid_end + 1):
                    cid_w_map[cid] = w
                i += 3
            except (TypeError, ValueError):
                # array form: cid_start [w0 w1 ...]
                for j, w_val in enumerate(next_entry):
                    cid_w_map[cid_start + j] = int(w_val)
                i += 2

        return cid_w_map, default_w

    except Exception:  # noqa: BLE001
        return {}, 1000


def _sum_cid_advances(
    pdf_string: object,
    cid_w_map: dict[int, int],
    default_w: int,
) -> float:
    """Sum glyph advances (1/1000 em) for all 2-byte CIDs in *pdf_string*.

    Handles pikepdf's hex-encoded string representation (``<XXXX...>``) and
    raw byte strings.  Unknown CIDs use *default_w*.
    """
    try:
        raw_repr = pdf_string.unparse().decode("ascii", errors="replace").strip()  # type: ignore[union-attr]
        if raw_repr.startswith("<") and raw_repr.endswith(">"):
            raw_bytes = bytes.fromhex(raw_repr[1:-1])
        else:
            # Parenthesised literal string — unlikely for CIDFont but handle it.
            raw_bytes = bytes(pdf_string)  # type: ignore[call-overload]
    except Exception:  # noqa: BLE001
        return 0.0

    total: float = 0.0
    for i in range(0, len(raw_bytes) - 1, 2):
        cid = (raw_bytes[i] << 8) | raw_bytes[i + 1]
        total += cid_w_map.get(cid, default_w)
    return total


def _apply_ctm(cm: list[float], tx: float, ty: float) -> tuple[float, float]:
    """Apply CTM ``[a b c d e f]`` to point ``(tx, ty)``.

    Mirrors spike2.py: ``x' = a*tx + c*ty + e``, ``y' = b*tx + d*ty + f``.
    """
    a, b, c, d, e, f = cm
    return a * tx + c * ty + e, b * tx + d * ty + f


class PdfParser:
    """Parses a single PDF file into a :class:`Document`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def parse(self) -> Document:
        """Open the PDF and extract every page into the entity model."""
        if not self.path.exists():
            logger.error("PDF not found: %s", self.path)
            raise FileNotFoundError(self.path)

        logger.info("Opening %s (%d bytes)", self.path.name, self.path.stat().st_size)

        reader = pypdf.PdfReader(str(self.path))
        pike = pikepdf.open(str(self.path))
        try:
            metadata = self._extract_metadata(reader)
            document = Document(metadata=metadata)

            page_count = len(reader.pages)
            logger.info("Document has %d page(s)", page_count)

            for index in range(page_count):
                try:
                    page = self._parse_page(index, reader, pike)
                    document.push_page(page)
                except Exception:  # noqa: BLE001 — never let one page abort the doc
                    logger.warning(
                        "Page %d failed to parse; emitting empty page", index + 1,
                        exc_info=True,
                    )
                    document.push_page(Page(index, 0.0, 0.0))

            total_text = sum(len(p.text_chunks) for p in document.pages)
            total_img = sum(len(p.image_chunks) for p in document.pages)
            logger.info(
                "Parsed %s: %d page(s), %d text chunk(s), %d image(s)",
                self.path.name, document.number_of_pages, total_text, total_img,
            )
            return document
        finally:
            pike.close()

    def _extract_metadata(self, reader: pypdf.PdfReader) -> DocumentMetadata:
        meta = reader.metadata
        out = DocumentMetadata(
            file_name=self.path.name,
            author=str(meta.author) if meta and meta.author else None,
            title=str(meta.title) if meta and meta.title else None,
            creation_date=str(meta.get("/CreationDate")) if meta else None,
            modification_date=str(meta.get("/ModDate")) if meta else None,
        )
        logger.debug(
            "Metadata: title=%r author=%r created=%r",
            out.title, out.author, out.creation_date,
        )
        return out

    def _parse_page(
        self, index: int, reader: pypdf.PdfReader, pike: pikepdf.Pdf
    ) -> Page:
        width, height = self._page_dimensions(index, pike)
        page = Page(index, width, height)
        logger.debug("Page %d: %.1f x %.1f pt", index + 1, width, height)

        # --- text chunks via pypdf visitor ---
        chunks: list[TextChunk] = []

        # Pre-compute per-run glyph-advance widths from the content stream so
        # the visitor can use accurate measurements instead of the crude
        # 0.5-em-per-char estimate.  The map keys are (tm_x, tm_y) text-matrix
        # origins; values are page-space advance widths in points.
        run_widths = _build_run_widths(pike.pages[index])
        logger.debug(
            "Page %d: pre-scanned %d run widths via glyph advances",
            index + 1,
            len(run_widths),
        )
        _glyph_hits = [0]
        _fallback_hits = [0]

        def visitor(text, cm, tm, font_dict, font_size):
            if not text or not text.strip():
                return
            # Guard: Jira/wkhtmltopdf-generated PDFs with y-flipped CTM frames
            # emit a PHANTOM pre-position run at the CTM translation origin
            # before the actual Td/Tm repositioning. pypdf reports it at
            # (cm[4], cm[5]) because tm origin is (0,0); the real glyph renders
            # elsewhere. These phantoms inflate the chunk count ~26x, trip
            # false frame-rejection in bordered-table detection, and shatter
            # group_lines by injecting a baseline outlier between real glyphs.
            # Drop ONLY when tm origin is (0,0) AND the CTM is y-flipped
            # (cm[3] < 0 — the hallmark of HTML-to-PDF exporters). Normal
            # LaTeX/prose PDFs use positive cm[3]; their tm=(0,0) runs carry
            # real content (verified: 2408.02509v1, 1901.03003) and must NOT
            # be dropped.
            if (
                tm and abs(tm[4]) < 0.01 and abs(tm[5]) < 0.01
                and cm and cm[3] < 0
            ):
                return
            try:
                x, y = _apply_ctm(cm, tm[4], tm[5]) if cm and tm else (tm[4], tm[5])
                # Effective font size in PAGE space.
                # pypdf passes the raw Tf operand as font_size (text-space
                # nominal, NOT text-matrix scaled). The full rendered size is
                #   font_size * |tm[3]| * |cm[3]|
                # where tm[3] is the text-matrix vertical scale and cm[3] the
                # CTM vertical scale. For normal PDFs tm[3]=1.0 so this is
                # backward-compatible (e.g. lorem heading 267*1.0*0.12=32.0,
                # matching the JAR oracle 32.005). Omitting |tm[3]| caused a 33x
                # UNDER-estimate on PDFs that set Tf=1 and scale via the text
                # matrix (e.g. MGT-F14: 1.0*33.25*0.585=19.5pt, not 0.585pt) —
                # which collapsed the line-grouping gap and shattered the page
                # into one glyph per line (0% word recall).
                cm_scale = abs(cm[3]) if cm and cm[3] else 1.0
                tm_scale = abs(tm[3]) if tm and tm[3] else 1.0
                base = float(font_size) if font_size else 1.0
                fs = base * tm_scale * cm_scale

                # Width: prefer the pre-scanned glyph advance for this run (keyed
                # by its text-matrix origin).  Fall back to 0.5-em estimate only
                # when the position is not in the map (e.g. multi-TJ after a
                # single Tm, or content-stream parse failure).
                key = (tm[4], tm[5])
                if key in run_widths:
                    width_est = run_widths[key]
                    _glyph_hits[0] += 1
                else:
                    width_est = 0.5 * fs * len(text)
                    _fallback_hits[0] += 1

                bbox = BoundingBox.of(index, x, y, x + width_est, y + fs)
                chunk = TextChunk(
                    bbox,
                    value=text,
                    font_name=_font_name(font_dict),
                    font_size=fs,
                )
                chunk.index = len(chunks)
                chunks.append(chunk)
            except Exception:  # noqa: BLE001
                logger.warning("Dropping text run %r on page %d", text[:20], index + 1,
                               exc_info=True)

        try:
            reader.pages[index].extract_text(visitor_text=visitor)
        except Exception:  # noqa: BLE001
            logger.warning("Text extraction failed on page %d", index + 1, exc_info=True)

        for chunk in chunks:
            page.push_text(chunk)
        logger.debug(
            "Page %d: extracted %d text chunk(s) "
            "(width method: %d glyph, %d fallback)",
            index + 1,
            len(chunks),
            _glyph_hits[0],
            _fallback_hits[0],
        )

        # --- image chunks via pikepdf XObjects ---
        image_count = self._extract_images(index, pike, page)
        if image_count:
            logger.debug("Page %d: %d image XObject(s)", index + 1, image_count)

        # --- line-art chunks via content-stream path operators ---
        line_art_count = self._extract_line_art(index, pike, page)
        if line_art_count:
            logger.debug("Page %d: %d line-art path(s)", index + 1, line_art_count)

        return page

    def _page_dimensions(self, index: int, pike: pikepdf.Pdf) -> tuple[float, float]:
        try:
            box = pike.pages[index].MediaBox
            x0, y0, x1, y1 = (float(v) for v in box)
            return abs(x1 - x0), abs(y1 - y0)
        except Exception:  # noqa: BLE001
            logger.warning("No MediaBox on page %d; using 0x0", index + 1, exc_info=True)
            return 0.0, 0.0

    def _extract_images(self, index: int, pike: pikepdf.Pdf, page: Page) -> int:
        count = 0
        try:
            resources = pike.pages[index].get("/Resources")
            xobjects = resources.get("/XObject") if resources else None
            if not xobjects:
                return 0
            for _name, xobj in xobjects.items():
                if str(xobj.get("/Subtype", "")) == "/Image":
                    # Image placement requires the CTM at the Do operator; until
                    # the content-stream interpreter lands we record a
                    # zero-origin box sized by the image's own pixel dims so the
                    # chunk exists and downstream counts are correct.
                    w = float(xobj.get("/Width", 0))
                    h = float(xobj.get("/Height", 0))
                    page.push_image(ImageChunk(BoundingBox.of(index, 0.0, 0.0, w, h)))
                    count += 1
        except Exception:  # noqa: BLE001
            logger.warning("Image scan failed on page %d", index + 1, exc_info=True)
        return count

    def _extract_line_art(self, index: int, pike: pikepdf.Pdf, page: Page) -> int:
        """Emit one LineArtChunk per painted path in the content stream.

        Tracks the CTM through ``cm``/``q``/``Q`` and the path-construction
        operators (``m``/``l``/``re``/``c``/``v``/``y``). On a paint operator
        (``S``/``s``/``f``/``F``/``B``/``b`` and their ``*`` variants) the
        accumulated points' bounding box — transformed by the current CTM — is
        recorded. This is the line-art evidence the table-border detector and
        the hybrid triage's vector-table signal consume.
        """
        count = 0
        try:
            pdf_page = pike.pages[index]
            ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            ctm_stack: list[list[float]] = []
            pts: list[tuple[float, float]] = []
            paint_ops = {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}

            def flush_path() -> int:
                nonlocal pts
                if not pts:
                    return 0
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                pts = []
                page.push_line_art(
                    LineArtChunk(BoundingBox.of(index, min(xs), min(ys), max(xs), max(ys)))
                )
                return 1

            for operands, operator in pikepdf.parse_content_stream(pdf_page):
                op = str(operator)
                if op == "q":
                    ctm_stack.append(list(ctm))
                elif op == "Q":
                    if ctm_stack:
                        ctm = ctm_stack.pop()
                elif op == "cm":
                    m = [float(o) for o in operands]
                    ctm = _mat_mul(m, ctm)
                elif op in ("m", "l"):
                    x, y = float(operands[0]), float(operands[1])
                    pts.append(_apply_ctm(ctm, x, y))
                elif op == "re":
                    x, y, w, h = (float(o) for o in operands)
                    for cx, cy in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
                        pts.append(_apply_ctm(ctm, cx, cy))
                elif op in ("c", "v", "y"):
                    # Bezier: record the endpoint (and control points for bounds).
                    coords = [float(o) for o in operands]
                    for i in range(0, len(coords) - 1, 2):
                        pts.append(_apply_ctm(ctm, coords[i], coords[i + 1]))
                elif op in paint_ops:
                    count += flush_path()
                elif op == "n":
                    pts = []  # path ended with no paint (clip/no-op)
        except Exception:  # noqa: BLE001
            logger.warning("Line-art scan failed on page %d", index + 1, exc_info=True)
        return count


def _mat_mul(a: list[float], b: list[float]) -> list[float]:
    """Compose two PDF matrices ``a`` then ``b`` (both [a b c d e f])."""
    a0, a1, a2, a3, a4, a5 = a
    b0, b1, b2, b3, b4, b5 = b
    return [
        a0 * b0 + a1 * b2,
        a0 * b1 + a1 * b3,
        a2 * b0 + a3 * b2,
        a2 * b1 + a3 * b3,
        a4 * b0 + a5 * b2 + b4,
        a4 * b1 + a5 * b3 + b5,
    ]


def _font_name(font_dict) -> str:
    if not font_dict:
        return ""
    for key in ("/BaseFont", "/FontName"):
        if key in font_dict:
            return str(font_dict[key]).lstrip("/")
    return ""


def parse_pdf(path: str | Path) -> Document:
    """Convenience wrapper: parse ``path`` and return the :class:`Document`."""
    return PdfParser(path).parse()
