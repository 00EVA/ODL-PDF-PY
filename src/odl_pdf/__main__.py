# Copyright 2026 the OpenDataLoader-PDF Python authors.
# SPDX-License-Identifier: MIT
"""CLI entrypoint: ``python -m odl_pdf <file.pdf> [-f json|summary]``.

Runs the full extraction pipeline (parse -> group -> reading-order) and emits
either a per-page summary (default) or the oracle-shaped JSON. ``--debug`` (or
``ODL_LOG_LEVEL=DEBUG``) turns on the full per-stage trace.  ``--trace`` (or
``ODL_LOG_LEVEL=TRACE``) enables the even-more-verbose per-page stats dump.

Env vars honoured at startup (in addition to ``ODL_LOG_LEVEL``):

* ``ODL_LOG_COLOR``   — ``0``/``1`` force colour off/on.
* ``ODL_LOG_TIMING``  — ``1`` to emit per-stage timing lines.
* ``ODL_LOG_FORMAT``  — ``plain`` or ``rich``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import sys

from odl_pdf.logging_config import configure, get_logger, TRACE
from odl_pdf.output.json_writer import write_document_json
from odl_pdf.output.markdown_writer import document_to_markdown
from odl_pdf.pipeline import extract

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="odl_pdf", description="Extract structure from a PDF.")
    ap.add_argument("pdf", help="path to a PDF file")
    ap.add_argument(
        "-f", "--format", choices=["summary", "json", "markdown"], default="summary",
        help="output format (default: summary)",
    )
    ap.add_argument("-o", "--output", help="write output to this file instead of stdout")
    ap.add_argument("--debug", action="store_true", help="enable DEBUG logging")
    ap.add_argument(
        "--trace", action="store_true",
        help="enable TRACE (very-verbose) logging, including per-page stats",
    )
    ap.add_argument(
        "--timing", action="store_true",
        help="emit per-stage elapsed-time lines (implies --debug)",
    )
    args = ap.parse_args(argv)

    if args.trace:
        # TRACE beats DEBUG; configure() also sets ODL_TRACE=1.
        configure(TRACE)
    elif args.debug:
        configure(logging.DEBUG)
    else:
        configure()  # respects ODL_LOG_LEVEL env var, defaults to INFO

    if args.timing:
        os.environ["ODL_LOG_TIMING"] = "1"

    t0 = time.perf_counter()
    try:
        document = extract(args.pdf)
    except FileNotFoundError:
        logger.error("File not found: %s", args.pdf)
        return 1

    elapsed = time.perf_counter() - t0

    # Per-page TRACE stats (only fires when ODL_TRACE=1).
    from odl_pdf.debug import dump_page_stats, run_summary
    for page in document.pages:
        dump_page_stats(page, getattr(page, "_kids", []))
    run_summary(document, elapsed_s=elapsed)

    if args.format in ("json", "markdown"):
        out = (
            write_document_json(document)
            if args.format == "json"
            else document_to_markdown(document)
        )
        if args.output:
            with open(args.output, "w") as f:
                f.write(out)
            logger.info("wrote %s -> %s", args.format, args.output)
        else:
            print(out)
        return 0

    # summary
    lines = [
        f"=== {document.metadata.file_name} ===",
        f"title:  {document.metadata.title}",
        f"author: {document.metadata.author}",
        f"pages:  {document.number_of_pages}",
    ]
    for page in document.pages:
        kids = getattr(page, "_kids", [])
        preview = " ".join(getattr(o, "value", "") for o in kids)[:80].replace("\n", " ")
        lines.append(
            f"  page {page.page_number + 1}: "
            f"{len(kids)} objects, {len(page.text_chunks)} text chunks, "
            f"{page.width:.0f}x{page.height:.0f}pt | {preview!r}"
        )
    text = "\n".join(lines)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
