# ODL-PDF-PY

A native **Python** PDF → Markdown / JSON / Tagged-PDF extraction engine. It turns
born-digital PDFs into clean, structured output — preserving reading order, headings, and
tables — for RAG pipelines, search indexing, and document understanding.

It is a clean-room reimplementation of the layout and semantic algorithms behind
[opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf), with **no
JVM** at runtime: the byte layer is pure-Python ([pypdf](https://github.com/py-pdf/pypdf) +
[pikepdf](https://github.com/pikepdf/pikepdf)), so it is clone-and-go and easy to embed.

## Why this port

- **Pure-wheel, clone-and-go.** Core extraction needs no native binary and no JVM — just
  `pip install`. Trivial to embed in a service or notebook.
- **Structured, not just text.** Emits a typed document model → Markdown, an
  oracle-shaped JSON tree, or a tagged (accessible) PDF — not a flat text dump.
- **Permissive.** MIT-licensed, clean-room: no third-party extraction source was copied.

## Features

- **Reading order** — XY-Cut++ recursive cut (resolves multi-column layouts into human order).
- **Heading recovery** — font-size hierarchy (H1–H6 with level normalization) **plus** outline
  section numbers (`6`, `6.1`, `6.1.1`), run-in headings, and alignment / size-hierarchy signals
  for documents that number sections at body-font size.
- **Tables** — bordered (ruling-line) detection with a frame-rejection guard so page borders and
  call-out boxes aren't mistaken for data tables; opt-in borderless/cluster detection.
- **Lists** — label-based ordered/unordered detection.
- **Outputs** — Markdown, JSON (structured document tree), and Tagged-PDF (PDF/UA structure tree).
- **Optional hybrid AI mode** — triage routes only hard pages (CID-failure, vector-table,
  large-image) to a pluggable vision backend; the local engine stays primary. Backends are
  pluggable (a deterministic mock for tests, plus an AWS Bedrock vision adapter). Off by default.

## How it compares

On complex pages (data tables, multi-column layouts, figures), purpose-built layout engines
recover **document structure** — the heading and table markers that matter for RAG chunking —
more reliably than general-purpose vision models, while running on the order of **a thousand
times faster** and at **zero marginal cost** (no per-page inference). This engine was built and
tuned against that bar.

> Benchmarks are workload-dependent; run it on your own corpus to see how it does.

## Install

```bash
pip install -e .
# or, with uv:
uv sync

# optional hybrid AI mode (page rendering + AWS Bedrock vision):
pip install -e '.[hybrid]'
```

Requires Python 3.11+.

## Usage

```bash
# Markdown to stdout:
python -m odl_pdf document.pdf -f markdown

# Structured JSON:
python -m odl_pdf document.pdf -f json
```

```python
# Library:
from odl_pdf.pipeline import extract
from odl_pdf.output.markdown_writer import document_to_markdown

doc = extract("document.pdf")
print(document_to_markdown(doc))
```

## Package layout

| Path | Role |
|------|------|
| `src/odl_pdf/parser/` | PDF → typed chunk model (pypdf + pikepdf) |
| `src/odl_pdf/entities/` | Geometry + semantic model (BoundingBox, TextChunk, tables, lists) |
| `src/odl_pdf/processors/` | Reading order (XY-Cut++), grouping, heading detection, tables, lists |
| `src/odl_pdf/output/` | Markdown + JSON writers |
| `src/odl_pdf/tagging/` | Tagged-PDF (PDF/UA structure tree) |
| `src/odl_pdf/hybrid/` | Optional AI triage + pluggable vision backends |
| `src/odl_pdf/pipeline.py` | End-to-end extract() |
| `src/odl_pdf/__main__.py` | `python -m odl_pdf` CLI |

## Tests

```bash
pytest -q
```

## License

[MIT](LICENSE).

Clean-room implementation — no third-party extraction source was copied; behavior was
reconstructed from public algorithm descriptions. The upstream
[opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf) project is
Apache-2.0.
