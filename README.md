
Document intelligence system that extracts data from PDFs (pharmaceutical), parse for desired fields, and flags compliance risks

## SDF Extraction Pipeline

Current: A lightweight, no-OCR pipeline that extracts key fields from pharmaceutical
**Site Documentation File (SDF)** PDFs — and similar document types — using
pdfplumber, regex, and spatial heuristics.

Future: OCR included
 (See Architecture Flow doc for system mapping)

---

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | **Entry point.** Run this. Handles args, dependency setup, batch processing, CSV reports. |
| `extractor.py` | **Core engine.** `PageExtractor` class + `extract_pdf()`. Import in notebooks or other scripts. |
| `field_dict.json` | **Synonym dictionary.** The only file you normally need to edit to support new documents. |

---

## Quick Start (Google Colab)

### 1 — Clone the repo

```python
!git clone https://github.com/nessawessa/pharmaDoc_intelSys.git
%cd sdf-pipeline
```

### 2 — Upload your PDF(s)

```python
from google.colab import files
uploaded = files.upload()   # files land in /content/
```

Or use the Colab file-browser sidebar.

### 3 — Run

```python
# Single PDF
!python pipeline.py /content/my_document.pdf

# Whole folder
!python pipeline.py /content/pdfs/

# Skip annotation images (faster)
!python pipeline.py /content/pdfs/ --no-annotate

# Higher-resolution images
!python pipeline.py my_document.pdf --dpi 200
```

Dependencies (`pdfplumber`, `opencv`, `pandas`, `poppler-utils`) are installed
automatically on the first run.

### 4 — View results

```python
import pandas as pd

results = pd.read_csv("output/results.csv")
display(results)

summary = pd.read_csv("output/summary.csv", index_col=0)
display(summary)
```

### 5 — Download outputs

```python
from google.colab import files
import zipfile
from pathlib import Path

with zipfile.ZipFile("sdf_output.zip", "w") as z:
    for f in Path("output").glob("*"):
        z.write(f)
files.download("sdf_output.zip")
```

---

All date values are normalised to **ISO-8601** (`YYYY-MM-DD`).


## Outputs

```
output/
  results.csv           -- every extracted field, with bbox coords + confidence
  summary.csv           -- pivot: one row per PDF, one column per field
  mydoc_page01.png      -- annotated page images with coloured bounding boxes
  mydoc_page02.png
  ...
```

### results.csv columns

| Column | Description |
|---|---|
| `source` | PDF filename |
| `page` | Page number (1-based) |
| `field` | Canonical field name |
| `value` | Extracted value (dates in ISO-8601) |
| `bbox_x0/top/x1/bottom` | Bounding box in PDF user-space units |
| `method` | Which strategy produced this result |
| `confidence` | `high` / `medium` / `low` |

---

## Extending for New Document Types

### Add label synonyms to field_dict.json

```json
{
  "lot_number": [
    "batch ref", "batch reference", "material number"
  ],
  "manufacturing_date": [
    "date of manufacture", "fill date", "production run date"
  ]
}
```

Built-in synonyms are always merged in — only add **new** phrasings.

### Add a brand-new field

```json
{
  "country_of_origin": [
    "country of origin", "country", "made in", "coo"
  ]
}
```

No code changes needed. The new field is automatically extracted wherever
its labels appear in any document.

### Add a custom extraction strategy

Open `extractor.py`, add a `_my_strategy()` method to `PageExtractor`,
and call it from `extract_all()`.  All built-in strategies follow the same
`self._add(field, value, bbox, method)` pattern.

---

## Confidence Levels

| Level | Meaning |
|---|---|
| `high` | Found via explicit label match or well-defined pattern |
| `medium` | Found via spatial heuristic (address block, top-right name) |
| `low` | Found via fallback (first heading, standalone date) |

Filter low-confidence results in pandas:

```python
high_confidence = results[results["confidence"] == "high"]
```

---

## Limitations

- **No OCR** — text must exist as a selectable layer in the PDF.
  Image-stamped overlays (e.g. DocuSign JPEG stamps) cannot be extracted
  without adding Tesseract + pytesseract.
- **Table cells** — label/value pairs inside table cells are extracted only
  when they share the same horizontal line. Vertically stacked tables may
  need a dedicated strategy.
- **Requires Python ≥ 3.9.**
