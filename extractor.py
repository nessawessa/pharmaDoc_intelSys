from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pdfplumber
from dateutil import parser as dateutil_parser


# Load pharma–field–dict 

def load_synonyms(json_path):
    if not json_path or not Path(json_path).exists():
        raise FileNotFoundError(
            "field_dict.json not found. "
            "Make sure it is in the same folder as pipeline.py."
        )
    with open(json_path) as f:
        data = json.load(f)
    return {
        k: [s.lower() for s in v]
        for k, v in data.items()
        if not k.startswith("_")
    }

# Annotation colours (BGR).  Unknown fields get green.
FIELD_COLOURS: dict[str, tuple[int, int, int]] = {
    "effective_date":       (0,   200,   0),
    "manufacturing_date":   (255, 100,   0),
    "vendor_name":          (0,   140, 255),
    "document_type":        (180,   0, 255),
    "revision_number":      (0,   200, 200),
    "docusign_timestamp":   (50,   50, 220),
    "company_address":      (200, 200,   0),
    "lot_number":           (255,   0, 128),
    "product_name":         (128,   0, 255),
    "expiration_date":      (0,   100, 200),
}


# Date helpers

_DATE_PATS = [
    re.compile(r"\b(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b"),
    re.compile(r"\b\d{1,2}[-/]\w{3,9}[-/]\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+\w{3,9},?\s*\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\w{3,9}\s+\d{1,2},?\s*\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}\b"),
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
]


def find_dates(text: str) -> list[str]:
    found = []
    for pat in _DATE_PATS:
        for m in pat.finditer(text):
            found.append(m.group())
    return list(dict.fromkeys(found))


def normalise_date(raw: str) -> str:
    raw = raw.strip()
    m = re.fullmatch(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    try:
        return dateutil_parser.parse(raw, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return raw

# Bounding-box + line helpers

def merge_boxes(boxes: list[tuple]) -> tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def word_box(w: dict) -> tuple[float, float, float, float]:
    return w["x0"], w["top"], w["x1"], w["bottom"]


def group_into_lines(words: list[dict], y_tol: float = 5.0) -> list[list[dict]]:
    if not words:
        return []
    sw = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = [[sw[0]]]
    for w in sw[1:]:
        if abs(w["top"] - lines[-1][-1]["top"]) <= y_tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda w: w["x0"])
    return lines


def line_text(line: list[dict]) -> str:
    return " ".join(w["text"] for w in line)


# PageExtractor
class PageExtractor:
    """
    Extract target fields from a single pdfplumber page using three
    complementary strategies:
      1. Label-scan  — synonym lookup, value follows the label
      2. Patterns    — regex for revision numbers, date stand-alones
      3. Spatial     — address blocks, document headings, top-right names
    """

    DATE_FIELDS = frozenset({
        "effective_date", "manufacturing_date",
        "docusign_timestamp", "expiration_date",
    })

    def __init__(self, page, page_num: int, synonyms: dict[str, list[str]]):
        self.page = page
        self.page_num = page_num
        self.synonyms = synonyms
        self.words: list[dict] = page.extract_words(
            x_tolerance=5, y_tolerance=3, keep_blank_chars=False
        )
        self.lines: list[list[dict]] = group_into_lines(self.words, y_tol=5)
        self._hits: dict[str, list[dict]] = defaultdict(list)

    def extract_all(self) -> list[dict]:
        self._label_scan()
        self._revision_pattern()
        self._address_block()
        self._document_heading()
        self._vendor_from_issuer()
        self._standalone_date()
        return self._compile()

    # --- Strategy 1: label scan -------------------------------------------

    def _label_scan(self):
        SKIP = {"revision_number", "company_address", "document_type", "vendor_name"}
        n = len(self.lines)
        for i, line in enumerate(self.lines):
            text_lower = line_text(line).lower()
            for field, syns in self.synonyms.items():
                if field in SKIP:
                    continue
                matched = next(
                    (s for s in sorted(syns, key=len, reverse=True) if s in text_lower),
                    None,
                )
                if not matched:
                    continue
                value_words, bbox = self._value_after_label(line, matched, i, n)
                if not value_words:
                    continue
                raw = self._clean(" ".join(w["text"] for w in value_words))
                if not raw:
                    continue
                if field in self.DATE_FIELDS:
                    hits = find_dates(raw)
                    if hits:
                        raw = normalise_date(hits[0])

                # Guard: reject obviously bad values
                if self._is_junk_value(field, raw):
                    continue

                self._add(field, raw, bbox, "label_scan")

    def _value_after_label(self, line, label, line_idx, n_lines):
        full = line_text(line)
        label_end = full.lower().find(label) + len(label)
        after = full[label_end:].lstrip(": ").strip()
        if after:
            n_lw = self._count_label_words(line, label)
            vw = self._trim_at_next_label(line[n_lw:])
            if vw:
                return vw, merge_boxes([word_box(w) for w in vw])
        else:
            if line_idx + 1 < n_lines:
                nxt = self.lines[line_idx + 1]
                if not line_text(nxt).strip().endswith(":"):
                    return nxt, merge_boxes([word_box(w) for w in nxt])
        return [], ()

    def _count_label_words(self, line, label):
        acc = ""
        for i, w in enumerate(line):
            acc = (acc + " " + w["text"]).strip().lower()
            if label in acc:
                return i + 1
        return 0

    # --- Strategy 2: revision-number pattern ------------------------------

    def _revision_pattern(self):
        PAT = re.compile(r"\bRev\.?\s*([A-Z]{1,3}|\d{1,4})\b", re.IGNORECASE)
        for line in self.lines:
            m = PAT.search(line_text(line))
            if not m:
                continue
            value = m.group(0).strip()
            rev_words = []
            for idx, w in enumerate(line):
                if re.search(r"\bRev\.?\b", w["text"], re.I):
                    rev_words.append(w)
                    if idx + 1 < len(line):
                        rev_words.append(line[idx + 1])
                    break
            bbox = (merge_boxes([word_box(w) for w in rev_words])
                    if rev_words else merge_boxes([word_box(w) for w in line]))
            self._add("revision_number", value, bbox, "rev_pattern")
            break

    # --- Strategy 3: address block ----------------------------------------

    def _address_block(self):
        h, pw = self.page.height, self.page.width
        ADDR = re.compile(
            r"\b(\d{5}|Way|Street|St\.|Ave|Blvd|Drive|Dr\.|Road|Rd\.|"
            r"Lane|Ln\.|United|States|Walkup|Results|"
            r"MA|CA|NY|TX|PA|NJ|OH|IL|GA|NC|VA|UK|Ltd|GmbH)\b",
            re.IGNORECASE,
        )
        NON = re.compile(r"(\.com|\.net|\.org|www\.|http|@)", re.IGNORECASE)
        candidate_lines, has_real = [], False
        for line in self.lines:
            if not line:
                continue
            top = line[0]["top"]
            avg_x = sum(w["x0"] for w in line) / len(line)
            text = line_text(line)
            if top < h * 0.25 and avg_x > pw * 0.55 and not NON.search(text):
                candidate_lines.append(line)
                if ADDR.search(text):
                    has_real = True
        if len(candidate_lines) >= 2 and has_real:
            all_words = [w for ln in candidate_lines for w in ln]
            value = " | ".join(line_text(ln) for ln in candidate_lines)
            self._add("company_address", value,
                      merge_boxes([word_box(w) for w in all_words]), "address_block")

    # --- Strategy 4: document heading -------------------------------------

    def _document_heading(self):
        PATS = [
            re.compile(r"^Re\s*:\s*(.+)$", re.IGNORECASE),
            re.compile(r"\b(Certificate\s+of\s+\w[\w\s]{0,30})\b", re.IGNORECASE),
            re.compile(r"\b(Report\s+of\s+\w[\w\s]{0,30})\b", re.IGNORECASE),
            re.compile(r"\b(Statement\s+of\s+\w[\w\s]{0,30})\b", re.IGNORECASE),
        ]
        BOILERPLATE = re.compile(
            r"(cytiva|page\s*\d|\.com|issued by|this product|this document)",
            re.IGNORECASE,
        )
        for line in self.lines:
            text = line_text(line)
            for pat in PATS:
                m = pat.search(text)
                if m:
                    val = (m.group(1) if m.lastindex else m.group(0)).strip()
                    val = re.sub(r"([a-z])([A-Z])", r"\1 \2", val)
                    val = re.sub(r"TM\s*", "™ ", val).strip()
                    self._add("document_type", val,
                              merge_boxes([word_box(w) for w in line]), "heading_pattern")
                    return
        zone_y = self.page.height * 0.30
        for line in self.lines:
            if not line or line[0]["top"] > zone_y:
                continue
            text = line_text(line)
            if len(text.split()) >= 2 and not BOILERPLATE.search(text):
                self._add("document_type", text.strip(),
                          merge_boxes([word_box(w) for w in line]), "first_heading")
                return

    # --- Strategy 5: vendor from issuer line ------------------------------

    def _vendor_from_issuer(self):
        ISSUER = re.compile(
            r"issued\s+by\s+(.+?)(?:\s+quality|\s+assurance|$)", re.IGNORECASE
        )
        for line in self.lines:
            m = ISSUER.search(line_text(line))
            if m:
                val = re.split(r"\s+(Quality|Assurance)", m.group(1), flags=re.I)[0].strip()
                self._add("vendor_name", val,
                          merge_boxes([word_box(w) for w in line]), "issuer_line")
                return
        pw = self.page.width
        for line in self.lines[:4]:
            if not line:
                continue
            avg_x = sum(w["x0"] for w in line) / len(line)
            text = line_text(line)
            if avg_x > pw * 0.7 and 1 <= len(line) <= 3 and re.search(r"[A-Z][a-z]", text):
                self._add("vendor_name", text.strip(),
                          merge_boxes([word_box(w) for w in line]), "top_right_name")
                return

    # --- Strategy 6: standalone date fallback -----------------------------

    def _standalone_date(self):
        if self._hits.get("effective_date"):
            return
        SKIP = re.compile(
            r"manufactur|expir|lot|product|article|batch", re.IGNORECASE
        )
        zone_y = self.page.height * 0.30
        for line in self.lines:
            if not line or line[0]["top"] > zone_y:
                continue
            text = line_text(line)
            if SKIP.search(text):
                continue
            dates = find_dates(text)
            if dates:
                self._add("effective_date", normalise_date(dates[0]),
                          merge_boxes([word_box(w) for w in line]), "standalone_date")
                return

    # --- Internals --------------------------------------------------------

    def _add(self, field: str, value: str, bbox: tuple, method: str):
        if not value:
            return
        if value not in {r["value"] for r in self._hits[field]}:
            self._hits[field].append({"value": value, "bbox": bbox, "method": method})

    @staticmethod
    def _is_junk_value(field: str, value: str) -> bool:
        """
        Return True if *value* looks like boilerplate or table-header text
        rather than a genuine field value.
        """
        # Too many words for a typical atomic value (except address / description)
        word_count = len(value.split())
        if field not in ("company_address", "product_name", "document_type"):
            if word_count > 6:
                return True
        # For product_name specifically, reject table headers and body sentences
        if field == "product_name":
            JUNK_PRODUCT = re.compile(
                r"(operating temperature|part number|this product|is manufactured"
                r"|compliance|quality management|release criteria"
                r"|product manager|product release|manager|criteria)",
                re.IGNORECASE,
            )
            if JUNK_PRODUCT.search(value):
                return True
            # Single generic words are not valid product names
            if len(value.split()) == 1 and value.istitle() and len(value) < 12:
                return True
        return False

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^[:\s,;]+", "", text)
        text = re.sub(r"[:\s,;]+$", "", text)
        return text.strip()

    @staticmethod
    def _trim_at_next_label(value_words: list[dict]) -> list[dict]:
        """
        Stop before a secondary label on the same line.

        Detects colon-terminated title-case words (e.g. "Number:") and
        removes them and any preceding title-case run that belongs to the
        label (e.g. "Product Article Number:").
        """
        colon_idx = None
        for i, w in enumerate(value_words):
            t = w["text"]
            if t.endswith(":") and len(t) >= 3 and t[0].isupper():
                colon_idx = i
                break
        if colon_idx is None:
            return value_words
        # Walk backwards through consecutive title-case words
        start = colon_idx
        while start > 0 and value_words[start - 1]["text"][0].isupper():
            start -= 1
        return value_words[:start]

    def _compile(self) -> list[dict]:
        PRIORITY = {
            "label_scan": 0, "issuer_line": 1, "address_block": 1,
            "rev_pattern": 1, "heading_pattern": 1,
            "top_right_name": 2, "first_heading": 3, "standalone_date": 4,
        }
        records = []
        for field, hits in self._hits.items():
            best = sorted(hits, key=lambda h: PRIORITY.get(h["method"], 99))[0]
            records.append({
                "page":        self.page_num,
                "field":       field,
                "value":       best["value"],
                "bbox_x0":     round(best["bbox"][0], 2),
                "bbox_top":    round(best["bbox"][1], 2),
                "bbox_x1":     round(best["bbox"][2], 2),
                "bbox_bottom": round(best["bbox"][3], 2),
                "method":      best["method"],
                "confidence":  _method_confidence(best["method"]),
            })
        return records


def _method_confidence(method: str) -> str:
    return {
        "label_scan": "high", "issuer_line": "high",
        "rev_pattern": "high", "heading_pattern": "high",
        "address_block": "medium", "top_right_name": "medium",
        "first_heading": "low", "standalone_date": "low",
    }.get(method, "low")


# Rendering + annotation

def render_page(pdf_path: str | Path, page_num: int, dpi: int = 150) -> np.ndarray:
    """Render one page to a BGR NumPy image via pdftoppm (poppler-utils)."""
    prefix = f"/tmp/_sdf_{Path(pdf_path).stem}_p{page_num}"
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", str(dpi),
         "-f", str(page_num), "-l", str(page_num),
         str(pdf_path), prefix],
        check=True, capture_output=True,
    )
    files = sorted(glob.glob(f"{prefix}-*.jpg") + glob.glob(f"{prefix}-*.jpeg"))
    if not files:
        raise FileNotFoundError(f"pdftoppm produced no output for page {page_num}")
    img = cv2.imread(files[0])
    os.unlink(files[0])
    return img


def annotate_image(
    img: np.ndarray,
    records: list[dict],
    page_width: float,
    page_height: float,
) -> np.ndarray:
    """Draw coloured bounding boxes + field labels onto a copy of img."""
    out = img.copy()
    ih, iw = out.shape[:2]
    sx, sy = iw / page_width, ih / page_height

    for rec in records:
        colour = FIELD_COLOURS.get(rec["field"], (0, 255, 0))
        x0 = max(0, int(rec["bbox_x0"] * sx) - 3)
        y0 = max(0, int(rec["bbox_top"] * sy) - 3)
        x1 = min(iw - 1, int(rec["bbox_x1"] * sx) + 3)
        y1 = min(ih - 1, int(rec["bbox_bottom"] * sy) + 3)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)

        tag = f"{rec['field']}: {rec['value'][:45]}"
        font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        (tw, th), bl = cv2.getTextSize(tag, font, fscale, thick)
        ty = max(y0 - 4, th + 4)
        cv2.rectangle(out, (x0, ty - th - 2), (x0 + tw + 4, ty + bl), colour, -1)
        cv2.putText(out, tag, (x0 + 2, ty), font, fscale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return out


# Main extraction function

def extract_pdf(
    pdf_path: str | Path,
    *,
    dict_path: str | Path | None = None,
    dpi: int = 150,
    output_dir: str | Path = "output",
    annotate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract fields from every page of *pdf_path*.

    Parameters
    ----------
    pdf_path   : path to the input PDF
    dict_path  : path to field_dict.json (optional; built-ins always apply)
    dpi        : render resolution for annotated images
    output_dir : where to write CSV + annotated PNGs
    annotate   : whether to write annotated page images
    verbose    : print progress to stdout

    Returns
    -------
    pandas.DataFrame with columns:
      source, page, field, value, bbox_x0, bbox_top, bbox_x1,
      bbox_bottom, method, confidence
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synonyms = load_synonyms(dict_path)
    all_records: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        for idx, page in enumerate(pdf.pages):
            pnum = idx + 1
            if verbose:
                print(f"  [page {pnum}/{n}] extracting …", end=" ", flush=True)

            records = PageExtractor(page, pnum, synonyms).extract_all()
            for r in records:
                r["source"] = pdf_path.name
            all_records.extend(records)

            if verbose:
                print(f"{len(records)} field(s) found")
                for r in sorted(records, key=lambda x: x["field"]):
                    icon = {"high": "✓", "medium": "~", "low": "?"}.get(
                        r["confidence"], " "
                    )
                    print(f"    {icon} {r['field']:25s} = {r['value']!r}")

            if annotate:
                try:
                    img = render_page(pdf_path, pnum, dpi)
                    ann = annotate_image(img, records, page.width, page.height)
                    out_img = output_dir / f"{pdf_path.stem}_page{pnum:02d}.png"
                    cv2.imwrite(str(out_img), ann)
                    if verbose:
                        print(f"    → {out_img}")
                except Exception as exc:
                    if verbose:
                        print(f"    [warn] annotation failed: {exc}")

    df = pd.DataFrame(all_records)
    if not df.empty:
        cols = ["source", "page", "field", "value",
                "bbox_x0", "bbox_top", "bbox_x1", "bbox_bottom",
                "method", "confidence"]
        df = df[[c for c in cols if c in df.columns]]
        df = df.sort_values(["source", "page", "field"]).reset_index(drop=True)

    return df
