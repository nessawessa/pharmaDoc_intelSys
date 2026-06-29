"""
ocr_processor.py
================
Scanned-PDF processing pipeline for the SDF extraction system.

Handles documents where the text layer is absent or unreliable by:
  1. Rendering each PDF page to a high-resolution image
  2. Pre-processing the image (grayscale, contrast, denoise, resize)
  3. Running Tesseract OCR with word-level bounding boxes + confidence
  4. Correcting common OCR errors via ocr_errors.json
  5. Extracting target fields using the same synonym dictionary as extractor.py
  6. Drawing green word-boxes (all OCR words) and red field-boxes (key fields)
  7. Flagging low-confidence words for optional human review (threshold: 85%)
  8. Exporting per-page results to JSON (values + locations) for downstream AI

All pre-existing functions in extractor.py are preserved and unchanged.
This module calls load_synonyms() and the field-matching logic from extractor.py.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import glob
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None  # handled gracefully at runtime

# Reuse synonym loading and field-matching from the existing pipeline
from extractor import (
    load_synonyms,
    find_dates,
    normalise_date,
    group_into_lines,
    FIELD_COLOURS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 85   # words below this % are flagged for review
OCR_CONFIG = "--oem 3 --psm 6 -l eng"   # OEM 3 = LSTM; PSM 6 = uniform block
SCALE_FACTOR = 2.0          # upscale factor before OCR (200 %)

# Annotation colours (BGR)
WORD_BOX_COLOUR  = (0, 200, 0)      # green  — all OCR word boxes
FIELD_BOX_COLOUR = (0, 0, 220)      # red    — extracted key fields
LOW_CONF_COLOUR  = (0, 140, 255)    # orange — low-confidence words
REVIEW_COLOUR    = (255, 0, 255)    # magenta — review-flagged regions


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_tesseract():
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract"
        )
    result = subprocess.run(["which", "tesseract"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "tesseract binary not found. Run: apt-get install -y tesseract-ocr"
        )


# ---------------------------------------------------------------------------
# OCR error correction
# ---------------------------------------------------------------------------

def load_ocr_corrections(json_path: str | Path | None = None) -> dict[str, str]:
    """
    Load word-level OCR corrections from ocr_errors.json.
    Returns a flat dict of {bad_string: correct_string}, all lowercase keys.
    Falls back to an empty dict if the file is missing.
    """
    corrections: dict[str, str] = {}

    if json_path is None:
        candidates = [
            Path(__file__).parent / "ocr_errors.json",
            Path("ocr_errors.json"),
        ]
        for c in candidates:
            if c.exists():
                json_path = c
                break

    if json_path and Path(json_path).exists():
        with open(json_path) as fh:
            data = json.load(fh)
        word_corrections = data.get("word_corrections", {})
        for bad, good in word_corrections.items():
            if isinstance(good, str):
                corrections[bad.lower()] = good.lower()

    # Sort by length descending so longer phrases match before substrings
    return dict(sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True))


def apply_ocr_corrections(text: str, corrections: dict[str, str]) -> str:
    """
    Apply word-correction dictionary to a text string.
    Matches are case-insensitive; replacement preserves the correction's case.
    Also strips duplicate whitespace and isolated line-break artefacts.
    """
    # Fix common structural artefacts first
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse 3+ blank lines → 2
    text = re.sub(r"[ \t]{2,}", " ", text)      # collapse repeated spaces
    text = re.sub(r"^\s*[-_|]{2,}\s*$", "", text, flags=re.MULTILINE)  # divider lines

    # Word-level corrections (whole-word match)
    for bad, good in corrections.items():
        text = re.sub(
            r"\b" + re.escape(bad) + r"\b",
            good,
            text,
            flags=re.IGNORECASE,
        )

    # Character-level fixes that are safe globally
    # Zero used as letter O between alphabetic chars: "C0MPANY" → "COMPANY"
    text = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "O", text)
    # Pipe char used as capital I: "| ssued" → "Issued"
    text = re.sub(r"\|\s*(?=[a-z])", "I", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Prepare a raw page image for Tesseract OCR.

    Pipeline
    --------
    1. Grayscale conversion
    2. Upscale 200 % (Lanczos interpolation — best for text)
    3. Bilateral filter  (reduce noise, preserve edges)
    4. CLAHE contrast enhancement
    5. Adaptive threshold  (handles uneven illumination in scans)
    6. Morphological opening  (remove tiny specks)

    Returns the processed single-channel binary image.
    """
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Upscale 200 %
    h, w = gray.shape[:2]
    scaled = cv2.resize(
        gray,
        (int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # 3. Bilateral filter — reduces scan noise while keeping text edges sharp
    denoised = cv2.bilateralFilter(scaled, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. CLAHE — adaptive contrast enhancement (better than global equalisation
    #    for scans with uneven brightness across the page)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Adaptive threshold — converts to clean black-on-white binary image
    #    Gaussian weighting handles shadow gradients across the scan
    binary = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

    # 6. Morphological opening — remove isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


def detect_skew_and_deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect page skew using Hough line transform and rotate to correct it.
    Returns (deskewed_image, angle_degrees).
    Only corrects if skew > 0.3° and < 15° to avoid false corrections.
    """
    # Work on a binary image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None:
        return img, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only collect near-horizontal lines
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img, 0.0

    median_angle = float(np.median(angles))

    # Only correct meaningful skew
    if abs(median_angle) < 0.3 or abs(median_angle) > 15:
        return img, median_angle

    h, w = img.shape[:2]
    centre = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
    deskewed = cv2.warpAffine(
        img, rot_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed, median_angle


# ---------------------------------------------------------------------------
# OCR extraction
# ---------------------------------------------------------------------------

def run_ocr(processed_img: np.ndarray) -> dict:
    """
    Run Tesseract on a pre-processed image.
    Returns the full pytesseract data dict (word boxes + confidence scores).
    """
    _check_tesseract()
    pil_img = Image.fromarray(processed_img)
    data = pytesseract.image_to_data(
        pil_img,
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT,
    )
    return data


def parse_ocr_data(
    ocr_data: dict,
    corrections: dict[str, str],
    scale: float = SCALE_FACTOR,
) -> list[dict]:
    """
    Convert raw Tesseract output into a clean list of word records.

    Each record contains:
      text        : corrected word text
      raw_text    : original OCR text (before correction)
      conf        : confidence score (0-100)
      x0, top, x1, bottom : bounding box in ORIGINAL image coordinates
                            (divided back by scale factor)
      low_conf    : True if conf < CONFIDENCE_THRESHOLD
      needs_review: True if conf < CONFIDENCE_THRESHOLD and word is non-trivial
    """
    words = []
    n = len(ocr_data["text"])

    for i in range(n):
        raw = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])

        # Tesseract returns conf=-1 for non-word rows; skip empties too
        if conf < 0 or not raw:
            continue

        # Scale bounding box back to original image coordinates
        x0     = int(ocr_data["left"][i]   / scale)
        top    = int(ocr_data["top"][i]    / scale)
        width  = int(ocr_data["width"][i]  / scale)
        height = int(ocr_data["height"][i] / scale)
        x1     = x0 + width
        bottom = top + height

        # Apply OCR corrections
        corrected = apply_ocr_corrections(raw, corrections)

        words.append({
            "text":         corrected,
            "raw_text":     raw,
            "conf":         conf,
            "x0":           x0,
            "top":          top,
            "x1":           x1,
            "bottom":       bottom,
            "low_conf":     conf < CONFIDENCE_THRESHOLD,
            "needs_review": conf < CONFIDENCE_THRESHOLD and len(raw) > 1,
        })

    return words


# ---------------------------------------------------------------------------
# Field extraction from OCR words
# ---------------------------------------------------------------------------

def extract_fields_from_ocr(
    words: list[dict],
    synonyms: dict[str, list[str]],
) -> list[dict]:
    """
    Apply the same label-scan strategy used in PageExtractor, but operating
    on OCR word records instead of pdfplumber words.

    Returns a list of field records matching the extractor.py schema:
      field, value, bbox_x0, bbox_top, bbox_x1, bbox_bottom,
      method, confidence, ocr_conf (average OCR confidence of value words)
    """
    DATE_FIELDS = frozenset({
        "effective_date", "manufacturing_date",
        "docusign_timestamp", "expiration_date",
    })
    SKIP = {"revision_number", "company_address", "document_type", "vendor_name"}

    # Convert word list to pdfplumber-compatible dicts for reuse
    compat_words = [
        {
            "text":   w["text"],
            "x0":     float(w["x0"]),
            "top":    float(w["top"]),
            "x1":     float(w["x1"]),
            "bottom": float(w["bottom"]),
        }
        for w in words
    ]

    lines = group_into_lines(compat_words, y_tol=12)  # larger tol for scans
    results: list[dict] = []
    found_fields: set[str] = set()

    def line_text(line):
        return " ".join(w["text"] for w in line)

    def avg_conf_for_words(value_words):
        matched = []
        for vw in value_words:
            for ow in words:
                if (ow["text"] == vw["text"]
                        and abs(ow["x0"] - vw["x0"]) < 5
                        and abs(ow["top"] - vw["top"]) < 5):
                    matched.append(ow["conf"])
                    break
        return round(sum(matched) / len(matched), 1) if matched else 0.0

    for i, line in enumerate(lines):
        text_lower = line_text(line).lower()

        for field, syns in synonyms.items():
            if field in SKIP or field in found_fields:
                continue

            matched_syn = next(
                (s for s in sorted(syns, key=len, reverse=True)
                 if s in text_lower),
                None,
            )
            if not matched_syn:
                continue

            # Find value words after the label
            label_end = text_lower.find(matched_syn) + len(matched_syn)
            after = line_text(line)[label_end:].lstrip(": ").strip()

            if after:
                # Count how many words belong to the label
                acc = ""
                n_label = 0
                for w in line:
                    acc = (acc + " " + w["text"]).strip().lower()
                    n_label += 1
                    if matched_syn in acc:
                        break
                value_words = line[n_label:]
            elif i + 1 < len(lines):
                next_line = lines[i + 1]
                if not line_text(next_line).strip().endswith(":"):
                    value_words = next_line
                else:
                    continue
            else:
                continue

            # Trim secondary labels on the same line
            value_words = _trim_secondary_label(value_words)
            if not value_words:
                continue

            raw = " ".join(w["text"] for w in value_words).strip().lstrip(":").strip()
            if not raw:
                continue

            if field in DATE_FIELDS:
                hits = find_dates(raw)
                if hits:
                    raw = normalise_date(hits[0])

            bbox = (
                min(w["x0"]    for w in value_words),
                min(w["top"]   for w in value_words),
                max(w["x1"]    for w in value_words),
                max(w["bottom"]for w in value_words),
            )
            ocr_conf = avg_conf_for_words(value_words)

            results.append({
                "field":       field,
                "value":       raw,
                "bbox_x0":     round(bbox[0], 2),
                "bbox_top":    round(bbox[1], 2),
                "bbox_x1":     round(bbox[2], 2),
                "bbox_bottom": round(bbox[3], 2),
                "method":      "ocr_label_scan",
                "confidence":  _ocr_conf_label(ocr_conf),
                "ocr_conf":    ocr_conf,
            })
            found_fields.add(field)

    # Revision-number pattern
    if "revision_number" not in found_fields:
        REV = re.compile(r"\bRev\.?\s*([A-Z]{1,3}|\d{1,4})\b", re.IGNORECASE)
        for line in lines:
            m = REV.search(line_text(line))
            if m:
                value = m.group(0).strip()
                rev_words = [w for w in line
                             if re.search(r"\bRev\.?\b", w["text"], re.I)]
                all_words_in_line = line
                bbox_words = rev_words if rev_words else all_words_in_line
                bbox = (
                    min(w["x0"]    for w in bbox_words),
                    min(w["top"]   for w in bbox_words),
                    max(w["x1"]    for w in bbox_words),
                    max(w["bottom"]for w in bbox_words),
                )
                results.append({
                    "field":       "revision_number",
                    "value":       value,
                    "bbox_x0":     round(bbox[0], 2),
                    "bbox_top":    round(bbox[1], 2),
                    "bbox_x1":     round(bbox[2], 2),
                    "bbox_bottom": round(bbox[3], 2),
                    "method":      "ocr_rev_pattern",
                    "confidence":  "high",
                    "ocr_conf":    100.0,
                })
                break

    return results


def _trim_secondary_label(value_words: list[dict]) -> list[dict]:
    colon_idx = None
    for i, w in enumerate(value_words):
        t = w["text"]
        if t.endswith(":") and len(t) >= 3 and t[0].isupper():
            colon_idx = i
            break
    if colon_idx is None:
        return value_words
    start = colon_idx
    while start > 0 and value_words[start - 1]["text"][0].isupper():
        start -= 1
    return value_words[:start]


def _ocr_conf_label(ocr_conf: float) -> str:
    if ocr_conf >= CONFIDENCE_THRESHOLD:
        return "high"
    elif ocr_conf >= 60:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_ocr_image(
    original_img: np.ndarray,
    words: list[dict],
    field_records: list[dict],
) -> np.ndarray:
    """
    Draw two layers of annotation on the original (non-preprocessed) image:

    Layer 1 — GREEN boxes around every OCR word
              (lets you see what Tesseract found and what it missed)
    Layer 2 — RED boxes around extracted key fields with field labels
    ORANGE   — low-confidence word boxes flagged for review
    """
    out = original_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Layer 1: all OCR word boxes ---
    for w in words:
        colour = LOW_CONF_COLOUR if w["low_conf"] else WORD_BOX_COLOUR
        cv2.rectangle(out,
                      (w["x0"], w["top"]),
                      (w["x1"], w["bottom"]),
                      colour, 1)
        # Show confidence score on low-confidence words
        if w["low_conf"]:
            cv2.putText(out, f"{w['conf']}%",
                        (w["x0"], max(w["top"] - 2, 10)),
                        font, 0.28, LOW_CONF_COLOUR, 1, cv2.LINE_AA)

    # --- Layer 2: key field boxes (red) with labels ---
    for rec in field_records:
        x0     = int(rec["bbox_x0"])
        y0     = int(rec["bbox_top"])
        x1     = int(rec["bbox_x1"])
        y1     = int(rec["bbox_bottom"])

        # Expand slightly for visibility
        x0, y0 = max(0, x0 - 4), max(0, y0 - 4)
        x1 = min(out.shape[1] - 1, x1 + 4)
        y1 = min(out.shape[0] - 1, y1 + 4)

        cv2.rectangle(out, (x0, y0), (x1, y1), FIELD_BOX_COLOUR, 2)

        tag = f"{rec['field']}: {rec['value'][:40]}  [{rec['ocr_conf']}%]"
        fscale, thick = 0.38, 1
        (tw, th), bl = cv2.getTextSize(tag, font, fscale, thick)
        ty = max(y0 - 4, th + 4)
        cv2.rectangle(out, (x0, ty - th - 2), (x0 + tw + 4, ty + bl),
                      FIELD_BOX_COLOUR, -1)
        cv2.putText(out, tag, (x0 + 2, ty), font, fscale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# JSON export for downstream AI / database storage
# ---------------------------------------------------------------------------

def export_fields_to_json(
    pdf_name: str,
    page_num: int,
    field_records: list[dict],
    words: list[dict],
    skew_angle: float,
    output_dir: Path,
) -> Path:
    """
    Export extracted fields and their locations to a structured JSON file.

    Schema is designed for:
      - Training future ML/inference models
      - Database ingestion (document intelligence pipelines)
      - Human review integration

    Output: output_dir/<pdf_stem>_page<N>_fields.json
    """
    review_words = [
        {
            "text":    w["text"],
            "raw_ocr": w["raw_text"],
            "conf":    w["conf"],
            "bbox":    {"x0": w["x0"], "top": w["top"],
                        "x1": w["x1"], "bottom": w["bottom"]},
        }
        for w in words if w["needs_review"]
    ]

    payload = {
        "meta": {
            "source_pdf":          pdf_name,
            "page":                page_num,
            "ocr_engine":          "tesseract",
            "ocr_config":          OCR_CONFIG,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "skew_angle_degrees":  round(skew_angle, 3),
            "scale_factor":        SCALE_FACTOR,
            "pipeline_version":    "2.0.0",
        },
        "fields": [
            {
                "field":      rec["field"],
                "value":      rec["value"],
                "method":     rec["method"],
                "confidence": rec["confidence"],
                "ocr_conf":   rec["ocr_conf"],
                "bbox": {
                    "x0":     rec["bbox_x0"],
                    "top":    rec["bbox_top"],
                    "x1":     rec["bbox_x1"],
                    "bottom": rec["bbox_bottom"],
                },
                "needs_human_review": rec["ocr_conf"] < CONFIDENCE_THRESHOLD,
            }
            for rec in field_records
        ],
        "human_review": {
            "required": len(review_words) > 0,
            "low_confidence_words": review_words,
            "note": (
                "Words below the confidence threshold are listed here. "
                "A human reviewer should verify these before downstream use."
                if review_words else "All words met the confidence threshold."
            ),
        },
    }

    stem = Path(pdf_name).stem
    out_path = output_dir / f"{stem}_page{page_num:02d}_fields.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Top-level: process one scanned PDF
# ---------------------------------------------------------------------------

def process_scanned_pdf(
    pdf_path: str | Path,
    *,
    dict_path: str | Path | None = None,
    ocr_errors_path: str | Path | None = None,
    dpi: int = 300,
    output_dir: str | Path = "output",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full OCR pipeline for a scanned (non-text-layer) PDF.

    Parameters
    ----------
    pdf_path        : path to the scanned PDF
    dict_path       : path to field_dict.json
    ocr_errors_path : path to ocr_errors.json (auto-detected if omitted)
    dpi             : render DPI — 300 recommended for scanned docs
    output_dir      : where to write annotated PNGs, field JSONs, and CSV
    verbose         : print progress

    Returns
    -------
    pd.DataFrame with same schema as extract_pdf() in extractor.py,
    plus an additional 'ocr_conf' column.
    """
    _check_tesseract()

    pdf_path   = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synonyms    = load_synonyms(dict_path)
    corrections = load_ocr_corrections(ocr_errors_path)

    all_records: list[dict] = []

    # Render all pages to images via pdftoppm
    prefix = f"/tmp/_ocr_{pdf_path.stem}"
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", str(dpi), str(pdf_path), prefix],
        check=True, capture_output=True,
    )
    page_images = sorted(
        glob.glob(f"{prefix}-*.jpg") + glob.glob(f"{prefix}-*.jpeg")
    )

    if not page_images:
        raise FileNotFoundError(
            f"pdftoppm produced no images for {pdf_path.name}"
        )

    for pnum, img_path in enumerate(page_images, 1):
        if verbose:
            print(f"  [OCR page {pnum}/{len(page_images)}]", end=" ", flush=True)

        # Load original image (kept for annotation)
        original = cv2.imread(img_path)
        os.unlink(img_path)

        # --- Pre-processing ---
        deskewed, skew_angle = detect_skew_and_deskew(original)
        if verbose and abs(skew_angle) > 0.3:
            print(f"skew corrected {skew_angle:.2f}°", end=" ", flush=True)

        processed = preprocess_image(deskewed)

        # --- OCR ---
        ocr_data = run_ocr(processed)
        words    = parse_ocr_data(ocr_data, corrections, scale=SCALE_FACTOR)

        # --- Field extraction ---
        field_records = extract_fields_from_ocr(words, synonyms)
        for r in field_records:
            r["source"] = pdf_path.name
            r["page"]   = pnum
        all_records.extend(field_records)

        if verbose:
            print(f"{len(field_records)} field(s) found")
            for r in sorted(field_records, key=lambda x: x["field"]):
                icon = {"high": "✓", "medium": "~", "low": "?"}.get(
                    r["confidence"], " "
                )
                print(f"    {icon} {r['field']:25s} = {r['value']!r}"
                      f"  [OCR conf: {r['ocr_conf']}%]")

            low_conf_count = sum(1 for w in words if w["needs_review"])
            if low_conf_count:
                print(f"    ⚠  {low_conf_count} word(s) below "
                      f"{CONFIDENCE_THRESHOLD}% confidence — flagged for review")

        # --- Annotated image ---
        annotated = annotate_ocr_image(original, words, field_records)
        ann_path  = output_dir / f"{pdf_path.stem}_page{pnum:02d}_ocr.png"
        cv2.imwrite(str(ann_path), annotated)

        # --- Preprocessed image (for debugging) ---
        pre_path = output_dir / f"{pdf_path.stem}_page{pnum:02d}_preprocessed.png"
        cv2.imwrite(str(pre_path), processed)

        # --- JSON export ---
        json_path = export_fields_to_json(
            pdf_path.name, pnum, field_records, words, skew_angle, output_dir
        )

        if verbose:
            print(f"    → {ann_path.name}  |  {json_path.name}")

    # Build DataFrame
    df = pd.DataFrame(all_records)
    if not df.empty:
        cols = ["source", "page", "field", "value",
                "bbox_x0", "bbox_top", "bbox_x1", "bbox_bottom",
                "method", "confidence", "ocr_conf"]
        df = df[[c for c in cols if c in df.columns]]
        df = df.sort_values(["source", "page", "field"]).reset_index(drop=True)

    return df
