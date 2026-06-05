# Single PDF
python pipeline.py my_document.pdf

  # Whole folder
python pipeline.py /content/pdfs/

  # With a custom field dictionary
python pipeline.py /content/pdfs/ --dict field_dict.json

  # Skip annotated images (faster)
python pipeline.py /content/pdfs/ --no-annotate

  # Higher resolution annotations
python pipeline.py my_document.pdf --dpi 200

Outputs  (written to ./output/ by default)
---------
  output/
    results.csv              -- all extractions, all PDFs, all pages
    summary.csv              -- one row per PDF: which fields were found
    <stem>_page01.png        -- annotated page images (one per page per PDF)
    ...

The pipeline installs its own dependencies on first run so it works
out-of-the-box in a fresh Google Colab runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency bootstrap  (safe to run repeatedly; skips if already installed)
# ---------------------------------------------------------------------------

def _ensure_dependencies():
    """Install Python packages and system tools that are missing."""
    # poppler-utils (provides pdftoppm for page rendering)
    if subprocess.run(["which", "pdftoppm"], capture_output=True).returncode != 0:
        print("[setup] installing poppler-utils …")
        subprocess.run(["apt-get", "install", "-y", "-q", "poppler-utils"], check=True)

    # Python packages
    for pkg, module in [
        ("pdfplumber",             "pdfplumber"),
        ("python-dateutil",        "dateutil"),
        ("opencv-python-headless", "cv2"),
        ("pandas",                 "pandas"),
    ]:
        if importlib.util.find_spec(module) is None:
            print(f"[setup] installing {pkg} …")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "--break-system-packages", pkg],
                check=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="SDF field extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "input",
        help="Path to a PDF file, or a directory containing PDF files.",
    )
    ap.add_argument(
        "--dict", default=None, metavar="PATH",
        help="Path to field_dict.json  (auto-detected if omitted)",
    )
    ap.add_argument(
        "--out", default="output", metavar="DIR",
        help="Output directory  (default: ./output)",
    )
    ap.add_argument(
        "--dpi", type=int, default=150,
        help="Render DPI for annotated images  (default: 150)",
    )
    ap.add_argument(
        "--no-annotate", action="store_true",
        help="Skip writing annotated images  (faster, less disk)",
    )
    ap.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-field output; only print totals",
    )
    args = ap.parse_args()

    _ensure_dependencies()

    # Import after deps are confirmed present
    import pandas as pd
    from extractor import extract_pdf

    # --- locate input PDFs -------------------------------------------------
    input_path = Path(args.input)
    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("**/*.pdf"))
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_files = [input_path]
    else:
        print(f"ERROR: {args.input!r} is not a PDF file or directory.", file=sys.stderr)
        sys.exit(1)

    if not pdf_files:
        print(f"ERROR: no PDF files found under {args.input!r}.", file=sys.stderr)
        sys.exit(1)

    # --- resolve field dictionary ------------------------------------------
    dict_path = args.dict
    if dict_path is None:
        for candidate in [
            Path(__file__).parent / "field_dict.json",
            Path("field_dict.json"),
        ]:
            if candidate.exists():
                dict_path = str(candidate)
                break

    # --- run extraction ----------------------------------------------------
    output_dir = Path(args.out)
    all_frames: list[pd.DataFrame] = []

    print(f"\n{'='*60}")
    print(f" SDF Extraction Pipeline")
    print(f"{'='*60}")
    print(f" PDFs found   : {len(pdf_files)}")
    print(f" Field dict   : {dict_path or '(built-ins only)'}")
    print(f" Output dir   : {output_dir}/")
    print(f" Annotate     : {not args.no_annotate}  (DPI={args.dpi})")
    print(f"{'='*60}\n")

    for i, pdf in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf.name}")
        try:
            df = extract_pdf(
                pdf,
                dict_path=dict_path,
                dpi=args.dpi,
                output_dir=output_dir,
                annotate=not args.no_annotate,
                verbose=not args.quiet,
            )
            if not df.empty:
                all_frames.append(df)
        except Exception as exc:
            print(f"  [ERROR] could not process {pdf.name}: {exc}")
        print()

    # --- write consolidated outputs ----------------------------------------
    if not all_frames:
        print("No extractions produced.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # results.csv — every extraction from every page
    results_path = output_dir / "results.csv"
    combined.to_csv(results_path, index=False)

    # summary.csv — pivot: one row per PDF, one column per field
    summary_long = (
        combined.groupby(["source", "field"])["value"]
        .apply(lambda s: " | ".join(s.unique()))
        .reset_index()
        .rename(columns={"value": "values_found"})
    )
    try:
        pivot = summary_long.pivot(
            index="source", columns="field", values="values_found"
        )
        summary_path = output_dir / "summary.csv"
        pivot.to_csv(summary_path)
    except Exception:
        summary_path = output_dir / "summary_long.csv"
        summary_long.to_csv(summary_path, index=False)
        pivot = None

    # --- final report ------------------------------------------------------
    sep = "=" * 60
    print(sep)
    print(" Extraction complete")
    print(sep)
    print(f" PDFs processed  : {combined['source'].nunique()}")
    print(f" Total fields    : {len(combined)}")
    print(f" Fields found    : {sorted(combined['field'].unique())}")
    print()
    print(" Confidence breakdown:")
    for conf, grp in combined.groupby("confidence"):
        pct = 100 * len(grp) / len(combined)
        print(f"   {conf:8s}: {len(grp):4d}  ({pct:.0f}%)")
    print()
    print(f" results.csv  -> {results_path}")
    print(f" summary.csv  -> {summary_path}")
    print(sep)

    if pivot is not None:
        print("\nSummary table:\n")
        print(pivot.fillna("—").to_string())
        print()


if __name__ == "__main__":
    main()
