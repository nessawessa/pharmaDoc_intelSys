#!/usr/bin/env python3
# pipeline.py -- SDF extraction pipeline entry point
# Usage: !python pipeline.py <pdf_or_folder> [--dict field_dict.json] [--out output] [--dpi 150] [--no-annotate] [--quiet]

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _ensure_dependencies():
    if subprocess.run(["which", "pdftoppm"], capture_output=True).returncode != 0:
        print("[setup] installing poppler-utils ...")
        subprocess.run(["apt-get", "install", "-y", "-q", "poppler-utils"], check=True)

    for pkg, module in [
        ("pdfplumber",             "pdfplumber"),
        ("python-dateutil",        "dateutil"),
        ("opencv-python-headless", "cv2"),
        ("pandas",                 "pandas"),
    ]:
        if importlib.util.find_spec(module) is None:
            print(f"[setup] installing {pkg} ...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "--break-system-packages", pkg],
                check=True,
            )


def main():
    ap = argparse.ArgumentParser(description="SDF field extraction pipeline")
    ap.add_argument("input", help="PDF file or directory of PDFs")
    ap.add_argument("--dict", default=None, metavar="PATH",
                    help="Path to field_dict.json (auto-detected if omitted)")
    ap.add_argument("--out", default="output", metavar="DIR",
                    help="Output directory (default: ./output)")
    ap.add_argument("--dpi", type=int, default=150,
                    help="Render DPI for annotated images (default: 150)")
    ap.add_argument("--no-annotate", action="store_true",
                    help="Skip writing annotated images")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-field output")
    args = ap.parse_args()

    _ensure_dependencies()

    import pandas as pd
    from extractor import extract_pdf

    # Locate input PDFs
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

    # Resolve field dictionary
    dict_path = args.dict
    if dict_path is None:
        for candidate in [
            Path(__file__).parent / "field_dict.json",
            Path("field_dict.json"),
        ]:
            if candidate.exists():
                dict_path = str(candidate)
                break

    output_dir = Path(args.out)
    all_frames: list[pd.DataFrame] = []

    sep = "=" * 60
    print(f"\n{sep}")
    print(" SDF Extraction Pipeline")
    print(sep)
    print(f" PDFs found   : {len(pdf_files)}")
    print(f" Field dict   : {dict_path or '(built-ins only)'}")
    print(f" Output dir   : {output_dir}/")
    print(f" Annotate     : {not args.no_annotate}  (DPI={args.dpi})")
    print(f"{sep}\n")

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

    if not all_frames:
        print("No extractions produced.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    results_path = output_dir / "results.csv"
    combined.to_csv(results_path, index=False)

    summary_long = (
        combined.groupby(["source", "field"])["value"]
        .apply(lambda s: " | ".join(s.unique()))
        .reset_index()
        .rename(columns={"value": "values_found"})
    )
    try:
        pivot = summary_long.pivot(index="source", columns="field", values="values_found")
        summary_path = output_dir / "summary.csv"
        pivot.to_csv(summary_path)
    except Exception:
        pivot = None
        summary_path = output_dir / "summary_long.csv"
        summary_long.to_csv(summary_path, index=False)

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
        print(pivot.fillna("-").to_string())
        print()


if __name__ == "__main__":
    main()
