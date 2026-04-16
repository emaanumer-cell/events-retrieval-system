#!/usr/bin/env python
"""
Build and cache indexes for the Event Retrieval System.

This script should be run ONCE to:
1. Parse the tracking plan Excel file
2. Embed all events with Cohere
3. Upload embeddings to Pinecone
4. Save index information locally

After this runs, the web server can start quickly without rebuilding indexes.

Usage:
    python build_indexes.py
"""
from pathlib import Path

from src.indexer import build_indexes
from src.parser import parse_events


def find_input_file() -> Path:
    """Resolve the input xlsx file path."""
    data_dir = Path("data")
    if data_dir.exists():
        xlsx_files = list(data_dir.glob("*.xlsx"))
        if len(xlsx_files) == 1:
            return xlsx_files[0]
        if len(xlsx_files) > 1:
            print("Multiple xlsx files found in data/:")
            for i, f in enumerate(xlsx_files, 1):
                print(f"  {i}. {f.name}")
            choice = input("Select file number: ").strip()
            return xlsx_files[int(choice) - 1]

    raise FileNotFoundError(
        "No xlsx files found in data/ directory. Add your tracking plan file there."
    )


def main() -> None:
    print("\n" + "=" * 70)
    print("  Event Retrieval System - Index Builder")
    print("=" * 70)
    print("\nThis will parse your tracking plan, embed events, and")
    print("upload them to Pinecone. This process may take a few minutes.\n")

    # Find input file
    xlsx_path = find_input_file()
    print(f"Loading tracking plan: {xlsx_path}")

    # Parse events
    all_events = parse_events(xlsx_path)
    print(f"Parsed {len(all_events)} events from tracking plan.")

    # Build indexes
    print("\nBuilding indexes...")
    print("  • Building BM25 index (in-memory)")
    print("  • Embedding events with Cohere")
    print("  • Uploading to Pinecone")
    print("  (This may take 1-2 minutes)\n")

    co, pinecone_idx, bm25 = build_indexes(all_events)

    print("\n" + "=" * 70)
    print("  SUCCESS!")
    print("=" * 70)
    print("\nIndexes are ready. You can now run the web server:")

if __name__ == "__main__":
    main()
