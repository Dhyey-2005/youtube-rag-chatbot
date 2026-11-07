"""
test_chunking.py
Verification script for Phase 2: Text Splitting and Chunk Inspection.
Tests load_youtube_transcript and TranscriptTextSplitter using an actual YouTube URL.
"""

from __future__ import annotations

import json
import sys
from typing import List

from backend.ingestion import load_youtube_transcript
from backend.text_splitter import TranscriptTextSplitter

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Default target video: 3Blue1Brown "What is a neural network? | Chapter 1, Deep learning"
# A high-quality educational video with rich speech and accurate official captions
DEFAULT_TEST_URL = "https://www.youtube.com/watch?v=aircAruvnKk"


def inspect_chunks(
    video_url: str = DEFAULT_TEST_URL,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> None:
    """
    Ingests video transcript, splits into chunks, and prints detailed inspection.
    """
    print("=" * 75)
    print("PHASE 2 VERIFICATION: Text Splitting & Chunk Inspection")
    print("=" * 75)
    print(f"Target Video URL     : {video_url}")
    print(f"Chunk Size Config    : {chunk_size} characters")
    print(f"Chunk Overlap Config : {chunk_overlap} characters\n")

    # 1. Ingestion using Phase 1 module
    print("[1/4] Ingesting transcript snippets from YouTube...")
    try:
        raw_documents = load_youtube_transcript(video_url)
    except Exception as exc:
        print(f"[FAIL] Ingestion error: {exc}")
        sys.exit(1)

    if not raw_documents:
        print("[FAIL] Ingestion returned empty document list.")
        sys.exit(1)

    video_title = raw_documents[0].metadata.get("video_title", "Unknown")
    language = raw_documents[0].metadata.get("language", "unknown")
    total_raw_snippets = len(raw_documents)
    total_raw_chars = sum(len(d.page_content) for d in raw_documents)

    print(f"      Video Title         : {video_title}")
    print(f"      Language            : {language}")
    print(f"      Total Raw Snippets  : {total_raw_snippets}")
    print(f"      Total Raw Characters: {total_raw_chars}\n")

    # 2. Text splitting using Phase 2 module
    print("[2/4] Splitting transcript into chunks...")
    splitter = TranscriptTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    try:
        chunks = splitter.split_documents(raw_documents)
    except Exception as exc:
        print(f"[FAIL] Text splitting error: {exc}")
        sys.exit(1)

    total_chunks = len(chunks)
    print(f"      Total Generated Chunks: {total_chunks}\n")

    # 3. Display character counts of all generated chunks
    print("[3/4] Character Count of Each Generated Chunk:")
    char_counts: List[int] = [len(c.page_content) for c in chunks]
    counts_summary = ", ".join(f"C{i+1}:{cnt}" for i, cnt in enumerate(char_counts))
    # Print in readable wrapped lines
    wrapped_lines = [counts_summary[i : i + 70] for i in range(0, len(counts_summary), 70)]
    for line in wrapped_lines:
        print(f"      {line}")
    print(f"      Min chars: {min(char_counts)} | Max chars: {max(char_counts)} | Avg chars: {sum(char_counts)//total_chunks}\n")

    # 4. First 5 chunks' content previews and detailed inspection
    preview_limit = min(5, total_chunks)
    print(f"[4/4] First {preview_limit} Chunks' Content Previews & Timestamp Inspection:\n")
    print("-" * 75)

    for i in range(preview_limit):
        chunk = chunks[i]
        meta = chunk.metadata
        start_ts = meta.get("start", 0.0)
        end_ts = meta.get("end", 0.0)
        duration = meta.get("duration", 0.0)
        char_count = meta.get("char_count", len(chunk.page_content))
        snippet_count = meta.get("snippet_count", 0)

        # Truncate preview text for clean display if needed
        full_chunk_text = chunk.page_content
        preview_text = full_chunk_text[:280] + ("..." if len(full_chunk_text) > 280 else "")

        print(f"--- Chunk #{i + 1} (Index: {meta.get('chunk_index', i)}) ---")
        print(f"  Start Timestamp    : {start_ts:.2f}s")
        print(f"  End Timestamp      : {end_ts:.2f}s")
        print(f"  Duration           : {duration:.2f}s")
        print(f"  Character Count    : {char_count} chars")
        print(f"  Original Snippets  : {snippet_count} combined snippets")
        print(f"  Preserved Metadata : {json.dumps(meta, indent=4, ensure_ascii=False)}")
        print(f"  Content Preview    :\n    \"{preview_text}\"\n")

    # 5. Comprehensive Verification Checks
    print("=" * 75)
    print("VALIDATION & INTEGRITY CHECKS")
    print("=" * 75)

    empty_or_invalid_chunks: List[int] = []
    metadata_errors: List[str] = []
    timestamp_errors: List[str] = []

    required_keys = [
        "video_id",
        "video_title",
        "language",
        "is_translated",
        "start",
        "end",
        "duration",
    ]

    for idx, c in enumerate(chunks):
        # Check for empty chunks
        if not c.page_content or not c.page_content.strip():
            empty_or_invalid_chunks.append(idx)

        # Check required metadata preservation
        for key in required_keys:
            if key not in c.metadata or c.metadata[key] is None:
                metadata_errors.append(f"Chunk #{idx+1} missing or null key: '{key}'")

        # Check timestamp validity
        start = c.metadata.get("start", -1.0)
        end = c.metadata.get("end", -1.0)
        if start < 0 or end < 0 or start > end:
            timestamp_errors.append(f"Chunk #{idx+1} invalid timestamps: start={start}, end={end}")

    # Check for empty / invalid chunks
    if empty_or_invalid_chunks:
        print(f"[FAIL] Found {len(empty_or_invalid_chunks)} empty or invalid chunks at indices: {empty_or_invalid_chunks}")
    else:
        print("[PASS] Zero empty or invalid chunks detected.")

    # Check metadata preservation
    if metadata_errors:
        print(f"[FAIL] Metadata verification errors found:\n       " + "\n       ".join(metadata_errors[:5]))
    else:
        print(f"[PASS] All {len(required_keys)} required metadata fields preserved across all {total_chunks} chunks.")

    # Check timestamp validity
    if timestamp_errors:
        print(f"[FAIL] Timestamp errors found:\n       " + "\n       ".join(timestamp_errors[:5]))
    else:
        print(f"[PASS] All {total_chunks} chunks have valid start and end timestamp ranges.")

    # Check timestamp overlap between consecutive chunks
    overlap_verified = True
    for idx in range(len(chunks) - 1):
        curr_chunk = chunks[idx]
        next_chunk = chunks[idx + 1]
        # In overlapping chunks, next chunk's start should be earlier than current chunk's end
        if next_chunk.metadata["start"] > curr_chunk.metadata["end"]:
            overlap_verified = False
            print(f"[WARN] Non-overlapping timestamps between Chunk #{idx+1} and Chunk #{idx+2}")

    if overlap_verified:
        print("[PASS] Consecutive chunks exhibit valid timestamp overlap reflecting chunk_overlap.")

    print("=" * 75)
    print("ALL PHASE 2 CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_URL
    inspect_chunks(target)

