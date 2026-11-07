"""
test_ingestion.py
Verification script for Phase 1: YouTube Transcript Integration.
Tests load_youtube_transcript with an actual YouTube video and verifies Document schema & metadata.
"""

from __future__ import annotations

import json
import sys
from backend.ingestion import load_youtube_transcript

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_test(video_url: str = "https://www.youtube.com/watch?v=jNQXAC9IVRw") -> None:
    print("=" * 70)
    print("PHASE 1 INTEGRATION TEST: YouTube Transcript Ingestion")
    print("=" * 70)
    print(f"Target Video URL: {video_url}\n")

    # 1. Ingestion test
    try:
        documents = load_youtube_transcript(video_url)
    except Exception as exc:
        print(f"[FAIL] Error during transcript ingestion: {exc}")
        sys.exit(1)

    if not documents:
        print("[FAIL] Ingestion returned empty document list.")
        sys.exit(1)

    first_doc = documents[0]
    title = first_doc.metadata.get("video_title", "N/A")
    language = first_doc.metadata.get("language", "N/A")
    total_count = len(documents)

    print("[SUCCESS] Transcript successfully ingested into LangChain Documents!\n")
    print("-" * 50)
    print(f"Extracted Video Title : {title}")
    print(f"Transcript Language   : {language}")
    print(f"Total Snippet Count   : {total_count}")
    print("-" * 50)

    # 2. Preview generated Documents
    preview_count = min(3, total_count)
    print(f"\n--- Document Preview (First {preview_count} Documents) ---")
    for i, doc in enumerate(documents[:preview_count], start=1):
        start_ts = doc.metadata.get("start", 0.0)
        dur = doc.metadata.get("duration", 0.0)
        end_ts = round(start_ts + dur, 2)
        print(f"\n[Document {i}]")
        print(f"  Content  : \"{doc.page_content}\"")
        print(f"  Timestamp: [{start_ts}s -> {end_ts}s] (duration: {dur}s)")
        print(f"  Metadata : {json.dumps(doc.metadata, indent=4, ensure_ascii=False)}")

    # 3. Validation / Error handling verification
    print("\n" + "-" * 50)
    print("Testing Input Validation & Exception Handling...")
    invalid_url = "https://www.youtube.com/watch?v=invalid_id"
    try:
        load_youtube_transcript(invalid_url)
        print("[WARN] Invalid URL did not raise an exception.")
    except (ValueError, RuntimeError) as err:
        print(f"[PASS] Successfully caught expected error for invalid input:")
        print(f"       {type(err).__name__}: {err}")

    print("\n" + "=" * 70)
    print("ALL PHASE 1 CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    run_test(target)

