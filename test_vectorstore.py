"""
test_vectorstore.py
Verification script for Phase 4: FAISS Vector Store Integration.
Tests creating a FAISS index, local persistence, loading, document count verification,
and deduplication to prevent redundant re-embedding.
"""

from __future__ import annotations

import sys
from pathlib import Path

from backend.ingestion import load_youtube_transcript
from backend.text_splitter import split_transcript_documents
from backend.vectorstore import (
    DEFAULT_VECTOR_DB_DIR,
    add_documents_to_vectorstore,
    create_vectorstore,
    get_vectorstore_stats,
    load_vectorstore,
    save_vectorstore,
)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Default video: 3Blue1Brown Chapter 1 Neural Networks
DEFAULT_TEST_URL = "https://www.youtube.com/watch?v=aircAruvnKk"


def test_vectorstore_lifecycle(video_url: str = DEFAULT_TEST_URL) -> None:
    print("=" * 75)
    print("PHASE 4 VERIFICATION: FAISS Vector Store Lifecycle")
    print("=" * 75)
    print(f"Target Video URL: {video_url}\n")

    # 1. Ingestion (Phase 1)
    print("[1/5] Ingesting transcript snippets (Phase 1)...")
    raw_docs = load_youtube_transcript(video_url)
    video_title = raw_docs[0].metadata.get("video_title", "Unknown")
    print(f"      Video Title      : {video_title}")
    print(f"      Raw Snippet Count: {len(raw_docs)}\n")

    # 2. Text Splitting (Phase 2)
    print("[2/5] Splitting transcript into chunks (Phase 2)...")
    chunks = split_transcript_documents(raw_docs, chunk_size=1000, chunk_overlap=200)
    expected_chunk_count = len(chunks)
    print(f"      Generated Chunks : {expected_chunk_count}\n")

    # 3. Create FAISS Vector Store (Phase 4)
    print("[3/5] Creating FAISS index and generating embeddings (Phase 3 & 4)...")
    vectorstore = create_vectorstore(chunks, save=False)
    initial_stats = get_vectorstore_stats(vectorstore)
    print(f"      Indexed Documents   : {initial_stats['total_documents']}")
    print(f"      Vector Dimensions   : {initial_stats['vector_dimension']}")
    print(f"      Associated Video IDs: {initial_stats['indexed_video_ids']}\n")

    if initial_stats["total_documents"] != expected_chunk_count:
        print(f"[FAIL] Expected {expected_chunk_count} documents, got {initial_stats['total_documents']}")
        sys.exit(1)
    print("[PASS] Vector store document count matches generated chunk count.")

    # 4. Save and Load Index
    print("\n[4/5] Persisting and reloading FAISS index...")
    save_path = save_vectorstore(vectorstore, folder_path=DEFAULT_VECTOR_DB_DIR)
    print(f"      Saved FAISS index to: {save_path}")

    loaded_vectorstore = load_vectorstore(folder_path=DEFAULT_VECTOR_DB_DIR)
    loaded_stats = get_vectorstore_stats(loaded_vectorstore)
    print(f"      Loaded Index Documents: {loaded_stats['total_documents']}")
    print(f"      Loaded Vector Dimension: {loaded_stats['vector_dimension']}")

    if loaded_stats["total_documents"] != expected_chunk_count:
        print(f"[FAIL] Reloaded index document count mismatch: {loaded_stats['total_documents']} != {expected_chunk_count}")
        sys.exit(1)
    print("[PASS] Reloaded FAISS index contains all expected documents and vectors.")

    # 5. Deduplication and Idempotent Indexing Verification
    print("\n[5/5] Testing deduplication (preventing duplicate records & re-embedding)...")
    added_count = add_documents_to_vectorstore(loaded_vectorstore, chunks, save_after=True)
    after_dedup_stats = get_vectorstore_stats(loaded_vectorstore)
    print(f"      Attempted to add {len(chunks)} existing chunks.")
    print(f"      Newly added documents : {added_count}")
    print(f"      Total documents now   : {after_dedup_stats['total_documents']}")

    if added_count != 0:
        print(f"[FAIL] Expected 0 newly added documents for duplicate chunks, got {added_count}")
        sys.exit(1)
    if after_dedup_stats["total_documents"] != expected_chunk_count:
        print(f"[FAIL] Document count changed unexpectedly after deduplication check.")
        sys.exit(1)
    print("[PASS] Deduplication check passed: 0 duplicate chunks added, 0 redundant embeddings generated.")

    print("\n" + "=" * 75)
    print("ALL VECTOR STORE TESTS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_URL
    test_vectorstore_lifecycle(target)

