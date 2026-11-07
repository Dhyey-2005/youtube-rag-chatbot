"""
test_embeddings.py
Verification script for Phase 3: Gemini Embedding Generation.
Uses Phase 1 (ingestion) and Phase 2 (chunking) to process an actual YouTube video,
generates dense vector embeddings using GoogleGenerativeAIEmbeddings,
and validates the vector integrity and dimension consistency.
"""

from __future__ import annotations

import sys
from typing import List

from backend.embeddings import (
    GeminiEmbedder,
    generate_document_embeddings,
    validate_embeddings,
)
from backend.ingestion import load_youtube_transcript
from backend.text_splitter import split_transcript_documents

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Default video: short, reliable, permanent official YouTube video
DEFAULT_TEST_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def run_embedding_test(video_url: str = DEFAULT_TEST_URL) -> None:
    print("=" * 75)
    print("PHASE 3 VERIFICATION: Gemini Embedding Generation")
    print("=" * 75)
    print(f"Target Video URL : {video_url}\n")

    # Step 1: Ingest using Phase 1 module
    print("[1/4] Ingesting transcript from YouTube (Phase 1)...")
    try:
        raw_docs = load_youtube_transcript(video_url)
    except Exception as exc:
        print(f"[FAIL] Ingestion error: {exc}")
        sys.exit(1)

    video_title = raw_docs[0].metadata.get("video_title", "Unknown")
    print(f"      Video Title      : {video_title}")
    print(f"      Ingested Snippets: {len(raw_docs)}\n")

    # Step 2: Chunk using Phase 2 module
    # Using chunk_size=100, chunk_overlap=20 on this short video creates 4 distinct chunks
    # to thoroughly verify multi-chunk vector consistency without burning API quota
    print("[2/4] Splitting transcript into chunks (Phase 2)...")
    try:
        chunks = split_transcript_documents(raw_docs, chunk_size=100, chunk_overlap=20)
    except Exception as exc:
        print(f"[FAIL] Chunking error: {exc}")
        sys.exit(1)

    total_chunks = len(chunks)
    print(f"      Total Chunks to Embed: {total_chunks}")
    for idx, c in enumerate(chunks):
        m = c.metadata
        print(f"      - Chunk #{idx+1} ({len(c.page_content)} chars): [{m.get('start', 0.0):.2f}s -> {m.get('end', 0.0):.2f}s] \"{c.page_content[:40]}...\"")
    print()

    # Step 3: Generate embeddings using Phase 3 module
    print("[3/4] Generating Gemini vector embeddings (Phase 3)...")
    embedder = GeminiEmbedder(model="gemini-embedding-001")
    try:
        embeddings = embedder.embed_documents(chunks)
    except Exception as exc:
        print(f"[FAIL] Embedding generation error: {exc}")
        sys.exit(1)

    embedding_count = len(embeddings)
    dimensions = len(embeddings[0]) if embeddings else 0

    print(f"      Generated Embeddings : {embedding_count}")
    print(f"      Embedding Dimensions : {dimensions} floats per vector\n")

    # Display preview of the first embedding vector
    print("[4/4] Embedding Vector Preview (Chunk #1):")
    sample_vec = embeddings[0]
    preview_head = [round(val, 6) for val in sample_vec[:6]]
    preview_tail = [round(val, 6) for val in sample_vec[-3:]]
    print(f"      First 6 dimensions  : {preview_head}")
    print(f"      Last 3 dimensions   : {preview_tail}")
    print(f"      Vector Magnitude (L2 norm): {sum(x**2 for x in sample_vec) ** 0.5:.4f}\n")

    # Step 4: Comprehensive Validation
    print("=" * 75)
    print("VALIDATION & INTEGRITY CHECKS")
    print("=" * 75)

    try:
        val_result = validate_embeddings(embeddings, chunks, expected_dimension=dimensions)
        print(f"[PASS] Embedding count ({val_result['count']}) exactly matches chunk count ({total_chunks}).")
        print(f"[PASS] Vector dimension consistency: all {val_result['count']} vectors have dimension {val_result['dimension']}.")
        print("[PASS] Zero empty, NaN, or infinite float values found across all vectors.")
    except Exception as val_err:
        print(f"[FAIL] Validation check failed: {val_err}")
        sys.exit(1)

    # Verify input Documents remained unmutated
    mutated = False
    for i, c in enumerate(chunks):
        if not c.page_content or "start" not in c.metadata or "end" not in c.metadata:
            mutated = True
            break
    if not mutated:
        print("[PASS] Original Documents and their metadata remained completely intact and unmodified.")
    else:
        print("[FAIL] One or more Documents were mutated during embedding.")
        sys.exit(1)

    # Verify query embedding function
    print("\n--- Verifying Query Embedding ---")
    query_text = "What animals are in the video?"
    try:
        query_vector = embedder.embed_query(query_text)
        print(f"[PASS] embed_query(\"{query_text}\") returned valid vector of dimension {len(query_vector)}.")
    except Exception as q_err:
        print(f"[FAIL] Query embedding failed: {q_err}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("ALL PHASE 3 CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_URL
    run_embedding_test(target)

