"""
backend/ingestion.py
Phase 1: YouTube Transcript Ingestion Module
Converts YouTube video transcripts into LangChain Document objects preserving snippet-level metadata.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

# Ensure project root is available in sys.path when running or importing from subdirectories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from get_youtube_transcript import extract_video_id, get_youtube_transcript


def load_youtube_transcript(
    video_url: str,
    translate_to_english: bool = False,
    preferred_languages: Optional[List[str]] = None,
) -> List[Document]:
    """
    Ingests YouTube video transcript snippets and converts them into LangChain Document objects.

    Args:
        video_url: YouTube video URL (full, short, shorts) or 11-character video ID.
        translate_to_english: If True, translates non-English transcripts to English.
        preferred_languages: Optional list of priority language codes (e.g. ['en', 'es']).

    Returns:
        List[Document]: List of LangChain Document objects, one per transcript snippet,
        with metadata preserving video_id, video_title, start, duration, language, and is_translated.

    Raises:
        ValueError: If video_url is empty, invalid type, or contains an invalid YouTube video ID.
        RuntimeError: If transcript extraction fails (no captions, network error, etc.).
    """
    # 1. Input validation
    if not video_url or not isinstance(video_url, str) or not video_url.strip():
        raise ValueError("Video URL must be a non-empty string.")

    cleaned_url = video_url.strip()
    video_id = extract_video_id(cleaned_url)

    # YouTube video IDs are exactly 11 characters long
    if not re.match(r"^[0-9A-Za-z_-]{11}$", video_id):
        raise ValueError(
            f"Invalid YouTube URL or video ID: '{video_url}'. "
            "Could not extract a valid 11-character YouTube video ID."
        )

    # 2. Extract transcript using existing get_youtube_transcript module
    try:
        transcript_data = get_youtube_transcript(
            video_url_or_id=video_id,
            preferred_languages=preferred_languages,
            translate_to_english=translate_to_english,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract transcript for video '{video_id}': {exc}"
        ) from exc

    snippets = transcript_data.get("snippets", [])
    if not snippets:
        raise RuntimeError(
            f"No transcript snippets found for video '{video_id}' ({transcript_data.get('title', 'Unknown')})."
        )

    video_title = transcript_data.get("title", video_id)
    language = transcript_data.get("language", "unknown")
    is_translated = bool(transcript_data.get("is_translated", False))

    # 3. Convert snippets into LangChain Document objects
    documents: List[Document] = []
    for snippet in snippets:
        text = snippet.get("text", "").strip()
        if not text:
            continue

        # Preserve exact timestamps as float for timestamp-based source references
        start = float(snippet.get("start", 0.0))
        duration = float(snippet.get("duration", 0.0))

        doc = Document(
            page_content=text,
            metadata={
                "video_id": video_id,
                "video_title": video_title,
                "start": start,
                "duration": duration,
                "language": language,
                "is_translated": is_translated,
            },
        )
        documents.append(doc)

    return documents


# Alias for load_youtube_transcript
ingest_youtube_transcript = load_youtube_transcript

