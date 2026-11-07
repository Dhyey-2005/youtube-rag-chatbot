"""
backend/ingestion.py
Phase 1: YouTube Transcript Ingestion Module
Converts YouTube video transcripts into LangChain Document objects preserving snippet-level metadata.
Includes reliable fallback to youtube_transcript_api with automatic cookies.txt authentication
for cloud environments where anonymous requests encounter bot challenges.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from langchain_core.documents import Document

# Ensure project root is available in sys.path when running or importing from subdirectories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from get_youtube_transcript import extract_video_id, get_youtube_transcript

logger = logging.getLogger("ingestion")


def _get_cookie_session() -> Optional[requests.Session]:
    """
    Loads cookies.txt if present (supports local repo root and Render's /etc/secrets/ mount).
    Returns an authenticated requests.Session for bypassing YouTube cloud datacenter bot challenges.
    """
    candidates = [
        Path("cookies.txt"),
        PROJECT_ROOT / "cookies.txt",
        Path("/etc/secrets/cookies.txt"),
    ]
    cookie_file = next((p for p in candidates if p.is_file()), None)
    if not cookie_file:
        return None

    try:
        jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
        jar.load(ignore_discard=True, ignore_expires=True)
        session = requests.Session()
        session.cookies = jar
        logger.info(f"Successfully loaded authenticated session from '{cookie_file}'")
        return session
    except Exception as exc:
        logger.warning(f"Failed to load cookies from '{cookie_file}': {exc}")
        return None


def _fetch_via_youtube_transcript_api(
    video_id: str,
    preferred_languages: Optional[List[str]] = None,
    translate_to_english: bool = False,
) -> Dict[str, Any]:
    """
    Fallback transcript fetcher using youtube_transcript_api.
    Bypasses yt-dlp web player bot checks on cloud datacenter IPs (Render, AWS, GCP).
    Automatically attaches authenticated cookies session if available.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    session = _get_cookie_session()
    api = YouTubeTranscriptApi(http_client=session)
    transcript_list = api.list(video_id)

    selected_transcript = None
    selected_lang = "en"
    is_translated = False

    # 1. Check preferred languages if specified
    if preferred_languages:
        try:
            selected_transcript = transcript_list.find_transcript(preferred_languages)
            selected_lang = selected_transcript.language_code
        except Exception:
            pass

    # 2. Check English if requested
    if selected_transcript is None and translate_to_english:
        try:
            selected_transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
            selected_lang = selected_transcript.language_code
        except Exception:
            # Try auto-translating the first available track
            try:
                first_t = next(iter(transcript_list))
                if first_t.is_translatable:
                    selected_transcript = first_t.translate("en")
                    selected_lang = "en"
                    is_translated = True
            except Exception:
                pass

    # 3. Fall back to any available transcript
    if selected_transcript is None:
        selected_transcript = next(iter(transcript_list))
        selected_lang = selected_transcript.language_code

    fetched_raw = selected_transcript.fetch()
    snippets: List[Dict[str, Any]] = []
    for item in fetched_raw:
        text = (item.text if hasattr(item, "text") else item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.start if hasattr(item, "start") else item.get("start", 0.0))
        duration = float(item.duration if hasattr(item, "duration") else item.get("duration", 0.0))
        snippets.append({"text": text, "start": start, "duration": duration})

    # Fetch title via official public oEmbed API
    title = video_id
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(
            oembed_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            oembed_data = json.loads(resp.read().decode("utf-8"))
            title = oembed_data.get("title", video_id)
    except Exception:
        title = video_id

    return {
        "video_id": video_id,
        "title": title,
        "language": selected_lang,
        "snippets": snippets,
        "is_translated": is_translated,
    }


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
        RuntimeError: If transcript extraction fails across all methods.
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

    # 2. Extract transcript using primary extractor with fallback
    try:
        transcript_data = get_youtube_transcript(
            video_url_or_id=video_id,
            preferred_languages=preferred_languages,
            translate_to_english=translate_to_english,
        )
    except Exception as exc:
        logger.warning(
            f"Primary yt-dlp extractor failed for '{video_id}' ({exc}). Attempting fallback via youtube_transcript_api..."
        )
        try:
            transcript_data = _fetch_via_youtube_transcript_api(
                video_id=video_id,
                preferred_languages=preferred_languages,
                translate_to_english=translate_to_english,
            )
            logger.info(
                f"Fallback youtube_transcript_api succeeded for '{video_id}' with {len(transcript_data.get('snippets', []))} snippets."
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Failed to extract transcript for video '{video_id}' via primary extractor ({exc}) and fallback ({fallback_exc})"
            ) from fallback_exc

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
