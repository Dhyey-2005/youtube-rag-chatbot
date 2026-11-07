"""
Reliable YouTube Transcript Extractor
Bypasses YouTube's IP blocking (HTTP 429 / bot detection) by routing through
the YouTube Android mobile client API via yt-dlp.
Supports any video language with zero-LLM-quota English translation capabilities.
"""

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, TypedDict
import yt_dlp


class TranscriptSnippet(TypedDict):
    text: str
    start: float
    duration: float


class TranscriptResult(TypedDict, total=False):
    video_id: str
    title: str
    language: str
    full_text: str
    snippets: List[TranscriptSnippet]
    is_translated: bool
    original_language: Optional[str]


def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from any URL format or raw ID."""
    cleaned = url_or_id.strip().strip("'\"")
    # Handles: /watch?v=ID, /shorts/ID, /embed/ID, /v/ID, youtu.be/ID
    match = re.search(
        r"(?:v=|\/shorts\/|\/embed\/|\/v\/|youtu\.be\/)([0-9A-Za-z_-]{11})",
        cleaned,
    )
    if match:
        return match.group(1)
    # If raw 11-char ID is passed
    id_match = re.search(r"^[0-9A-Za-z_-]{11}$", cleaned)
    if id_match:
        return id_match.group(0)
    return cleaned


def translate_text_to_english(text: str) -> str:
    """
    Translates text to English using Google's free client translation endpoint.
    Uses 0 LLM tokens and requires no API keys.
    Automatically chunks text exceeding 4,000 characters to stay within safe payload limits.
    """
    if not text or not text.strip():
        return text

    max_len = 4000
    if len(text) <= max_len:
        chunks = [text]
    else:
        chunks = []
        current: List[str] = []
        current_len = 0
        for line in text.split("\n"):
            if current_len + len(line) + 1 > max_len and current:
                chunks.append("\n".join(current))
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += len(line) + 1
        if current:
            chunks.append("\n".join(current))

    translated_parts: List[str] = []
    for chunk in chunks:
        url = (
            "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q="
            + urllib.parse.quote(chunk)
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                chunk_translated = "".join(part[0] for part in data[0] if part and part[0])
                translated_parts.append(chunk_translated)
        except Exception:
            # Fallback to original text if connection fails
            translated_parts.append(chunk)

    return "\n".join(translated_parts)


def batch_translate_snippets_to_english(
    snippets: List[TranscriptSnippet], batch_size: int = 30
) -> List[TranscriptSnippet]:
    """
    Translates transcript snippets to English in batches, preserving exact start and duration timestamps.
    """
    if not snippets:
        return []

    translated_snippets: List[TranscriptSnippet] = []
    separator = "\n<<<SEP>>>\n"

    for i in range(0, len(snippets), batch_size):
        batch = snippets[i : i + batch_size]
        raw_texts = [s["text"] for s in batch]
        joined = separator.join(raw_texts)

        try:
            translated_joined = translate_text_to_english(joined)
            parts = [p.strip() for p in translated_joined.split("<<<SEP>>>")]
            if len(parts) == len(batch):
                for s, trans in zip(batch, parts):
                    translated_snippets.append({
                        "text": trans,
                        "start": s["start"],
                        "duration": s["duration"],
                    })
                continue
        except Exception:
            pass

        # Per-snippet fallback if batch delimiter matching fails
        for s in batch:
            try:
                translated_text = translate_text_to_english(s["text"])
            except Exception:
                translated_text = s["text"]
            translated_snippets.append({
                "text": translated_text,
                "start": s["start"],
                "duration": s["duration"],
            })

    return translated_snippets


def _fetch_caption_data(caption_url: str, ydl: yt_dlp.YoutubeDL) -> Dict[str, Any]:
    """Fetches JSON3 caption data, prioritizing yt-dlp's connection pool to bypass 429 blocks."""
    # Attempt 1: Route through yt-dlp's urlopen
    try:
        content = ydl.urlopen(caption_url).read().decode("utf-8")
        return json.loads(content)
    except Exception:
        pass

    # Attempt 2: Standard urllib fallback
    req = urllib.request.Request(
        caption_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_caption_events(caption_data: Dict[str, Any]) -> tuple[List[TranscriptSnippet], str]:
    """Parses JSON3 caption events into structured snippets and unified text."""
    snippets: List[TranscriptSnippet] = []
    text_chunks: List[str] = []

    for event in caption_data.get("events", []):
        if not isinstance(event, dict) or "segs" not in event:
            continue
        segs = event.get("segs", [])
        if not isinstance(segs, list):
            continue

        snippet_text = "".join(
            str(seg.get("utf8", "")) for seg in segs if isinstance(seg, dict)
        ).strip()
        if not snippet_text:
            continue

        start_val = event.get("tStartMs", 0)
        dur_val = event.get("dDurationMs", 0)
        start_sec = round(float(start_val) / 1000.0, 2)
        dur_sec = round(float(dur_val) / 1000.0, 2)

        snippets.append({
            "text": snippet_text,
            "start": start_sec,
            "duration": dur_sec,
        })
        text_chunks.append(snippet_text)

    full_text = " ".join(text_chunks)
    return snippets, full_text


def get_youtube_transcript(
    video_url_or_id: str,
    preferred_languages: Optional[List[str]] = None,
    translate_to_english: bool = False,
) -> TranscriptResult:
    """
    Automatically retrieves the transcript for any YouTube video in any language.
    Supports seamless translation to English without consuming any LLM quota.

    Args:
        video_url_or_id: YouTube video URL (full, short, shorts) or 11-char video ID.
        preferred_languages: Optional priority list of language codes. If None,
            automatically picks the original audio language track or any available track.
        translate_to_english: When True, translates non-English transcripts to English
            using YouTube's server-side translation or free client translation (0 LLM tokens).

    Returns:
        TranscriptResult dict with video_id, title, language, full_text, snippets,
        is_translated, and original_language.
    """
    video_id = extract_video_id(video_url_or_id)
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Route through YouTube's mobile Android client to bypass 429 IP blocks
    ydl_opts: Dict[str, Any] = {
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios"]
            }
        },
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }
    ydl_params: Any = ydl_opts

    with yt_dlp.YoutubeDL(ydl_params) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        info = info_dict if isinstance(info_dict, dict) else {}

        raw_title = info.get("title")
        title: str = str(raw_title) if raw_title else video_id

        raw_subtitles = info.get("subtitles")
        subtitles: Dict[str, Any] = raw_subtitles if isinstance(raw_subtitles, dict) else {}

        raw_auto_captions = info.get("automatic_captions")
        auto_captions: Dict[str, Any] = raw_auto_captions if isinstance(raw_auto_captions, dict) else {}

        # Pool manual subtitles and automatic captions
        available_tracks: Dict[str, Any] = {}
        for lang, track in subtitles.items():
            available_tracks[lang] = track
        for lang, track in auto_captions.items():
            if lang not in available_tracks:
                available_tracks[lang] = track

        if not available_tracks:
            raise RuntimeError(f"No transcripts or captions available for video: {video_id}")

        selected_track: Optional[List[Dict[str, Any]]] = None
        selected_lang: Optional[str] = None

        # 0. If user requests English translation and an English track already exists, use it directly
        if translate_to_english and not preferred_languages:
            for lang_key, track_list in available_tracks.items():
                if lang_key.lower().startswith("en"):
                    selected_track = track_list if isinstance(track_list, list) else None
                    selected_lang = lang_key
                    break

        # 1. If preferred languages specified, check them first
        if not selected_track and preferred_languages:
            for pref in preferred_languages:
                for lang_key, track_list in available_tracks.items():
                    if lang_key.lower() == pref.lower() or lang_key.lower().startswith(pref.lower()):
                        selected_track = track_list if isinstance(track_list, list) else None
                        selected_lang = lang_key
                        break
                if selected_track:
                    break

        # 2. Prefer original speech language track (e.g. 'en-orig', 'hi-orig', 'es-orig')
        if not selected_track:
            for lang_key, track_list in available_tracks.items():
                if "orig" in lang_key.lower():
                    selected_track = track_list if isinstance(track_list, list) else None
                    selected_lang = lang_key
                    break

        # 3. Prefer manual subtitles if available
        if not selected_track and subtitles:
            for lang_key, track_list in subtitles.items():
                selected_track = track_list if isinstance(track_list, list) else None
                selected_lang = lang_key
                if selected_track:
                    break

        # 4. Fallback to any available language track
        if not selected_track:
            fallback_lang, fallback_track = next(iter(available_tracks.items()))
            selected_lang = fallback_lang
            selected_track = fallback_track if isinstance(fallback_track, list) else None

        if not selected_track:
            raise RuntimeError(f"Could not load captions track for video: {video_id}")

        # Find the JSON3 format URL (standard structured caption format)
        json3_entry: Optional[Dict[str, Any]] = next(
            (item for item in selected_track if isinstance(item, dict) and item.get("ext") == "json3"),
            None,
        )
        if not json3_entry:
            json3_entry = selected_track[0] if isinstance(selected_track[0], dict) else {}

        caption_url = str(json3_entry.get("url", ""))
        if not caption_url:
            raise RuntimeError(f"No valid caption URL found for video: {video_id}")

        original_lang = str(selected_lang or "unknown")
        is_already_english = original_lang.lower().startswith("en")
        needs_translation = translate_to_english and not is_already_english

        snippets: List[TranscriptSnippet] = []
        full_text: str = ""
        translated_successfully = False

        # Attempt A: YouTube server-side auto-translation (&tlang=en)
        if needs_translation:
            separator = "&" if "?" in caption_url else "?"
            tlang_caption_url = f"{caption_url}{separator}tlang=en"
            try:
                caption_data = _fetch_caption_data(tlang_caption_url, ydl)
                s_list, f_txt = _parse_caption_events(caption_data)
                if s_list:
                    snippets = s_list
                    full_text = f_txt
                    translated_successfully = True
            except Exception:
                pass

        # Attempt B: Fetch original native caption track
        if not snippets:
            caption_data = _fetch_caption_data(caption_url, ydl)
            snippets, full_text = _parse_caption_events(caption_data)

        # Attempt C: If translation needed but server-side tlang wasn't available, translate with client translator
        if needs_translation and not translated_successfully and snippets:
            snippets = batch_translate_snippets_to_english(snippets)
            full_text = " ".join(s["text"] for s in snippets)
            translated_successfully = True

        display_language = (
            f"en (translated from {original_lang})"
            if translated_successfully
            else original_lang
        )

        return {
            "video_id": video_id,
            "title": title,
            "language": display_language,
            "full_text": full_text,
            "snippets": snippets,
            "is_translated": translated_successfully,
            "original_language": original_lang if needs_translation else None,
        }


if __name__ == "__main__":
    import sys

    # Ensure Windows console supports Unicode characters for any language
    reconfigure_fn = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_fn):
        reconfigure_fn(encoding="utf-8")

    # Ask the user for the video ID or URL interactively
    user_input = input("Enter YouTube Video ID or URL: ").strip()

    if not user_input:
        print("Error: No video ID or URL was provided.")
        sys.exit(1)

    trans_choice = input("Translate to English if non-English? (y/n, default: y): ").strip().lower()
    translate_en = trans_choice not in ("n", "no", "0", "false")

    print(f"\nFetching transcript for: {user_input} ...")

    try:
        result: TranscriptResult = get_youtube_transcript(
            user_input, translate_to_english=translate_en
        )

        print(f"\nTitle: {result['title']}")
        print(f"Language: {result['language']}")
        if result.get("is_translated"):
            print(f"Original Audio Language: {result.get('original_language')}")
        print(f"Total Snippets: {len(result['snippets'])}")

        print("\n--- First 5 Snippets ---")
        for s in result["snippets"][:5]:
            start = s["start"]
            end = round(start + s["duration"], 2)
            print(f"[{start}s -> {end}s] {s['text']}")

        print("\n--- Transcript Preview (First 400 characters) ---")
        full_text_str = result["full_text"]
        print(full_text_str[:400] + ("..." if len(full_text_str) > 400 else ""))

    except Exception as err:
        print(f"\nError retrieving transcript: {err}")
        sys.exit(1)
