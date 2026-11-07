"""
backend/text_splitter.py
Phase 2: Transcript Text Splitting Module
Splits YouTube transcript Documents into chunks using LangChain's
RecursiveCharacterTextSplitter while computing and preserving exact
start/end timestamps across combined and overlapping snippets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Default configuration values specified for Phase 2
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200


class TranscriptTextSplitter:
    """
    Splits YouTube transcript Documents into chunk Documents using LangChain's
    RecursiveCharacterTextSplitter, computing exact start and end timestamp ranges
    across merged and overlapping snippets.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the TranscriptTextSplitter.

        Args:
            chunk_size: Maximum character length per chunk (default: 1000).
            chunk_overlap: Character overlap between consecutive chunks (default: 200).
            separators: Optional list of separators for RecursiveCharacterTextSplitter.
            **kwargs: Extra arguments forwarded to RecursiveCharacterTextSplitter.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            **kwargs,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Combines and splits transcript snippet Documents into chunk Documents.
        Accurately calculates start and end timestamps based on original snippet boundaries.

        Args:
            documents: List of LangChain Documents produced by Phase 1 ingestion.

        Returns:
            List[Document]: Chunked LangChain Documents with preserved metadata
            and calculated start/end timestamp ranges.

        Raises:
            TypeError: If documents is not a list.
            ValueError: If documents contains non-Document objects or is empty.
        """
        if not isinstance(documents, list):
            raise TypeError(f"Expected a list of Document objects, got {type(documents).__name__}")

        if not documents:
            raise ValueError("Input documents list cannot be empty.")

        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise ValueError(
                    f"Item at index {i} is not a LangChain Document instance ({type(doc).__name__})."
                )

        # Group documents by video_id to support processing multi-video inputs cleanly
        video_groups: Dict[str, List[Document]] = {}
        for doc in documents:
            vid = str(doc.metadata.get("video_id", "unknown_video"))
            video_groups.setdefault(vid, []).append(doc)

        chunked_documents: List[Document] = []

        for video_id, video_docs in video_groups.items():
            chunks_for_video = self._split_single_video_documents(video_id, video_docs)
            chunked_documents.extend(chunks_for_video)

        return chunked_documents

    def _split_single_video_documents(
        self, video_id: str, video_docs: List[Document]
    ) -> List[Document]:
        """
        Processes and splits transcript Documents belonging to a single video.
        """
        spans: List[Dict[str, Any]] = []
        joined_parts: List[str] = []
        curr_char = 0
        separator = " "

        for doc in video_docs:
            text = doc.page_content.strip()
            if not text:
                continue

            if joined_parts:
                joined_parts.append(separator)
                curr_char += len(separator)

            start_char = curr_char
            joined_parts.append(text)
            curr_char += len(text)
            end_char = curr_char

            start_time = float(doc.metadata.get("start", 0.0))
            duration = float(doc.metadata.get("duration", 0.0))
            end_time = round(start_time + duration, 2)

            spans.append({
                "start_char": start_char,
                "end_char": end_char,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "metadata": doc.metadata,
            })

        full_text = "".join(joined_parts)
        if not full_text.strip():
            logger.warning(f"Video '{video_id}' has no non-empty snippet text to chunk.")
            return []

        # Use LangChain's RecursiveCharacterTextSplitter to split the combined text
        raw_chunks = self.text_splitter.split_text(full_text)

        chunks: List[Document] = []
        search_pos = 0

        # Retrieve common metadata from the first valid snippet
        first_meta = video_docs[0].metadata
        video_title = first_meta.get("video_title", video_id)
        language = first_meta.get("language", "unknown")
        is_translated = first_meta.get("is_translated", False)

        for chunk_idx, raw_chunk in enumerate(raw_chunks):
            chunk_text = raw_chunk.strip()
            if not chunk_text:
                continue

            # Locate where this chunk starts and ends in full_text
            match_start = full_text.find(raw_chunk, search_pos)
            if match_start == -1:
                match_start = full_text.find(raw_chunk)
            if match_start == -1:
                match_start = full_text.find(chunk_text, search_pos)
            if match_start == -1:
                match_start = full_text.find(chunk_text)

            if match_start != -1:
                match_end = match_start + len(raw_chunk)
                # Advance search_pos for the next chunk, allowing for chunk_overlap
                search_pos = max(0, match_start + 1)
            else:
                match_start = 0
                match_end = len(full_text)

            # Find all original snippets that overlap with this chunk's character range
            overlapping = [
                s for s in spans
                if s["end_char"] > match_start and s["start_char"] < match_end
            ]

            if overlapping:
                first_s = overlapping[0]
                last_s = overlapping[-1]

                if first_s is last_s and (first_s["end_char"] - first_s["start_char"]) > 0:
                    # In the rare event that a single snippet spans multiple chunks, interpolate
                    snippet_len = first_s["end_char"] - first_s["start_char"]
                    rel_start = max(0, match_start - first_s["start_char"])
                    rel_end = min(snippet_len, match_end - first_s["start_char"])
                    chunk_start_ts = round(
                        first_s["start_time"] + (rel_start / snippet_len) * first_s["duration"], 2
                    )
                    chunk_end_ts = round(
                        first_s["start_time"] + (rel_end / snippet_len) * first_s["duration"], 2
                    )
                else:
                    chunk_start_ts = first_s["start_time"]
                    chunk_end_ts = last_s["end_time"]

                chunk_duration = round(chunk_end_ts - chunk_start_ts, 2)
            else:
                chunk_start_ts = float(video_docs[0].metadata.get("start", 0.0))
                last_doc = video_docs[-1]
                chunk_end_ts = round(
                    float(last_doc.metadata.get("start", 0.0))
                    + float(last_doc.metadata.get("duration", 0.0)),
                    2,
                )
                chunk_duration = round(chunk_end_ts - chunk_start_ts, 2)

            meta: Dict[str, Any] = {
                "video_id": video_id,
                "video_title": video_title,
                "language": language,
                "is_translated": is_translated,
                "start": chunk_start_ts,
                "end": chunk_end_ts,
                "duration": chunk_duration,
                "chunk_index": chunk_idx,
                "char_count": len(chunk_text),
                "snippet_count": len(overlapping),
            }

            chunks.append(Document(page_content=chunk_text, metadata=meta))

        return chunks


def split_transcript_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    **kwargs: Any,
) -> List[Document]:
    """
    Functional helper to split transcript Documents using TranscriptTextSplitter.

    Args:
        documents: List of LangChain Documents from Phase 1 ingestion.
        chunk_size: Maximum character count per chunk (default: 1000).
        chunk_overlap: Overlapping character count between chunks (default: 200).
        **kwargs: Additional options passed to TranscriptTextSplitter.

    Returns:
        List[Document]: Resulting chunk Documents with preserved metadata and timestamps.
    """
    splitter = TranscriptTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
    )
    return splitter.split_documents(documents)

