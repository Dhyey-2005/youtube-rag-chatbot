"""
backend/main.py
Phase 6: FastAPI Backend Server
Exposes REST API endpoints for video processing, semantic conversational chat with
source attribution, and health diagnostics for the YouTube RAG Chatbot.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from get_youtube_transcript import extract_video_id
from backend.ingestion import load_youtube_transcript
from backend.text_splitter import split_transcript_documents
from backend.vectorstore import (
    DEFAULT_VECTOR_DB_DIR,
    add_documents_to_vectorstore,
    create_vectorstore,
    get_vectorstore_stats,
    load_vectorstore,
)
from backend.chatbot import YouTubeChatbot, format_youtube_timestamp_url

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_api")

# Initialize FastAPI application
app = FastAPI(
    title="YouTube RAG Chatbot API",
    description="Backend API for YouTube video ingestion, semantic retrieval, and conversational Q&A with source attribution.",
    version="1.0.0",
)

# Configure CORS for React frontend (Vite default: 5173, Create-React-App default: 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-Memory State: Shared Vector Store & Session History Store
# ---------------------------------------------------------------------------
_vectorstore = None
_sessions: Dict[str, List[Dict[str, str]]] = {}


def get_or_load_vectorstore():
    """Returns the cached FAISS vector store, attempting to load from disk if present."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    index_file = DEFAULT_VECTOR_DB_DIR / "index.faiss"
    if index_file.exists():
        try:
            _vectorstore = load_vectorstore(folder_path=DEFAULT_VECTOR_DB_DIR)
            logger.info("Loaded persisted FAISS vector store from disk.")
        except Exception as exc:
            logger.warning(f"Could not load existing vector store: {exc}")
            _vectorstore = None
    return _vectorstore


# ---------------------------------------------------------------------------
# Pydantic Request & Response Models
# ---------------------------------------------------------------------------
class VideoProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or 11-character video ID")


class VideoProcessResponse(BaseModel):
    video_id: str
    title: str
    language: str
    chunk_count: int
    status: str
    message: str


class SourceItem(BaseModel):
    video_title: str
    video_id: str
    start: float
    end: float
    duration: float
    relevant_excerpt: str
    url: str
    similarity_score: Optional[float] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question or follow-up query")
    session_id: Optional[str] = Field(default="default_session", description="Conversation session ID")
    k: Optional[int] = Field(default=4, ge=1, le=10, description="Top-k chunks to retrieve")


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    retrieved_chunks: List[Dict[str, Any]]
    session_id: str


class ClearSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to clear")


class HealthResponse(BaseModel):
    status: str
    indexed_videos_count: int
    indexed_chunks_count: int
    indexed_videos: List[str]


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health status and current index diagnostics."""
    vs = get_or_load_vectorstore()
    if vs is not None:
        stats = get_vectorstore_stats(vs)
        return HealthResponse(
            status="healthy",
            indexed_videos_count=len(stats.get("indexed_video_ids", [])),
            indexed_chunks_count=stats.get("total_documents", 0),
            indexed_videos=stats.get("indexed_video_ids", []),
        )
    return HealthResponse(
        status="healthy (no index loaded yet)",
        indexed_videos_count=0,
        indexed_chunks_count=0,
        indexed_videos=[],
    )


@app.post("/api/video/process", response_model=VideoProcessResponse)
def process_video(request: VideoProcessRequest):
    """
    Ingests, chunks, embeds, and stores a YouTube video transcript.
    Skips reprocessing if the video has already been indexed.
    """
    global _vectorstore, _sessions

    clean_url = request.url.strip()
    if not clean_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video URL cannot be empty.",
        )

    # 1. Validate video ID
    video_id = extract_video_id(clean_url)
    if not re.match(r"^[0-9A-Za-z_-]{11}$", video_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid YouTube URL or ID '{clean_url}'. Could not extract an 11-character video ID.",
        )

    # 2. Check if this exact video is already the ONLY active indexed video
    vs = get_or_load_vectorstore()
    if vs is not None:
        stats = get_vectorstore_stats(vs)
        indexed_ids = stats.get("indexed_video_ids", [])
        if indexed_ids == [video_id]:
            logger.info(f"Video '{video_id}' is already the currently active indexed video. Resetting chat history.")
            _sessions.clear()
            doc_sample = next(
                (d for d in vs.docstore._dict.values() if d.metadata.get("video_id") == video_id),
                None,
            )
            title = doc_sample.metadata.get("video_title", video_id) if doc_sample else video_id
            language = doc_sample.metadata.get("language", "unknown") if doc_sample else "unknown"
            video_chunk_count = sum(
                1 for d in vs.docstore._dict.values() if d.metadata.get("video_id") == video_id
            )

            return VideoProcessResponse(
                video_id=video_id,
                title=title,
                language=language,
                chunk_count=video_chunk_count,
                status="already_indexed",
                message=f"Video '{title}' is already indexed and active. Chat history has been cleared.",
            )

    # 3. New video link detected: Clear all existing chat history and prior video indices
    logger.info(f"New video link detected ('{video_id}'). Clearing older video indices and chat history...")
    _sessions.clear()

    # 4. Phase 1: Ingestion
    logger.info(f"Ingesting transcript for video '{video_id}'...")
    try:
        raw_documents = load_youtube_transcript(video_id)
    except Exception as exc:
        logger.error(f"Transcript extraction failed for '{video_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not extract transcript for video '{video_id}': {exc}",
        )

    video_title = raw_documents[0].metadata.get("video_title", video_id)
    language = raw_documents[0].metadata.get("language", "unknown")

    # 5. Phase 2: Text Splitting
    logger.info(f"Splitting transcript into chunks for '{video_id}'...")
    try:
        chunks = split_transcript_documents(raw_documents, chunk_size=1000, chunk_overlap=200)
    except Exception as exc:
        logger.error(f"Text splitting failed for '{video_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text splitting failed: {exc}",
        )

    # 6. Phase 3 & 4: Embeddings & Fresh FAISS Storage (Replacing older video data)
    logger.info(f"Generating embeddings and creating fresh isolated FAISS store for '{video_id}'...")
    try:
        # Always create a brand-new vector store containing exclusively this video's chunks
        _vectorstore = create_vectorstore(chunks, save=True, folder_path=DEFAULT_VECTOR_DB_DIR)
    except Exception as exc:
        logger.error(f"Vector store indexing failed for '{video_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store indexing failed: {exc}",
        )

    return VideoProcessResponse(
        video_id=video_id,
        title=video_title,
        language=language,
        chunk_count=len(chunks),
        status="processed",
        message=f"Video '{video_title}' successfully processed into {len(chunks)} chunks. All older videos and chat history cleared.",
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_with_video(request: ChatRequest):
    """
    Answers a question about the indexed video(s) using conversational RAG
    with source attribution.
    """
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    vs = get_or_load_vectorstore()
    if vs is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No video has been processed yet. Please submit and process a YouTube video first.",
        )

    session_id = request.session_id.strip() if request.session_id else "default_session"
    history = _sessions.get(session_id, [])

    # Instantiate chatbot with the active vectorstore
    chatbot = YouTubeChatbot(vectorstore=vs)

    try:
        response_data = chatbot.ask(
            question=clean_question,
            conversation_history=history,
            k=request.k or 4,
        )
    except Exception as exc:
        logger.error(f"Chatbot failed to generate answer: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {exc}",
        )

    # Update in-memory session history
    _sessions[session_id] = response_data["conversation_history"]

    # Format sources into Pydantic models
    sources_out: List[SourceItem] = []
    for s in response_data.get("sources", []):
        sources_out.append(
            SourceItem(
                video_title=s.get("video_title", "Unknown"),
                video_id=s.get("video_id", "unknown"),
                start=s.get("start", 0.0),
                end=s.get("end", 0.0),
                duration=s.get("duration", 0.0),
                relevant_excerpt=s.get("relevant_excerpt", ""),
                url=s.get("url", ""),
                similarity_score=s.get("similarity_score"),
            )
        )

    return ChatResponse(
        answer=response_data["answer"],
        sources=sources_out,
        retrieved_chunks=response_data.get("retrieved_chunks", []),
        session_id=session_id,
    )


@app.post("/api/chat/clear")
def clear_session(request: ClearSessionRequest):
    """Clears conversational history for the given session ID."""
    sid = request.session_id.strip()
    if sid in _sessions:
        del _sessions[sid]
    return {"status": "cleared", "session_id": sid}


# ---------------------------------------------------------------------------
# Static File Mount: Serve Production React Frontend
# ---------------------------------------------------------------------------
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
    logger.info(f"Mounted React production frontend from: {FRONTEND_DIST_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
