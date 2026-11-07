"""
Backend package for YouTube RAG Chatbot.
"""

from .chatbot import (
    DEFAULT_CHAT_MODEL,
    YouTubeChatbot,
    ask_question,
    format_youtube_timestamp_url,
    get_chat_model,
)
from .embeddings import (
    GeminiEmbedder,
    generate_document_embeddings,
    get_embedding_model,
    validate_embeddings,
)
from .ingestion import ingest_youtube_transcript, load_youtube_transcript
from .retriever import (
    DEFAULT_K,
    TranscriptRetriever,
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_with_scores,
)
from .text_splitter import TranscriptTextSplitter, split_transcript_documents
from .vectorstore import (
    DEFAULT_VECTOR_DB_DIR,
    add_documents_to_vectorstore,
    create_vectorstore,
    get_vectorstore_stats,
    load_vectorstore,
    save_vectorstore,
)

__all__ = [
    # Ingestion (Phase 1)
    "load_youtube_transcript",
    "ingest_youtube_transcript",
    # Text Splitting (Phase 2)
    "TranscriptTextSplitter",
    "split_transcript_documents",
    # Embeddings (Phase 3)
    "GeminiEmbedder",
    "generate_document_embeddings",
    "get_embedding_model",
    "validate_embeddings",
    # Vector Store & Retrieval (Phase 4)
    "create_vectorstore",
    "save_vectorstore",
    "load_vectorstore",
    "add_documents_to_vectorstore",
    "get_vectorstore_stats",
    "DEFAULT_VECTOR_DB_DIR",
    "retrieve_relevant_chunks",
    "retrieve_relevant_chunks_with_scores",
    "TranscriptRetriever",
    "DEFAULT_K",
    # LLM & Chatbot (Phase 5)
    "YouTubeChatbot",
    "ask_question",
    "format_youtube_timestamp_url",
    "get_chat_model",
    "DEFAULT_CHAT_MODEL",
]
