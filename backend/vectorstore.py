"""
backend/vectorstore.py
Phase 4: FAISS Vector Store Integration Module
Creates, saves, loads, and updates local FAISS vector store indices for YouTube transcript chunks.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Default path for FAISS local persistence
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VECTOR_DB_DIR = PROJECT_ROOT / "backend" / "vector_db"
DEFAULT_INDEX_NAME = "index"

# Embedding rate limit configuration for Google Gemini Free Tier (30,000 tokens/min)
DEFAULT_EMBED_BATCH_SIZE: int = 8
DEFAULT_BATCH_PAUSE_SECONDS: float = 1.5
DEFAULT_MAX_RATE_LIMIT_RETRIES: int = 6


def _extract_retry_delay(error_str: str, default_wait: float = 25.0) -> float:
    """
    Extracts the recommended wait time from Google's 429 RESOURCE_EXHAUSTED error message.
    e.g. 'Please retry in 22.679882003s.' or 'retryDelay': '22s'.
    """
    match = re.search(r"retry in ([\d\.]+)s", error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.0  # Add 1s safety buffer
        except ValueError:
            pass

    match_delay = re.search(r"retryDelay':\s*'(\d+)s'", error_str)
    if match_delay:
        try:
            return float(match_delay.group(1)) + 1.0
        except ValueError:
            pass

    return default_wait


def _embed_and_add_in_batches(
    vectorstore: Optional[FAISS],
    documents: List[Document],
    ids: List[str],
    embeddings: Embeddings,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    pause_between_batches: float = DEFAULT_BATCH_PAUSE_SECONDS,
    max_retries: int = DEFAULT_MAX_RATE_LIMIT_RETRIES,
) -> FAISS:
    """
    Embeds documents and inserts them into FAISS in safe, rate-limited batches with
    automatic 429 RESOURCE_EXHAUSTED retry backoff to support long videos on Free Tier.
    """
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    current_vs = vectorstore

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_docs)
        batch_docs = documents[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]

        logger.info(
            f"Embedding batch {batch_idx + 1}/{total_batches} "
            f"({len(batch_docs)} chunks, items {start_idx + 1}-{end_idx} of {total_docs})..."
        )

        for attempt in range(1, max_retries + 1):
            try:
                if current_vs is None:
                    current_vs = FAISS.from_documents(
                        documents=batch_docs,
                        embedding=embeddings,
                        ids=batch_ids,
                    )
                else:
                    current_vs.add_documents(documents=batch_docs, ids=batch_ids)
                break  # Batch successfully embedded and indexed!
            except Exception as exc:
                err_str = str(exc)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries:
                    wait_sec = _extract_retry_delay(err_str, default_wait=25.0 * attempt)
                    logger.warning(
                        f"Rate limit reached (429 RESOURCE_EXHAUSTED) on batch {batch_idx + 1}/{total_batches}. "
                        f"Auto-pausing for {wait_sec:.1f}s for Google Free Tier quota window to reset "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(wait_sec)
                else:
                    logger.error(f"Embedding failed on batch {batch_idx + 1} after attempt {attempt}: {exc}")
                    raise exc

        # Polite pause between batches to stay safely below the 30k tokens/minute limit
        if pause_between_batches > 0 and (batch_idx + 1) < total_batches:
            time.sleep(pause_between_batches)

    if current_vs is None:
        raise RuntimeError("Vector store initialization failed: no documents were processed.")

    return current_vs


def generate_document_id(doc: Document) -> str:
    """
    Generates a deterministic unique ID for a transcript chunk based on its
    video_id, chunk_index, start timestamp, and content hash.
    Ensures idempotent indexing and prevents duplicate records.
    """
    video_id = str(doc.metadata.get("video_id", "unknown_video"))
    chunk_index = doc.metadata.get("chunk_index", "0")
    start_ts = doc.metadata.get("start", 0.0)
    content = doc.page_content.strip()

    raw_signature = f"{video_id}:{chunk_index}:{start_ts}:{content}"
    digest = hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()[:16]
    return f"{video_id}_{digest}"


def create_vectorstore(
    documents: List[Document],
    embedding_model: Optional[Embeddings] = None,
    save: bool = False,
    folder_path: Optional[str | Path] = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> FAISS:
    """
    Generates embeddings and builds a new FAISS vector store from transcript Documents.

    Args:
        documents: Non-empty list of LangChain Document objects from Phase 2.
        embedding_model: Optional LangChain Embeddings instance; defaults to GeminiEmbeddings.
        save: If True, saves the created vector store to disk immediately.
        folder_path: Target directory to save the index (defaults to backend/vector_db/).
        index_name: Filename prefix for the FAISS index files (default: 'index').

    Returns:
        FAISS: In-memory populated LangChain FAISS vector store.

    Raises:
        TypeError: If documents is not a list.
        ValueError: If documents list is empty or contains invalid items.
    """
    if not isinstance(documents, list):
        raise TypeError(f"Expected documents as a list, got {type(documents).__name__}")

    if not documents:
        raise ValueError("Cannot create a vector store from an empty documents list.")

    for i, doc in enumerate(documents):
        if not isinstance(doc, Document):
            raise ValueError(
                f"Item at index {i} is not a LangChain Document instance ({type(doc).__name__})."
            )
        if not doc.page_content or not doc.page_content.strip():
            raise ValueError(f"Document at index {i} has empty page_content.")

    embeddings = embedding_model or get_embedding_model()

    # Generate deterministic IDs for all chunks to support deduplication
    doc_ids = [generate_document_id(d) for d in documents]

    logger.info(f"Creating FAISS vector store with {len(documents)} documents (safe-batching with 429 auto-retry)...")
    vectorstore = _embed_and_add_in_batches(
        vectorstore=None,
        documents=documents,
        ids=doc_ids,
        embeddings=embeddings,
    )

    if save:
        target_dir = folder_path or DEFAULT_VECTOR_DB_DIR
        save_vectorstore(vectorstore, folder_path=target_dir, index_name=index_name)

    return vectorstore


def save_vectorstore(
    vectorstore: FAISS,
    folder_path: Optional[str | Path] = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> Path:
    """
    Persists a FAISS vector store index and docstore locally to disk.

    Args:
        vectorstore: Instantiated FAISS vector store.
        folder_path: Directory where index files will be stored.
        index_name: Base filename (generates <index_name>.faiss and <index_name>.pkl).

    Returns:
        Path: Resolved directory path where the index was saved.
    """
    if not isinstance(vectorstore, FAISS):
        raise TypeError(f"Expected a FAISS vectorstore instance, got {type(vectorstore).__name__}")

    save_dir = Path(folder_path) if folder_path else DEFAULT_VECTOR_DB_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(save_dir), index_name=index_name)
    logger.info(f"FAISS index successfully saved to: {save_dir}")
    return save_dir


def load_vectorstore(
    folder_path: Optional[str | Path] = None,
    embedding_model: Optional[Embeddings] = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> FAISS:
    """
    Loads an existing local FAISS vector store index from disk.

    Security Note on Pickle Deserialization:
        LangChain's FAISS module stores document metadata in a pickled docstore
        file (`<index_name>.pkl`). Loading pickle files can execute arbitrary
        code if loaded from untrusted external sources.
        `allow_dangerous_deserialization=True` is enabled here specifically because
        the index files are generated, stored, and managed locally within our
        trusted application workspace (`backend/vector_db/`).

    Args:
        folder_path: Directory containing the FAISS index files.
        embedding_model: Embeddings instance matching the index configuration.
        index_name: Base filename prefix (default: 'index').

    Returns:
        FAISS: Loaded FAISS vector store.

    Raises:
        FileNotFoundError: If the index folder or files do not exist.
    """
    load_dir = Path(folder_path) if folder_path else DEFAULT_VECTOR_DB_DIR

    faiss_file = load_dir / f"{index_name}.faiss"
    pkl_file = load_dir / f"{index_name}.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(
            f"FAISS index files not found in '{load_dir}'. "
            f"Expected '{index_name}.faiss' and '{index_name}.pkl'. "
            "Please create and save an index before attempting to load."
        )

    embeddings = embedding_model or get_embedding_model()

    vectorstore = FAISS.load_local(
        folder_path=str(load_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name=index_name,
    )
    logger.info(f"FAISS index successfully loaded from: {load_dir}")
    return vectorstore


def add_documents_to_vectorstore(
    vectorstore: FAISS,
    documents: List[Document],
    save_after: bool = False,
    folder_path: Optional[str | Path] = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> int:
    """
    Adds new Documents to an existing FAISS vector store while preventing duplicate
    records and avoiding unnecessary re-embedding of already indexed chunks.

    Args:
        vectorstore: Target FAISS vector store to append to.
        documents: List of candidate LangChain Documents.
        save_after: If True, persists the updated index to disk.
        folder_path: Directory path for saving if save_after is True.
        index_name: Index name prefix for saving.

    Returns:
        int: Number of new documents actually added (excluding duplicates).
    """
    if not isinstance(vectorstore, FAISS):
        raise TypeError(f"Expected a FAISS instance, got {type(vectorstore).__name__}")
    if not isinstance(documents, list):
        raise TypeError(f"Expected documents as a list, got {type(documents).__name__}")

    if not documents:
        return 0

    # Retrieve existing document IDs from the internal docstore
    existing_ids: Set[str] = set(getattr(vectorstore.docstore, "_dict", {}).keys())

    new_docs: List[Document] = []
    new_ids: List[str] = []

    for doc in documents:
        if not isinstance(doc, Document) or not doc.page_content.strip():
            continue
        doc_id = generate_document_id(doc)
        if doc_id not in existing_ids and doc_id not in new_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)

    if not new_docs:
        logger.info("All provided documents are already indexed. No new embeddings generated.")
        return 0

    logger.info(f"Adding {len(new_docs)} new unique documents to FAISS vector store (safe-batching with 429 auto-retry)...")
    _embed_and_add_in_batches(
        vectorstore=vectorstore,
        documents=new_docs,
        ids=new_ids,
        embeddings=getattr(vectorstore, "embedding_function", None) or get_embedding_model(),
    )

    if save_after:
        target_dir = folder_path or DEFAULT_VECTOR_DB_DIR
        save_vectorstore(vectorstore, folder_path=target_dir, index_name=index_name)

    return len(new_docs)


def get_vectorstore_stats(vectorstore: FAISS) -> Dict[str, Any]:
    """
    Returns diagnostic statistics about a FAISS vector store.
    """
    if not isinstance(vectorstore, FAISS):
        raise TypeError(f"Expected a FAISS instance, got {type(vectorstore).__name__}")

    doc_dict = getattr(vectorstore.docstore, "_dict", {})
    total_docs = len(doc_dict)

    video_ids: Set[str] = set()
    for doc in doc_dict.values():
        if isinstance(doc, Document):
            vid = doc.metadata.get("video_id")
            if vid:
                video_ids.add(str(vid))

    index_dim = vectorstore.index.d if hasattr(vectorstore.index, "d") else None
    ntotal = vectorstore.index.ntotal if hasattr(vectorstore.index, "ntotal") else total_docs

    return {
        "total_documents": total_docs,
        "index_vectors_count": ntotal,
        "vector_dimension": index_dim,
        "indexed_video_ids": list(video_ids),
    }

