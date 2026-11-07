"""
backend/embeddings.py
Phase 3: Gemini Embedding Generation Module
Generates dense vector embeddings for YouTube transcript chunks using GoogleGenerativeAIEmbeddings.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    GoogleGenerativeAIEmbeddings = None  # type: ignore

logger = logging.getLogger(__name__)

# Phase 3 model requirement
DEFAULT_EMBEDDING_MODEL: str = "gemini-embedding-001"
DEFAULT_BATCH_SIZE: int = 20
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_DELAY: float = 2.0


def _load_api_key(explicit_key: Optional[str] = None) -> str:
    """
    Loads and validates the GOOGLE_API_KEY from environment or .env file.
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # Load from .env file (current working directory or parent dirs)
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please define GOOGLE_API_KEY in your .env file "
            "or set it in your environment variables."
        )
    return api_key.strip()


def get_embedding_model(
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> GoogleGenerativeAIEmbeddings:
    """
    Creates and returns a configured GoogleGenerativeAIEmbeddings instance.

    Args:
        model: Embedding model name (default: 'gemini-embedding-001').
        api_key: Optional explicit API key; if omitted, loaded from .env.
        **kwargs: Additional parameters for GoogleGenerativeAIEmbeddings.

    Returns:
        GoogleGenerativeAIEmbeddings: Configured LangChain embedding model.
    """
    if GoogleGenerativeAIEmbeddings is None:
        raise ImportError(
            "langchain-google-genai is not installed in the current environment."
        )

    resolved_key = _load_api_key(api_key)
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=resolved_key,
        **kwargs,
    )


def validate_embeddings(
    embeddings: List[List[float]],
    documents: List[Document],
    expected_dimension: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validates that embeddings match input documents and contain valid numeric vectors.

    Args:
        embeddings: List of embedding vectors.
        documents: List of input Document objects.
        expected_dimension: Optional expected vector dimension to assert against.

    Returns:
        Dict[str, Any]: Summary of validation results.

    Raises:
        ValueError: If validation checks fail (count mismatch, empty vectors, NaN/Inf).
    """
    if not isinstance(embeddings, list):
        raise TypeError(f"embeddings must be a list, got {type(embeddings).__name__}")
    if not isinstance(documents, list):
        raise TypeError(f"documents must be a list, got {type(documents).__name__}")

    num_docs = len(documents)
    num_embeddings = len(embeddings)

    if num_docs != num_embeddings:
        raise ValueError(
            f"Embedding count ({num_embeddings}) does not match input document count ({num_docs})."
        )

    if num_docs == 0:
        return {
            "valid": True,
            "count": 0,
            "dimension": 0,
            "nan_count": 0,
            "inf_count": 0,
        }

    first_dim = len(embeddings[0])
    if first_dim == 0:
        raise ValueError("First embedding vector is empty (dimension 0).")

    target_dim = expected_dimension if expected_dimension is not None else first_dim

    for idx, vec in enumerate(embeddings):
        if not isinstance(vec, list):
            raise TypeError(
                f"Embedding at index {idx} is not a list ({type(vec).__name__})."
            )
        if len(vec) == 0:
            raise ValueError(f"Embedding at index {idx} is empty.")
        if len(vec) != target_dim:
            raise ValueError(
                f"Inconsistent vector dimension at index {idx}: expected {target_dim}, got {len(vec)}."
            )

        # Check for NaN or Inf
        for val_idx, val in enumerate(vec):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"Non-numeric value at vector {idx}, position {val_idx}: {val}"
                )
            if math.isnan(val):
                raise ValueError(
                    f"NaN detected in embedding vector at index {idx}, position {val_idx}."
                )
            if math.isinf(val):
                raise ValueError(
                    f"Infinite value detected in embedding vector at index {idx}, position {val_idx}."
                )

    return {
        "valid": True,
        "count": num_embeddings,
        "dimension": target_dim,
        "nan_count": 0,
        "inf_count": 0,
    }


class GeminiEmbedder:
    """
    Reusable embedder service for generating vector embeddings from LangChain Documents
    using Gemini models with batching, validation, and retry mechanisms.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        **kwargs: Any,
    ) -> None:
        self.model_name = model
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self.retry_delay = retry_delay

        self.embedding_model = get_embedding_model(
            model=self.model_name,
            api_key=api_key,
            **kwargs,
        )

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generates dense embeddings for a list of LangChain Document objects.
        Validates documents, processes them in safe batches, and validates the output vectors.

        Args:
            documents: List of LangChain Document objects from Phase 2 text splitting.

        Returns:
            List[List[float]]: List of embedding vectors corresponding 1:1 with input documents.

        Raises:
            ValueError: If input documents list is empty or contains invalid items.
            RuntimeError: If embedding generation fails after retries.
        """
        if not isinstance(documents, list):
            raise TypeError(
                f"Expected a list of Document objects, got {type(documents).__name__}"
            )

        if not documents:
            raise ValueError("Input documents list cannot be empty.")

        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise ValueError(
                    f"Item at index {i} is not a LangChain Document instance ({type(doc).__name__})."
                )
            if not doc.page_content or not doc.page_content.strip():
                raise ValueError(
                    f"Document at index {i} has empty page_content."
                )

        # Snapshot documents to verify immutability
        original_contents = [d.page_content for d in documents]
        original_metadatas = [dict(d.metadata) for d in documents]

        texts = [d.page_content for d in documents]
        all_embeddings: List[List[float]] = []

        # Process in batches to handle rate limits and payload boundaries smoothly
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_vectors)

        # Validate that the original documents remain unmodified
        for idx, doc in enumerate(documents):
            if doc.page_content != original_contents[idx]:
                raise RuntimeError(
                    f"Document at index {idx} was mutated during embedding generation."
                )
            if doc.metadata != original_metadatas[idx]:
                raise RuntimeError(
                    f"Document metadata at index {idx} was mutated during embedding generation."
                )

        # Validate output embeddings
        validate_embeddings(all_embeddings, documents)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string for retrieval tasks.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        retries = 0
        last_error: Optional[Exception] = None

        while retries < self.max_retries:
            try:
                vector = self.embedding_model.embed_query(query.strip())
                if not vector or not isinstance(vector, list):
                    raise ValueError("Received invalid or empty query embedding vector.")
                return vector
            except Exception as exc:
                last_error = exc
                retries += 1
                if retries < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (retries - 1))
                    logger.warning(
                        f"Query embedding attempt {retries} failed: {exc}. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to embed query after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _embed_batch_with_retry(self, batch_texts: List[str]) -> List[List[float]]:
        """
        Embeds a single batch of texts with exponential backoff on transient errors.
        """
        retries = 0
        last_error: Optional[Exception] = None

        while retries < self.max_retries:
            try:
                vectors = self.embedding_model.embed_documents(batch_texts)
                if len(vectors) != len(batch_texts):
                    raise RuntimeError(
                        f"Batch size mismatch: sent {len(batch_texts)} texts, received {len(vectors)} vectors."
                    )
                return vectors
            except Exception as exc:
                last_error = exc
                retries += 1
                if retries < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (retries - 1))
                    logger.warning(
                        f"Embedding batch attempt {retries} failed: {exc}. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to generate embeddings after {self.max_retries} attempts: {last_error}"
        ) from last_error


def generate_document_embeddings(
    documents: List[Document],
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs: Any,
) -> List[List[float]]:
    """
    Functional helper to generate embeddings for a list of LangChain Document chunks.

    Args:
        documents: List of LangChain Document objects from Phase 2.
        model: Gemini embedding model name (default: 'gemini-embedding-001').
        api_key: Optional Gemini API key; loaded from .env if None.
        batch_size: Texts per batch request.
        **kwargs: Extra parameters passed to GeminiEmbedder.

    Returns:
        List[List[float]]: Embedding vectors for all documents.
    """
    embedder = GeminiEmbedder(
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        **kwargs,
    )
    return embedder.embed_documents(documents)

