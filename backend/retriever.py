"""
backend/retriever.py
Phase 4: Semantic Retrieval Module
Retrieves the most relevant YouTube transcript chunks from FAISS vector store
for a natural-language query while preserving temporal and source metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from backend.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

# Default top-k retrieval count for Phase 4
DEFAULT_K: int = 4


def retrieve_relevant_chunks(
    query: str,
    vectorstore: Optional[FAISS] = None,
    k: int = DEFAULT_K,
    folder_path: Optional[str | Path] = None,
    include_score: bool = True,
) -> List[Document]:
    """
    Retrieves the top-k most relevant transcript chunk Documents for a query.

    Args:
        query: Natural-language question or search phrase.
        vectorstore: Instantiated FAISS vector store; if None, loads from local disk.
        k: Number of relevant chunks to retrieve (default: 4).
        folder_path: Path to local FAISS index if vectorstore is not provided.
        include_score: When True, adds 'similarity_score' into doc.metadata.

    Returns:
        List[Document]: Top-k retrieved LangChain Documents sorted by relevance,
        with all metadata (video_id, video_title, start, end, duration) preserved.

    Raises:
        ValueError: If query is empty or k is non-positive.
    """
    results_with_scores = retrieve_relevant_chunks_with_scores(
        query=query,
        vectorstore=vectorstore,
        k=k,
        folder_path=folder_path,
    )

    documents: List[Document] = []
    for doc, score in results_with_scores:
        if include_score:
            # Attach similarity distance score to metadata for transparency and inspection
            doc.metadata["similarity_score"] = float(score)
        documents.append(doc)

    return documents


def retrieve_relevant_chunks_with_scores(
    query: str,
    vectorstore: Optional[FAISS] = None,
    k: int = DEFAULT_K,
    folder_path: Optional[str | Path] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieves the top-k most relevant chunks along with their raw similarity scores.

    In FAISS with L2 index:
        Lower score = smaller Euclidean distance = higher semantic similarity.
        (For normalized Gemini vectors, L2 distance is directly monotonically
        related to cosine distance: dist^2 = 2 - 2*cos_sim).

    Args:
        query: Natural-language query string.
        vectorstore: FAISS vector store instance (loaded from disk if omitted).
        k: Number of top documents to return.
        folder_path: Local directory path if vectorstore needs to be loaded.

    Returns:
        List[Tuple[Document, float]]: Pairs of (Document, score) sorted by relevance.

    Raises:
        ValueError: If query is empty or k is non-positive.
    """
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    store = vectorstore if vectorstore is not None else load_vectorstore(folder_path=folder_path)

    # Perform similarity search with score
    results = store.similarity_search_with_score(query.strip(), k=k)
    return results


class TranscriptRetriever:
    """
    Configurable semantic retriever for YouTube transcript chunks.
    Wraps FAISS index operations, score extraction, and metadata verification.
    """

    def __init__(
        self,
        vectorstore: Optional[FAISS] = None,
        k: int = DEFAULT_K,
        folder_path: Optional[str | Path] = None,
        embedding_model: Optional[Embeddings] = None,
    ) -> None:
        """
        Initializes TranscriptRetriever.

        Args:
            vectorstore: Instantiated FAISS store, or loads from disk if None.
            k: Default number of documents to retrieve per query (default: 4).
            folder_path: Path to local FAISS index folder if loading from disk.
            embedding_model: Embedding model for query encoding.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self.k = k
        self.vectorstore = (
            vectorstore
            if vectorstore is not None
            else load_vectorstore(
                folder_path=folder_path,
                embedding_model=embedding_model,
            )
        )

    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieves top-k Documents for the query, including similarity_score in metadata.
        """
        target_k = k if k is not None else self.k
        return retrieve_relevant_chunks(
            query=query,
            vectorstore=self.vectorstore,
            k=target_k,
            include_score=True,
        )

    def retrieve_with_scores(
        self, query: str, k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k (Document, score) pairs for the query.
        """
        target_k = k if k is not None else self.k
        return retrieve_relevant_chunks_with_scores(
            query=query,
            vectorstore=self.vectorstore,
            k=target_k,
        )

    def as_langchain_retriever(self, k: Optional[int] = None) -> VectorStoreRetriever:
        """
        Converts to a standard LangChain VectorStoreRetriever for chain compatibility.
        """
        target_k = k if k is not None else self.k
        return self.vectorstore.as_retriever(search_kwargs={"k": target_k})

