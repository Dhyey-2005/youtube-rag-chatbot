"""
backend/chatbot.py
Phase 5: LLM Integration & Answer Generation with Source Attribution
Implements a conversational RAG chatbot using ChatGoogleGenerativeAI,
FAISS semantic retrieval, precise YouTube timestamp URL generation,
and conversational history support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai.chat_models import GoogleModelNotFoundError
except ImportError:
    ChatGoogleGenerativeAI = None  # type: ignore
    GoogleModelNotFoundError = Exception  # type: ignore

from backend.embeddings import _load_api_key
from backend.retriever import DEFAULT_K, TranscriptRetriever
from backend.vectorstore import DEFAULT_VECTOR_DB_DIR, load_vectorstore

logger = logging.getLogger(__name__)

# Initial model specified for Phase 5
DEFAULT_CHAT_MODEL: str = "gemini-2.5-flash"
# Active fallback model if gemini-2.5-flash is retired / unavailable for this API key
FALLBACK_CHAT_MODEL: str = "gemini-3.5-flash-lite"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_HISTORY_TURNS: int = 5

SYSTEM_PROMPT = """You are a precise, factual assistant answering questions about a YouTube video.

STRICT INSTRUCTIONS:
1. Answer the question using ONLY the provided video transcript context below.
2. If the provided context does not contain sufficient information to answer the question, state explicitly:
   "I cannot answer this question based on the provided video transcript."
3. Never guess, assume, or bring in outside knowledge not explicitly stated in the transcript.
4. Never invent or hallucinate facts, timestamps, or sources.
5. You may refer to the conversation history to understand context or conversational references, but NEVER treat past conversation as factual video evidence—only the Transcript Context constitutes verified facts from the video.
"""


class SourceAttribution(TypedDict, total=False):
    video_title: str
    video_id: str
    start: float
    end: float
    duration: float
    relevant_excerpt: str
    url: str
    similarity_score: float


class ChatbotResponse(TypedDict):
    answer: str
    sources: List[SourceAttribution]
    retrieved_chunks: List[Dict[str, Any]]
    conversation_history: List[Dict[str, str]]


def format_youtube_timestamp_url(video_id: str, start_seconds: float) -> str:
    """
    Generates a clickable YouTube timestamp URL in the exact required format:
    https://www.youtube.com/watch?v=VIDEO_ID&t=START_SECONDSs
    """
    start_seconds_int = max(0, int(round(start_seconds)))
    return f"https://www.youtube.com/watch?v={video_id}&t={start_seconds_int}s"


def extract_text_from_response(content: Any) -> str:
    """
    Safely extracts string content from LangChain / Google GenAI message content,
    handling string or list of text block structures.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts).strip()
    return str(content).strip()


def get_chat_model(
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """
    Initializes ChatGoogleGenerativeAI with secure API key loading and temperature=0.

    Args:
        model: Gemini model identifier (default: 'gemini-2.5-flash').
        temperature: Sampling temperature (default: 0.0 for deterministic factual responses).
        api_key: Optional explicit key; if None, loaded securely from .env.
        **kwargs: Additional parameters for ChatGoogleGenerativeAI.

    Returns:
        ChatGoogleGenerativeAI: Configured chat model instance.
    """
    if ChatGoogleGenerativeAI is None:
        raise ImportError(
            "langchain-google-genai is not installed in the current environment."
        )

    resolved_key = _load_api_key(api_key)
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=resolved_key,
        **kwargs,
    )


def _invoke_with_fallback(
    llm: ChatGoogleGenerativeAI,
    messages: List[BaseMessage | Tuple[str, str]],
) -> Any:
    """
    Invokes the LLM, seamlessly falling back to FALLBACK_CHAT_MODEL if
    DEFAULT_CHAT_MODEL is not available for this API key.
    """
    try:
        return llm.invoke(messages)
    except (GoogleModelNotFoundError, Exception) as exc:
        err_msg = str(exc)
        # Check if model is deprecated / 404
        if "404" in err_msg or "not found" in err_msg.lower() or "no longer available" in err_msg.lower():
            logger.warning(
                f"Model '{llm.model}' returned 404 / unavailable. Falling back to '{FALLBACK_CHAT_MODEL}'."
            )
            fallback_llm = ChatGoogleGenerativeAI(
                model=FALLBACK_CHAT_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                google_api_key=llm.google_api_key,
            )
            return fallback_llm.invoke(messages)
        raise


def reformulate_query_with_history(
    query: str,
    history: List[Dict[str, str]],
    llm: ChatGoogleGenerativeAI,
) -> str:
    """
    Uses conversation history to reformulate an ambiguous follow-up question
    into a standalone search query for FAISS retrieval.
    """
    if not history:
        return query.strip()

    history_text_parts: List[str] = []
    for turn in history[-4:]:  # Use last 2-3 turns for context
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        history_text_parts.append(f"{role}: {content}")
    history_str = "\n".join(history_text_parts)

    reformulate_prompt = f"""Given the conversation history and a follow-up question about a video, rephrase the follow-up question into an independent, standalone search query that includes all necessary context (e.g. resolving pronouns like 'it', 'they', 'them' or vague references).
If the follow-up question is already self-contained, return it unchanged.
Do NOT answer the question. Only return the rephrased standalone search query.

Conversation History:
{history_str}

Follow-up Question: {query}
Standalone Search Query:"""

    try:
        response = _invoke_with_fallback(llm, [HumanMessage(content=reformulate_prompt)])
        standalone = extract_text_from_response(response.content)
        return standalone if standalone else query.strip()
    except Exception as exc:
        logger.warning(f"Query reformulation failed: {exc}. Using original query.")
        return query.strip()


class YouTubeChatbot:
    """
    Conversational RAG Chatbot for YouTube transcripts.
    Integrates FAISS retrieval, Gemini LLM answer generation, source attribution,
    and conversational history support.
    """

    def __init__(
        self,
        vectorstore: Optional[FAISS] = None,
        retriever: Optional[TranscriptRetriever] = None,
        model: str = DEFAULT_CHAT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
        folder_path: Optional[str | Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initializes the YouTubeChatbot.

        Args:
            vectorstore: Instantiated FAISS store; if None and retriever is None, loaded from disk.
            retriever: Instantiated TranscriptRetriever; if None, constructed from vectorstore.
            model: Gemini model identifier (default: 'gemini-2.5-flash').
            temperature: Sampling temperature (default: 0.0).
            max_history_turns: Maximum conversation turns to retain in state.
            folder_path: Path to FAISS vector db if loading from disk.
            api_key: Optional Gemini API key; loaded from .env if None.
        """
        if retriever is not None:
            self.retriever = retriever
        else:
            store = vectorstore if vectorstore is not None else load_vectorstore(folder_path=folder_path)
            self.retriever = TranscriptRetriever(vectorstore=store, k=DEFAULT_K)

        self.model_name = model
        self.temperature = temperature
        self.max_history_turns = max_history_turns
        self.llm = get_chat_model(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
        )

        # Internal conversation history buffer: list of {"role": "user"|"assistant", "content": "..."}
        self.conversation_history: List[Dict[str, str]] = []

    def _invoke_llm(
        self, messages: List[BaseMessage | Tuple[str, str]]
    ) -> Any:
        """
        Invokes the LLM with automatic fallback to FALLBACK_CHAT_MODEL if
        the configured model is deprecated or unavailable for this API key.
        """
        try:
            return self.llm.invoke(messages)
        except (GoogleModelNotFoundError, Exception) as exc:
            err_msg = str(exc)
            if "404" in err_msg or "not found" in err_msg.lower() or "no longer available" in err_msg.lower():
                logger.warning(
                    f"Model '{self.llm.model}' is unavailable. Switching to '{FALLBACK_CHAT_MODEL}'."
                )
                self.llm = ChatGoogleGenerativeAI(
                    model=FALLBACK_CHAT_MODEL,
                    temperature=self.temperature,
                    google_api_key=self.llm.google_api_key,
                )
                return self.llm.invoke(messages)
            raise

    def _reformulate_query(
        self, query: str, history: List[Dict[str, str]]
    ) -> str:
        """
        Uses conversation history to reformulate an ambiguous follow-up question
        into a standalone search query for FAISS retrieval.
        """
        if not history:
            return query.strip()

        history_text_parts: List[str] = []
        for turn in history[-4:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            history_text_parts.append(f"{role}: {content}")
        history_str = "\n".join(history_text_parts)

        reformulate_prompt = f"""Given the conversation history and a follow-up question about a video, rephrase the follow-up question into an independent, standalone search query that includes all necessary context (e.g. resolving pronouns like 'it', 'they', 'them' or vague references).
If the follow-up question is already self-contained, return it unchanged.
Do NOT answer the question. Only return the rephrased standalone search query.

Conversation History:
{history_str}

Follow-up Question: {query}
Standalone Search Query:"""

        try:
            response = self._invoke_llm([HumanMessage(content=reformulate_prompt)])
            standalone = extract_text_from_response(response.content)
            return standalone if standalone else query.strip()
        except Exception as exc:
            logger.warning(f"Query reformulation failed: {exc}. Using original query.")
            return query.strip()

    def ask(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        k: int = DEFAULT_K,
    ) -> ChatbotResponse:
        """
        Answers a user question using the full RAG pipeline with source attribution.

        Workflow:
        1. Query Reformulation (if conversation history exists).
        2. FAISS Semantic Retrieval (top-k relevant chunks with scores).
        3. Contextual Answer Generation via Gemini LLM.
        4. Source Attribution with clickable YouTube timestamp URLs.
        5. Conversation History Update.

        Args:
            question: User's question string.
            conversation_history: Optional external conversation history list;
                                  if None, uses chatbot's internal history.
            k: Number of relevant chunks to retrieve (default: 4).

        Returns:
            ChatbotResponse dict with:
                - answer: Generated answer text
                - sources: List of source attribution dicts with timestamp links
                - retrieved_chunks: Full metadata and content of retrieved chunks
                - conversation_history: Updated conversation history list
        """
        if not question or not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        clean_question = question.strip()

        # Determine history to use
        active_history = (
            list(conversation_history)
            if conversation_history is not None
            else list(self.conversation_history)
        )

        # Step 1: Reformulate query for retrieval if history exists
        search_query = clean_question
        if active_history:
            search_query = self._reformulate_query(clean_question, active_history)

        # Step 2: Retrieve top-k relevant chunks with similarity scores
        retrieved_results = self.retriever.retrieve_with_scores(search_query, k=k)

        # Step 3: Format Context and Sources from retrieved chunks
        context_blocks: List[str] = []
        sources: List[SourceAttribution] = []
        retrieved_chunks: List[Dict[str, Any]] = []

        for idx, (doc, score) in enumerate(retrieved_results, start=1):
            meta = doc.metadata
            v_id = str(meta.get("video_id", "unknown"))
            v_title = str(meta.get("video_title", "Unknown Title"))
            start_ts = float(meta.get("start", 0.0))
            end_ts = float(meta.get("end", 0.0))
            dur = float(meta.get("duration", round(end_ts - start_ts, 2)))
            timestamp_url = format_youtube_timestamp_url(v_id, start_ts)

            context_blocks.append(
                f"[Source #{idx}] (Video: '{v_title}' | Start: {start_ts:.2f}s, End: {end_ts:.2f}s)\n"
                f"{doc.page_content}"
            )

            source_item: SourceAttribution = {
                "video_title": v_title,
                "video_id": v_id,
                "start": start_ts,
                "end": end_ts,
                "duration": dur,
                "relevant_excerpt": doc.page_content.strip(),
                "url": timestamp_url,
                "similarity_score": round(float(score), 4),
            }
            sources.append(source_item)

            chunk_info = dict(meta)
            chunk_info["content"] = doc.page_content
            chunk_info["similarity_score"] = float(score)
            retrieved_chunks.append(chunk_info)

        context_str = "\n\n".join(context_blocks)

        # Format history block for answer generation
        history_blocks: List[str] = []
        for turn in active_history[-self.max_history_turns :]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            history_blocks.append(f"{role}: {content}")
        history_section = (
            f"\nRecent Conversation History:\n" + "\n".join(history_blocks) + "\n"
            if history_blocks
            else ""
        )

        user_prompt = (
            f"Transcript Context:\n{context_str}\n"
            f"{history_section}\n"
            f"Question: {clean_question}\n"
            f"Answer:"
        )

        # Step 4: Invoke Gemini LLM for answer generation
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        llm_response = self._invoke_llm(messages)
        raw_answer = extract_text_from_response(llm_response.content)

        # Step 5: Update conversation history
        active_history.append({"role": "user", "content": clean_question})
        active_history.append({"role": "assistant", "content": raw_answer})

        # Trim history to max allowed turns (user + assistant pairs)
        max_entries = self.max_history_turns * 2
        if len(active_history) > max_entries:
            active_history = active_history[-max_entries:]

        # Update chatbot internal history if managing own state
        if conversation_history is None:
            self.conversation_history = list(active_history)

        return {
            "answer": raw_answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "conversation_history": active_history,
        }

    def clear_history(self) -> None:
        """Clears the conversational history buffer."""
        self.conversation_history = []


def ask_question(
    question: str,
    vectorstore: Optional[FAISS] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    k: int = DEFAULT_K,
    model: str = DEFAULT_CHAT_MODEL,
    folder_path: Optional[str | Path] = None,
    api_key: Optional[str] = None,
) -> ChatbotResponse:
    """
    Convenience function to ask a single question against the RAG pipeline.

    Args:
        question: User query string.
        vectorstore: Optional preloaded FAISS store (loaded from disk if None).
        conversation_history: Optional list of past conversation turns.
        k: Top-k chunks to retrieve (default: 4).
        model: Gemini model identifier.
        folder_path: Path to FAISS index.
        api_key: Gemini API key.

    Returns:
        ChatbotResponse dict.
    """
    bot = YouTubeChatbot(
        vectorstore=vectorstore,
        model=model,
        folder_path=folder_path,
        api_key=api_key,
    )
    return bot.ask(
        question=question,
        conversation_history=conversation_history,
        k=k,
    )
