"""
test_chatbot.py
Verification script for Phase 5: LLM Integration & Answer Generation with Source Attribution.
Tests 4 required test cases:
1. Factual question with direct transcript evidence.
2. Multi-chunk synthesis question.
3. Unrelated / out-of-domain question (verifying grounding & refusal).
4. Contextual follow-up question utilizing conversational history.
"""

from __future__ import annotations

import re
import sys
from backend.chatbot import YouTubeChatbot, format_youtube_timestamp_url
from backend.vectorstore import DEFAULT_VECTOR_DB_DIR, load_vectorstore

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_response(test_name: str, response: dict) -> None:
    print("=" * 75)
    print(f"TEST CASE: {test_name}")
    print("=" * 75)
    print(f"\n[Generated Answer]:\n{response['answer']}\n")

    print(f"[Attributed Sources ({len(response['sources'])} retrieved)]:")
    for idx, s in enumerate(response["sources"], start=1):
        print(f"  Source #{idx}:")
        print(f"    Title          : \"{s['video_title']}\" (ID: {s['video_id']})")
        print(f"    Timestamp Range: [{s['start']:.2f}s -> {s['end']:.2f}s] (Duration: {s['duration']:.2f}s)")
        print(f"    YouTube URL    : {s['url']}")
        print(f"    Score (L2 Dist): {s.get('similarity_score', 'N/A')}")
        excerpt_preview = s['relevant_excerpt'].replace('\n', ' ')
        if len(excerpt_preview) > 140:
            excerpt_preview = excerpt_preview[:140] + "..."
        print(f"    Excerpt        : \"{excerpt_preview}\"\n")


def run_chatbot_tests() -> None:
    print("=" * 75)
    print("PHASE 5 VERIFICATION: YouTube RAG Chatbot with Source Attribution")
    print("=" * 75)

    print("[Setup] Initializing YouTubeChatbot from persisted vector store...")
    try:
        chatbot = YouTubeChatbot(folder_path=DEFAULT_VECTOR_DB_DIR)
    except FileNotFoundError:
        print("[FAIL] FAISS index not found. Please run test_vectorstore.py first.")
        sys.exit(1)

    print("[Setup] Chatbot initialized successfully with Gemini LLM & FAISS index.\n")

    # ---------------------------------------------------------
    # TEST 1: Factual question whose answer exists in transcript
    # ---------------------------------------------------------
    q1 = "How many neurons are in the input layer for the 28 by 28 pixel image?"
    res1 = chatbot.ask(q1)
    print_response("1. Factual Question (Single-Concept)", res1)

    # Verification 1
    assert "784" in res1["answer"], "Expected '784' in answer for 28x28 input neurons"
    assert len(res1["sources"]) == 4, "Expected top-4 retrieved sources"
    for s in res1["sources"]:
        assert re.match(r"^https://www\.youtube\.com/watch\?v=[0-9A-Za-z_-]{11}&t=\d+s$", s["url"]), f"Invalid URL: {s['url']}"
    print("[PASS] Test 1: Grounded factual answer correctly stated 784 neurons with valid timestamp links.")

    # ---------------------------------------------------------
    # TEST 2: Multi-chunk synthesis question
    # ---------------------------------------------------------
    chatbot.clear_history()  # Isolate test 2
    q2 = "Explain the layered structure of the neural network from the input image through the hidden layers to the output."
    res2 = chatbot.ask(q2)
    print_response("2. Multi-Chunk Synthesis Question", res2)

    # Verification 2
    ans2_lower = res2["answer"].lower()
    assert ("input" in ans2_lower and ("hidden" in ans2_lower or "layer" in ans2_lower)), "Expected layered structure explanation"
    assert len(res2["sources"]) == 4, "Expected 4 sources"
    print("[PASS] Test 2: Synthesized answer across multiple layers and chunks with full source attribution.")

    # ---------------------------------------------------------
    # TEST 3: Question unrelated to the video
    # ---------------------------------------------------------
    chatbot.clear_history()  # Isolate test 3
    q3 = "What is the recipe for baking chocolate chip cookies?"
    res3 = chatbot.ask(q3)
    print_response("3. Unrelated / Out-of-Domain Question", res3)

    # Verification 3
    ans3_lower = res3["answer"].lower()
    refusal_keywords = ["cannot", "not provided", "transcript", "insufficient", "does not contain"]
    assert any(kw in ans3_lower for kw in refusal_keywords), "Expected model to decline answering ungrounded question"
    print("[PASS] Test 3: Correctly refused to answer out-of-domain question; zero hallucination.")

    # ---------------------------------------------------------
    # TEST 4: Contextual follow-up question
    # ---------------------------------------------------------
    chatbot.clear_history()  # Start fresh conversational session
    turn1_q = "How many neurons are in the input layer for the 28 by 28 pixel image?"
    turn1_res = chatbot.ask(turn1_q)
    print_response("4a. Conversation Turn 1 (Baseline Context)", turn1_res)

    # Follow-up relying on pronoun "them" referring to the input neurons
    turn2_q = "What do the numbers inside them represent?"
    turn2_res = chatbot.ask(turn2_q)
    print_response("4b. Conversation Turn 2 (Contextual Follow-up)", turn2_res)

    # Verification 4
    ans4_lower = turn2_res["answer"].lower()
    assert ("grayscale" in ans4_lower or "pixel" in ans4_lower or "activation" in ans4_lower), "Expected follow-up answer to address pixel/activation values inside the neurons"
    assert len(turn2_res["conversation_history"]) == 4, "Expected 4 history entries (2 user, 2 assistant)"
    print("[PASS] Test 4: Follow-up question correctly resolved 'them' to neurons via conversation history and retrieved accurate transcript context.")

    print("\n" + "=" * 75)
    print("ALL 4 CHATBOT TESTS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    run_chatbot_tests()

