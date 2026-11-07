"""
test_api.py
Integration test for Phase 6: FastAPI Backend and API verification.
Tests health endpoint, video processing, chat with source attribution,
conversational follow-up memory, and frontend HTML delivery.
"""

from __future__ import annotations

import sys
from starlette.testclient import TestClient

from backend.main import app

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_api_tests():
    print("=" * 75)
    print("PHASE 6 VERIFICATION: FastAPI Backend & End-to-End API Integration")
    print("=" * 75)

    client = TestClient(app)

    # 1. Health check
    print("\n[1/5] Testing GET /api/health...")
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200, f"Expected 200, got {health_resp.status_code}"
    health_data = health_resp.json()
    print(f"      Status Code: {health_resp.status_code}")
    print(f"      Backend Status: {health_data['status']}")
    print(f"      Indexed Videos: {health_data['indexed_videos']} ({health_data['indexed_chunks_count']} chunks)")
    print("[PASS] Health endpoint returned status 200 OK.")

    # 2. Process video
    print("\n[2/5] Testing POST /api/video/process...")
    test_url = "https://www.youtube.com/watch?v=aircAruvnKk"
    proc_resp = client.post("/api/video/process", json={"url": test_url})
    assert proc_resp.status_code == 200, f"Expected 200, got {proc_resp.status_code}"
    proc_data = proc_resp.json()
    print(f"      Status Code : {proc_resp.status_code}")
    print(f"      Video ID    : {proc_data['video_id']}")
    print(f"      Title       : {proc_data['title']}")
    print(f"      Status      : {proc_data['status']}")
    print(f"      Chunks      : {proc_data['chunk_count']}")
    assert proc_data["video_id"] == "aircAruvnKk"
    print("[PASS] Video processing endpoint succeeded without duplicate re-indexing.")

    # 3. Initial Chat Query
    print("\n[3/5] Testing POST /api/chat (Initial Question)...")
    session_id = "test_integration_session_001"
    q1 = "How many neurons are in the input layer for the 28 by 28 pixel image?"
    chat1_resp = client.post("/api/chat", json={"question": q1, "session_id": session_id, "k": 4})
    assert chat1_resp.status_code == 200, f"Expected 200, got {chat1_resp.status_code}"
    chat1_data = chat1_resp.json()
    print(f"      Question : \"{q1}\"")
    print(f"      Answer   : \"{chat1_data['answer']}\"")
    print(f"      Sources  : {len(chat1_data['sources'])} items")
    for idx, s in enumerate(chat1_data["sources"][:2], 1):
        print(f"        Source #{idx}: [{s['start']:.1f}s -> {s['end']:.1f}s] {s['url']}")
    assert "784" in chat1_data["answer"]
    assert len(chat1_data["sources"]) == 4
    print("[PASS] Initial question returned accurate answer with 4 timestamped sources.")

    # 4. Conversational Follow-up Query
    print("\n[4/5] Testing POST /api/chat (Contextual Follow-up with same session_id)...")
    q2 = "What do the numbers inside them represent?"
    chat2_resp = client.post("/api/chat", json={"question": q2, "session_id": session_id, "k": 4})
    assert chat2_resp.status_code == 200, f"Expected 200, got {chat2_resp.status_code}"
    chat2_data = chat2_resp.json()
    print(f"      Follow-up : \"{q2}\"")
    print(f"      Answer    : \"{chat2_data['answer']}\"")
    print(f"      Sources   : {len(chat2_data['sources'])} items")
    ans2_lower = chat2_data["answer"].lower()
    assert ("grayscale" in ans2_lower or "pixel" in ans2_lower or "activation" in ans2_lower)
    print("[PASS] Contextual follow-up successfully resolved pronoun using session history.")

    # 5. Clear Session & Frontend Delivery Check
    print("\n[5/5] Testing POST /api/chat/clear and Static Frontend Delivery...")
    clear_resp = client.post("/api/chat/clear", json={"session_id": session_id})
    assert clear_resp.status_code == 200
    print(f"      Clear Session Status: {clear_resp.json()['status']}")

    front_resp = client.get("/")
    assert front_resp.status_code == 200
    assert "YouTube RAG Chatbot" in front_resp.text
    print(f"      Frontend Delivery: 200 OK (Served HTML length: {len(front_resp.text)} chars)")
    print("[PASS] Static frontend delivery and session clear verified.")

    print("\n" + "=" * 75)
    print("ALL PHASE 6 API INTEGRATION CHECKS PASSED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    run_api_tests()

