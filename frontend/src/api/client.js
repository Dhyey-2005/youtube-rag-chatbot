/**
 * API Client for interacting with the FastAPI Backend.
 * Uses environment variable VITE_API_URL if configured, otherwise defaults to relative '/api'.
 * Includes 'ngrok-skip-browser-warning' to bypass interstitial check on ngrok free tier.
 */

const BASE_URL = import.meta.env.VITE_API_URL || '';

const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
  'ngrok-skip-browser-warning': '1',
};

export async function checkBackendHealth() {
  const resp = await fetch(`${BASE_URL}/api/health`, {
    headers: {
      'ngrok-skip-browser-warning': '1',
    },
  });
  if (!resp.ok) {
    throw new Error(`Health check failed with status ${resp.status}`);
  }
  return resp.json();
}

export async function processYouTubeVideo(videoUrl) {
  const resp = await fetch(`${BASE_URL}/api/video/process`, {
    method: 'POST',
    headers: DEFAULT_HEADERS,
    body: JSON.stringify({ url: videoUrl }),
  });

  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || 'Failed to process YouTube video.');
  }
  return data;
}

export async function postChatMessage(question, sessionId = 'default_session', k = 4) {
  const resp = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    headers: DEFAULT_HEADERS,
    body: JSON.stringify({
      question,
      session_id: sessionId,
      k,
    }),
  });

  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || 'Failed to generate answer.');
  }
  return data;
}

export async function clearChatSession(sessionId = 'default_session') {
  const resp = await fetch(`${BASE_URL}/api/chat/clear`, {
    method: 'POST',
    headers: DEFAULT_HEADERS,
    body: JSON.stringify({ session_id: sessionId }),
  });
  return resp.json();
}
