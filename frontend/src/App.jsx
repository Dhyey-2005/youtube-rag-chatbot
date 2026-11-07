import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import VideoProcessor from './components/VideoProcessor';
import ChatWindow from './components/ChatWindow';
import {
  checkBackendHealth,
  processYouTubeVideo,
  postChatMessage,
  clearChatSession,
} from './api/client';

export default function App() {
  const [backendHealth, setBackendHealth] = useState(null);
  const [activeVideo, setActiveVideo] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`);

  // Initial health check on mount
  useEffect(() => {
    async function initHealth() {
      try {
        const data = await checkBackendHealth();
        setBackendHealth(data);
        if (data.indexed_videos_count > 0 && data.indexed_videos && data.indexed_videos.length > 0) {
          // If a video is already indexed on backend, display it as active
          setActiveVideo({
            video_id: data.indexed_videos[0],
            title: 'Indexed Video Session',
            language: 'Available',
            chunk_count: data.indexed_chunks_count,
          });
        }
      } catch (err) {
        console.warn('Backend initial health check warning:', err);
      }
    }
    initHealth();
  }, []);

  const handleProcessVideo = async (url) => {
    setProcessing(true);
    setError(null);
    setSuccessMsg(null);
    // Clear chat messages and generate fresh session whenever a new video is uploaded
    setMessages([]);
    const freshSession = `session_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    setSessionId(freshSession);

    try {
      const result = await processYouTubeVideo(url);
      setActiveVideo({
        video_id: result.video_id,
        title: result.title,
        language: result.language,
        chunk_count: result.chunk_count,
      });

      setSuccessMsg(
        result.status === 'already_indexed'
          ? `"${result.title}" is ready. Chat history cleared.`
          : `Processed "${result.title}" into ${result.chunk_count} chunks. All previous history cleared!`
      );

      // Refresh health diagnostics
      const health = await checkBackendHealth();
      setBackendHealth(health);
    } catch (err) {
      setError(err.message || 'Error processing YouTube video.');
    } finally {
      setProcessing(false);
    }
  };

  const handleSendMessage = async (question) => {
    if (!question.trim()) return;

    setError(null);
    const userMessage = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await postChatMessage(question, sessionId);
      const aiMessage = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources || [],
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      setError(err.message || 'Failed to get an answer from the chatbot.');
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = async () => {
    try {
      await clearChatSession(sessionId);
    } catch (e) {
      console.warn('Error clearing backend session:', e);
    }
    setMessages([]);
    setError(null);
    setSuccessMsg('Chat history cleared.');
    setTimeout(() => setSuccessMsg(null), 3000);
  };

  return (
    <div className="app-container">
      <Header
        backendStatus={backendHealth?.status ? 'API Online' : 'Connecting...'}
        totalChunks={backendHealth?.indexed_chunks_count || 0}
      />

      {error && (
        <div className="alert-banner alert-error">
          <span>⚠️ {error}</span>
          <button
            onClick={() => setError(null)}
            style={{ background: 'none', border: 'none', color: '#fca5a5', cursor: 'pointer', fontWeight: 'bold' }}
          >
            ✕
          </button>
        </div>
      )}

      {successMsg && (
        <div className="alert-banner alert-success">
          <span>✓ {successMsg}</span>
          <button
            onClick={() => setSuccessMsg(null)}
            style={{ background: 'none', border: 'none', color: '#6ee7b7', cursor: 'pointer', fontWeight: 'bold' }}
          >
            ✕
          </button>
        </div>
      )}

      <VideoProcessor
        onProcess={handleProcessVideo}
        processing={processing}
        activeVideo={activeVideo}
      />

      <ChatWindow
        messages={messages}
        onSendMessage={handleSendMessage}
        onClearChat={handleClearChat}
        loading={loading}
        hasVideo={Boolean(activeVideo)}
      />
    </div>
  );
}

