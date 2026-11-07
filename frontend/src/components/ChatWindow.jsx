import React, { useState, useRef, useEffect } from 'react';
import MessageItem from './MessageItem';

export default function ChatWindow({
  messages,
  onSendMessage,
  onClearChat,
  loading,
  hasVideo,
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const starterQuestions = [
    "How many neurons are in the input layer for the 28 by 28 pixel image?",
    "Explain the overall layered architecture of the network.",
    "What do the numbers inside the neurons represent?",
  ];

  return (
    <div className="chat-window">
      <div className="chat-header">
        <h2>Conversation History</h2>
        {messages.length > 0 && (
          <button onClick={onClearChat} className="btn-secondary" title="Clear conversation history">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
            <span>Clear Chat</span>
          </button>
        )}
      </div>

      <div className="messages-list">
        {messages.length === 0 ? (
          <div className="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <p>
              {hasVideo
                ? "Ask any question about the video transcript below:"
                : "Enter a YouTube video URL above and click 'Process Video' to begin."}
            </p>

            {hasVideo && (
              <div className="starter-chips">
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                  SUGGESTED QUESTIONS:
                </span>
                {starterQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    className="chip-btn"
                    onClick={() => onSendMessage(q)}
                  >
                    "{q}"
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map((msg, idx) => <MessageItem key={idx} message={msg} />)
        )}

        {loading && (
          <div className="message-row assistant">
            <div className="avatar assistant">AI</div>
            <div className="message-bubble assistant" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="spinner"></span>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                Searching transcript & generating answer...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input-bar">
        <input
          type="text"
          className="chat-input"
          placeholder={hasVideo ? "Ask a question or follow-up about the video..." : "Process a video first to ask questions"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading || !hasVideo}
        />
        <button
          type="submit"
          className="btn-primary"
          disabled={loading || !input.trim() || !hasVideo}
        >
          {loading ? (
            <span className="spinner"></span>
          ) : (
            <>
              <span>Send</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </>
          )}
        </button>
      </form>
    </div>
  );
}

