import React from 'react';

export default function Header({ backendStatus, totalChunks }) {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-logo">
          <svg viewBox="0 0 24 24">
            <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" />
          </svg>
        </div>
        <div className="header-title-group">
          <h1>YouTube RAG Chatbot</h1>
          <p>Conversational Q&A with Timestamp Source Attribution</p>
        </div>
      </div>

      <div className="health-badge">
        <span className="status-dot"></span>
        <span>{backendStatus || 'Backend Connected'}</span>
        {totalChunks > 0 && <span>• {totalChunks} Chunks Indexed</span>}
      </div>
    </header>
  );
}

