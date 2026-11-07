import React from 'react';
import SourceCard from './SourceCard';

export default function MessageItem({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`message-row ${message.role}`}>
      <div className={`avatar ${message.role}`}>
        {isUser ? 'U' : 'AI'}
      </div>

      <div className="bubble-content">
        <div className={`message-bubble ${message.role}`}>
          <p style={{ whiteSpace: 'pre-wrap' }}>{message.content}</p>
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="sources-container">
            <div className="sources-header">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              <span>Supporting Sources ({message.sources.length} Retrieved Chunks)</span>
            </div>

            <div className="sources-grid">
              {message.sources.map((src, idx) => (
                <SourceCard key={idx} source={src} index={idx} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

