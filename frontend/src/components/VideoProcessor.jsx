import React, { useState } from 'react';

export default function VideoProcessor({ onProcess, processing, activeVideo }) {
  const [url, setUrl] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!url.trim()) return;
    onProcess(url.trim());
  };

  return (
    <div className="video-processor">
      <form onSubmit={handleSubmit} className="processor-form">
        <div className="input-wrapper">
          <svg className="input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z"></path>
            <polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02"></polygon>
          </svg>
          <input
            type="text"
            className="video-input"
            placeholder="Paste YouTube Video URL (e.g., https://www.youtube.com/watch?v=aircAruvnKk)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={processing}
          />
        </div>

        <button type="submit" className="btn-primary" disabled={processing || !url.trim()}>
          {processing ? (
            <>
              <span className="spinner"></span>
              <span>Processing...</span>
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="16 16 12 12 8 16"></polyline>
                <line x1="12" y1="12" x2="12" y2="21"></line>
                <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"></path>
              </svg>
              <span>Process Video</span>
            </>
          )}
        </button>
      </form>

      {activeVideo && (
        <div className="video-info-card">
          <div className="video-meta-main">
            <span className="video-title">{activeVideo.title}</span>
            <div className="video-badges">
              <span className="badge">ID: {activeVideo.video_id}</span>
              <span className="badge">Lang: {activeVideo.language}</span>
              <span className="badge">{activeVideo.chunk_count} Chunks</span>
              <span className="badge badge-green">✓ Indexed & Ready</span>
            </div>
          </div>
          <a
            href={`https://www.youtube.com/watch?v=${activeVideo.video_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="source-link"
          >
            Open on YouTube ↗
          </a>
        </div>
      )}
    </div>
  );
}

