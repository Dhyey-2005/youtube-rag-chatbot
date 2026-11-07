import React, { useState } from 'react';

export default function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);
  const startSec = Math.round(source.start);
  const endSec = Math.round(source.end);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div className="source-card">
      <div className="source-top">
        <span className="timestamp-pill">
          ⏱ {formatTime(startSec)} - {formatTime(endSec)} ({source.start}s - {source.end}s)
        </span>
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="source-link"
          title="Open video at this timestamp"
        >
          ▶ Watch at {startSec}s
        </a>
      </div>

      <div className="source-excerpt">
        "{expanded ? source.relevant_excerpt : `${source.relevant_excerpt.slice(0, 140)}...`}"
        {source.relevant_excerpt.length > 140 && (
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              background: 'none',
              border: 'none',
              color: '#60a5fa',
              cursor: 'pointer',
              marginLeft: '4px',
              fontSize: '0.75rem',
            }}
          >
            {expanded ? 'Show less' : 'Read more'}
          </button>
        )}
      </div>
    </div>
  );
}

