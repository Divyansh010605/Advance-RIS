import React from 'react';

export default function MultiDiseaseResults({ predictions = [] }) {
  if (!predictions.length) return null;

  return (
    <div className="card mb-4" id="multi-disease-results">
      <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '16px' }}>
        Multi-Disease Pathology Screening
      </h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '16px' }}>
        {predictions.map((p) => {
          const isDetected = p.detected;
          const probPct = (p.probability * 100).toFixed(1);
          const threshPct = (p.threshold * 100).toFixed(1);

          return (
            <div
              key={p.disease}
              style={{
                background: isDetected ? 'rgba(239, 68, 68, 0.08)' : 'var(--color-bg-alt, #18181b)',
                border: `1px solid ${isDetected ? '#ef4444' : '#27272a'}`,
                borderRadius: '8px',
                padding: '14px',
                display: 'flex',
                flexDirection: 'column',
                justify: 'space-between',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontWeight: 700, fontSize: '0.95rem', color: isDetected ? '#f87171' : 'var(--color-text)' }}>
                  {p.disease}
                </span>
                <span
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: '12px',
                    background: isDetected ? '#ef4444' : '#22c55e',
                    color: '#ffffff',
                  }}
                >
                  {isDetected ? 'Detected' : 'Normal'}
                </span>
              </div>

              <div style={{ margin: '8px 0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#9ca3af', marginBottom: '4px' }}>
                  <span>Probability</span>
                  <span style={{ fontWeight: 600, color: isDetected ? '#ef4444' : '#10b981' }}>{probPct}%</span>
                </div>
                <div style={{ height: '6px', background: '#27272a', borderRadius: '3px', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.min(100, Math.max(0, p.probability * 100))}%`,
                      background: isDetected ? '#ef4444' : '#10b981',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
                <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: '4px', textAlign: 'right' }}>
                  Cutoff: {threshPct}%
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
