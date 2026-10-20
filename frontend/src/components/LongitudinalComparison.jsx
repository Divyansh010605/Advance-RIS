import React, { useState, useEffect, useRef } from 'react';

/**
 * LongitudinalComparison — Phase 3 side-by-side study comparison.
 *
 * Allows users to load two analysis results (current + prior) and
 * compare disease probabilities, with delta arrows and a sync-scroll
 * image viewer panel.
 */
export default function LongitudinalComparison({ currentResult, priorResult }) {
  const [syncScroll, setSyncScroll] = useState(true);

  if (!currentResult || !priorResult) {
    return (
      <div
        id="longitudinal-comparison"
        style={{
          background: '#0f172a',
          border: '1px solid #1e293b',
          borderRadius: 12,
          padding: 24,
          textAlign: 'center',
          color: '#475569',
        }}
      >
        <div style={{ fontSize: '1.4rem', marginBottom: 8 }}>📋</div>
        <p style={{ margin: 0, fontSize: '0.875rem' }}>
          Load a prior study from History to enable longitudinal comparison.
        </p>
      </div>
    );
  }

  const currentPreds = currentResult.predictions || [];
  const priorPreds = priorResult.predictions || [];

  // Build a merged comparison map keyed by disease
  const priorMap = {};
  priorPreds.forEach((p) => { priorMap[p.disease] = p; });

  const comparisons = currentPreds.map((curr) => {
    const prior = priorMap[curr.disease];
    const delta = prior ? curr.probability - prior.probability : null;
    return { ...curr, prior, delta };
  });

  const getArrow = (delta) => {
    if (delta == null) return '—';
    if (delta > 0.05) return '▲';
    if (delta < -0.05) return '▼';
    return '↔';
  };
  const getArrowColor = (delta) => {
    if (delta == null) return '#64748b';
    if (delta > 0.05) return '#ef4444';
    if (delta < -0.05) return '#22c55e';
    return '#94a3b8';
  };

  const currentDate = currentResult?.metadata?.study_date || 'Current';
  const priorDate = priorResult?.metadata?.study_date || 'Prior';

  return (
    <div id="longitudinal-comparison" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700, color: '#e2e8f0' }}>
          Longitudinal Comparison
        </h3>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem', color: '#94a3b8', cursor: 'pointer' }}>
          <input type="checkbox" checked={syncScroll} onChange={(e) => setSyncScroll(e.target.checked)} />
          Sync scroll
        </label>
      </div>

      {/* Side-by-side image panels */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {[
          { label: `Prior — ${priorDate}`, result: priorResult },
          { label: `Current — ${currentDate}`, result: currentResult },
        ].map(({ label, result }) => (
          <div key={label} style={{ background: '#000', borderRadius: 8, border: '1px solid #1e293b', overflow: 'hidden' }}>
            <div style={{ background: '#0f172a', padding: '5px 10px', fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600 }}>
              {label}
            </div>
            {result?.imageUrl ? (
              <img src={result.imageUrl} alt={label} style={{ width: '100%', display: 'block', objectFit: 'contain', maxHeight: 280 }} />
            ) : (
              <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#334155', fontSize: '0.8rem' }}>
                No image
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Comparison table */}
      <div style={{ overflowX: 'auto' }}>
        <table
          id="comparison-table"
          style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}
          aria-label="Longitudinal disease probability comparison table"
        >
          <thead>
            <tr style={{ background: '#0f172a' }}>
              <th style={thStyle('left')}>Disease</th>
              <th style={thStyle()}>Prior</th>
              <th style={thStyle()}>Current</th>
              <th style={thStyle()}>Delta</th>
              <th style={thStyle()}>Status</th>
            </tr>
          </thead>
          <tbody>
            {comparisons.map((c, idx) => {
              const rowBg = idx % 2 === 0 ? '#18181b' : '#1c1c1f';
              const isNewDetection = c.detected && c.prior && !c.prior.detected;
              const isResolved = !c.detected && c.prior && c.prior.detected;
              return (
                <tr key={c.disease} style={{ background: rowBg }}>
                  <td style={{ ...cellStyle('left'), fontWeight: 600, color: c.detected ? '#f87171' : '#e2e8f0' }}>
                    {c.disease}
                    {isNewDetection && <span style={badgeStyle('#ef4444')}> NEW</span>}
                    {isResolved && <span style={badgeStyle('#22c55e')}> RESOLVED</span>}
                  </td>
                  <td style={cellStyle()}>
                    {c.prior ? (
                      <span>
                        <span style={{ color: c.prior.detected ? '#f87171' : '#10b981', fontWeight: 600 }}>
                          {(c.prior.probability * 100).toFixed(1)}%
                        </span>
                      </span>
                    ) : (
                      <span style={{ color: '#475569' }}>—</span>
                    )}
                  </td>
                  <td style={cellStyle()}>
                    <span style={{ color: c.detected ? '#f87171' : '#10b981', fontWeight: 600 }}>
                      {(c.probability * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td style={{ ...cellStyle(), color: getArrowColor(c.delta), fontWeight: 700 }}>
                    {getArrow(c.delta)}{' '}
                    {c.delta != null && (
                      <span style={{ fontSize: '0.72rem', fontWeight: 400 }}>
                        {c.delta > 0 ? '+' : ''}{(c.delta * 100).toFixed(1)}%
                      </span>
                    )}
                  </td>
                  <td style={cellStyle()}>
                    <span
                      style={{
                        fontSize: '0.68rem',
                        fontWeight: 700,
                        padding: '1px 6px',
                        borderRadius: 8,
                        background: c.detected ? '#ef444422' : '#22c55e22',
                        color: c.detected ? '#f87171' : '#4ade80',
                      }}
                    >
                      {c.detected ? 'Detected' : 'Normal'}
                    </span>
                    {c.is_uncertain && (
                      <span style={{ marginLeft: 4, fontSize: '0.65rem', color: '#f59e0b' }}>⚠</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function thStyle(align = 'center') {
  return {
    padding: '8px 10px',
    textAlign: align,
    color: '#94a3b8',
    fontWeight: 600,
    borderBottom: '1px solid #1e293b',
    whiteSpace: 'nowrap',
  };
}

function cellStyle(align = 'center') {
  return {
    padding: '7px 10px',
    textAlign: align,
    borderBottom: '1px solid #18181b',
    verticalAlign: 'middle',
  };
}

function badgeStyle(color) {
  return {
    fontSize: '0.6rem',
    fontWeight: 700,
    padding: '1px 5px',
    borderRadius: 6,
    background: color + '22',
    color,
    border: `1px solid ${color}55`,
    marginLeft: 5,
    verticalAlign: 'middle',
  };
}
