import React, { useState } from 'react';

/**
 * UncertaintyBadge — renders a colored badge for uncertainty level.
 */
function UncertaintyBadge({ uncertainty, isUncertain }) {
  if (uncertainty == null) return null;
  const level = uncertainty < 0.04 ? 'low' : uncertainty < 0.08 ? 'medium' : 'high';
  const colors = { low: '#22c55e', medium: '#f59e0b', high: '#ef4444' };
  const labels = { low: 'Low σ', medium: 'Med σ', high: 'High σ' };
  return (
    <span
      title={`Predictive uncertainty (σ): ${uncertainty.toFixed(4)}${isUncertain ? ' — Requires radiologist review' : ''}`}
      style={{
        fontSize: '0.65rem',
        fontWeight: 700,
        padding: '1px 6px',
        borderRadius: '8px',
        background: colors[level] + '22',
        color: colors[level],
        border: `1px solid ${colors[level]}55`,
        marginLeft: 4,
        cursor: 'help',
      }}
    >
      {labels[level]}
    </span>
  );
}

/**
 * ConsensusBadge — renders a badge showing how many ensemble models agree.
 */
function ConsensusBadge({ consensus, modelsAgreeing = [] }) {
  if (!consensus && (!modelsAgreeing || !modelsAgreeing.length)) return null;
  const text = consensus || `${modelsAgreeing.length}/3 Agree`;
  const isHigh = text.startsWith('3/3');
  const isMed = text.startsWith('2/3');

  const bgColor = isHigh ? 'rgba(16, 185, 129, 0.15)' : isMed ? 'rgba(245, 158, 11, 0.15)' : 'rgba(100, 116, 139, 0.15)';
  const borderColor = isHigh ? 'rgba(16, 185, 129, 0.4)' : isMed ? 'rgba(245, 158, 11, 0.4)' : 'rgba(100, 116, 139, 0.3)';
  const textColor = isHigh ? '#34d399' : isMed ? '#fbbf24' : '#94a3b8';

  return (
    <span
      title={`Model consensus: ${text}${modelsAgreeing.length ? ` (${modelsAgreeing.join(', ')})` : ''}`}
      style={{
        fontSize: '0.65rem',
        fontWeight: 700,
        padding: '2px 6px',
        borderRadius: '6px',
        background: bgColor,
        border: `1px solid ${borderColor}`,
        color: textColor,
        whiteSpace: 'nowrap',
      }}
    >
      {text}
    </span>
  );
}

/**
 * ConfidenceInterval — renders a small CI bar beneath the probability bar.
 */
function ConfidenceInterval({ ci, probability }) {
  if (!ci || ci.length < 2) return null;
  const lo = Math.round(ci[0] * 100);
  const hi = Math.round(ci[1] * 100);
  const midPct = Math.round(probability * 100);
  return (
    <div style={{ marginTop: 3 }}>
      <div style={{ position: 'relative', height: 4, background: '#3f3f46', borderRadius: 2, overflow: 'hidden' }}>
        <div
          style={{
            position: 'absolute',
            left: `${lo}%`,
            width: `${hi - lo}%`,
            height: '100%',
            background: 'rgba(59,130,246,0.4)',
            borderRadius: 2,
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: `${midPct - 0.5}%`,
            width: '1%',
            height: '100%',
            background: '#3b82f6',
          }}
        />
      </div>
      <div style={{ fontSize: '0.65rem', color: '#6b7280', textAlign: 'right', marginTop: 2 }}>
        95% CI: [{lo}%, {hi}%]
      </div>
    </div>
  );
}

/**
 * Normalize model names for display
 */
function formatModelName(key) {
  const lower = key.toLowerCase();
  if (lower.includes('swin')) return 'SwinCheX';
  if (lower.includes('dense')) return 'DenseNet-121';
  if (lower.includes('res')) return 'ResNet-50';
  return key;
}

/**
 * MultiDiseaseResults — Enhanced pathology card grid with model attribution.
 * Shows ensemble probability bars, per-model predictions, consensus, uncertainty, and CI.
 */
export default function MultiDiseaseResults({ predictions = [], onSelectDisease }) {
  const [sortMode, setSortMode] = useState('probability'); // 'probability' | 'name' | 'detected'
  const [showModelBreakdown, setShowModelBreakdown] = useState(true);

  if (!predictions.length) return null;

  const uncertainCount = predictions.filter((p) => p.is_uncertain).length;
  const detectedCount = predictions.filter((p) => p.detected).length;

  const sorted = [...predictions].sort((a, b) => {
    if (sortMode === 'probability') return b.probability - a.probability;
    if (sortMode === 'name') return a.disease.localeCompare(b.disease);
    if (sortMode === 'detected') return Number(b.detected) - Number(a.detected);
    return 0;
  });

  return (
    <div className="card mb-4" id="multi-disease-results">
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0 }}>
            Multi-Disease Pathology Screening
          </h2>
          <div style={{ fontSize: '0.78rem', color: '#9ca3af', marginTop: 2 }}>
            <span style={{ color: detectedCount > 0 ? '#f87171' : '#22c55e', fontWeight: 600 }}>
              {detectedCount} detected
            </span>
            {' · '}
            {predictions.length} screened
            {uncertainCount > 0 && (
              <>
                {' · '}
                <span style={{ color: '#f59e0b', fontWeight: 600 }}>{uncertainCount} uncertain</span>
              </>
            )}
          </div>
        </div>

        {/* View toggles & Sort controls */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          <button
            onClick={() => setShowModelBreakdown(!showModelBreakdown)}
            style={{
              padding: '3px 10px',
              borderRadius: 6,
              border: `1px solid ${showModelBreakdown ? '#10b981' : '#27272a'}`,
              background: showModelBreakdown ? '#10b98118' : 'transparent',
              color: showModelBreakdown ? '#34d399' : '#9ca3af',
              fontSize: '0.75rem',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            {showModelBreakdown ? '✓ Model Breakdown' : '+ Model Breakdown'}
          </button>

          <span style={{ color: '#3f3f46', fontSize: '0.8rem' }}>|</span>

          {['probability', 'detected', 'name'].map((mode) => (
            <button
              key={mode}
              onClick={() => setSortMode(mode)}
              style={{
                padding: '3px 10px',
                borderRadius: 6,
                border: `1px solid ${sortMode === mode ? '#3b82f6' : '#27272a'}`,
                background: sortMode === mode ? '#3b82f620' : 'transparent',
                color: sortMode === mode ? '#3b82f6' : '#9ca3af',
                fontSize: '0.75rem',
                cursor: 'pointer',
                textTransform: 'capitalize',
              }}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {/* Uncertain banner */}
      {uncertainCount > 0 && (
        <div
          role="alert"
          style={{
            background: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.4)',
            borderRadius: 8,
            padding: '8px 12px',
            marginBottom: 12,
            fontSize: '0.8rem',
            color: '#fbbf24',
          }}
        >
          ⚠ {uncertainCount} finding(s) have high predictive uncertainty — mandatory radiologist review recommended.
        </div>
      )}

      {/* Card grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))',
          gap: 12,
        }}
      >
        {sorted.map((p) => {
          const isDetected = p.detected;
          const isUncertain = p.is_uncertain;
          const probPct = (p.probability * 100).toFixed(1);
          const threshPct = (p.threshold * 100).toFixed(1);

          const borderColor = isUncertain ? '#f59e0b' : isDetected ? '#ef4444' : '#27272a';
          const bgColor = isUncertain
            ? 'rgba(245,158,11,0.06)'
            : isDetected
            ? 'rgba(239,68,68,0.07)'
            : 'var(--color-bg-alt, #18181b)';

          const modelPreds = p.model_predictions || {};
          const modelKeys = Object.keys(modelPreds);

          return (
            <div
              key={p.disease}
              id={`disease-card-${p.disease.toLowerCase().replace(/\s+/g, '-')}`}
              style={{
                background: bgColor,
                border: `1.5px solid ${borderColor}`,
                borderRadius: 10,
                padding: '13px 14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
                textAlign: 'left',
                position: 'relative',
              }}
            >
              {/* Disease name + badges row */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 6 }}>
                <div>
                  <span
                    style={{
                      fontWeight: 700,
                      fontSize: '0.90rem',
                      color: isDetected ? '#f87171' : 'var(--color-text)',
                      cursor: onSelectDisease ? 'pointer' : 'default',
                    }}
                    onClick={() => onSelectDisease && onSelectDisease(p.disease)}
                    title={onSelectDisease ? `Click to view ${p.disease} CAM` : p.disease}
                  >
                    {p.disease}
                  </span>
                  {p.uncertainty != null && (
                    <div style={{ marginTop: 2 }}>
                      <UncertaintyBadge uncertainty={p.uncertainty} isUncertain={isUncertain} />
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 3 }}>
                  <span
                    style={{
                      fontSize: '0.68rem',
                      fontWeight: 700,
                      padding: '2px 7px',
                      borderRadius: 10,
                      background: isDetected ? '#ef4444' : '#22c55e',
                      color: '#fff',
                      flexShrink: 0,
                    }}
                  >
                    {isDetected ? 'Detected' : 'Normal'}
                  </span>
                  {p.consensus && (
                    <ConsensusBadge consensus={p.consensus} modelsAgreeing={p.models_agreeing} />
                  )}
                </div>
              </div>

              {/* Ensemble Probability bar */}
              <div style={{ marginTop: 4 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#9ca3af', marginBottom: 3 }}>
                  <span>Ensemble Confidence</span>
                  <span style={{ fontWeight: 700, color: isDetected ? '#ef4444' : '#10b981' }}>{probPct}%</span>
                </div>
                <div style={{ height: 6, background: '#27272a', borderRadius: 3, overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.min(100, Math.max(0, p.probability * 100))}%`,
                      background: isDetected ? 'linear-gradient(90deg, #dc2626, #f87171)' : 'linear-gradient(90deg, #059669, #10b981)',
                      transition: 'width 0.4s ease',
                      borderRadius: 3,
                    }}
                  />
                </div>
                {/* Threshold marker */}
                <div style={{ position: 'relative', height: 0 }}>
                  <div
                    title={`Threshold: ${threshPct}%`}
                    style={{
                      position: 'absolute',
                      left: `${Math.min(99, p.threshold * 100)}%`,
                      top: -8,
                      width: 1,
                      height: 8,
                      background: '#6b7280',
                    }}
                  />
                </div>
                <div style={{ fontSize: '0.67rem', color: '#6b7280', marginTop: 5, textAlign: 'right' }}>
                  Cutoff: {threshPct}%
                </div>
              </div>

              {/* 95% CI bar */}
              <ConfidenceInterval ci={p.confidence_interval} probability={p.probability} />

              {/* Per-Model Breakdown (DenseNet-121, SwinCheX, ResNet-50) */}
              {showModelBreakdown && modelKeys.length > 0 && (
                <div
                  style={{
                    marginTop: 6,
                    paddingTop: 6,
                    borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                    fontSize: '0.70rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginBottom: 4, fontWeight: 600 }}>
                    <span>Model Breakdown</span>
                    <span>Prob / Vote</span>
                  </div>

                  {modelKeys.map((k) => {
                    const prob = modelPreds[k];
                    const probNum = typeof prob === 'number' ? prob : 0;
                    const isPositive = probNum >= p.threshold;
                    const displayName = formatModelName(k);

                    return (
                      <div
                        key={k}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '2px 0',
                        }}
                      >
                        <span style={{ color: '#cbd5e1' }}>{displayName}</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                          <span style={{ fontWeight: 600, color: isPositive ? '#f87171' : '#a1a1aa' }}>
                            {(probNum * 100).toFixed(1)}%
                          </span>
                          <span
                            title={isPositive ? `${displayName} detected ${p.disease}` : `${displayName} did not detect`}
                            style={{
                              display: 'inline-block',
                              width: 7,
                              height: 7,
                              borderRadius: '50%',
                              backgroundColor: isPositive ? '#ef4444' : '#52525b',
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* View CAM Button */}
              {onSelectDisease && (
                <button
                  onClick={() => onSelectDisease(p.disease)}
                  style={{
                    marginTop: 4,
                    padding: '4px 8px',
                    borderRadius: 6,
                    border: '1px solid #3f3f46',
                    background: 'transparent',
                    color: '#93c5fd',
                    fontSize: '0.70rem',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 4,
                  }}
                >
                  <span>👁 View Grad-CAM</span>
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
