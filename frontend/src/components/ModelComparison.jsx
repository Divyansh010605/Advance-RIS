import React, { useState, useMemo } from 'react';

/**
 * ModelComparison — Phase 3 Tri-Model Architecture Benchmark & Attribution View.
 * Compares DenseNet-121, SwinCheX, and ResNet-50 side-by-side:
 *  - Prediction probabilities & agreement metrics
 *  - Multi-backbone LayerCAM visual localization comparison
 *  - Consensus analysis
 */
export default function ModelComparison({
  predictions = [],
  heatmaps = [],
  previewUrl = '',
}) {
  // Find detected diseases
  const detectedDiseases = useMemo(() => {
    const list = predictions.filter((p) => p.detected).map((p) => p.disease);
    if (list.length > 0) return list;
    // Fallback if normal: top probability disease
    if (predictions.length > 0) {
      const top = [...predictions].sort((a, b) => b.probability - a.probability)[0];
      return [top.disease];
    }
    return [];
  }, [predictions]);

  const [activeDisease, setActiveDisease] = useState(detectedDiseases[0] || 'Atelectasis');

  // Ensure activeDisease stays valid
  const currentDisease = detectedDiseases.includes(activeDisease)
    ? activeDisease
    : detectedDiseases[0] || '';

  // Get active prediction object
  const activePred = useMemo(() => {
    return predictions.find((p) => p.disease === currentDisease) || null;
  }, [predictions, currentDisease]);

  // Model predictions for active disease with robust prefix lookup
  const modelPreds = activePred?.model_predictions || {};
  const getProb = (prefix) => {
    for (const [k, v] of Object.entries(modelPreds)) {
      if (k.toLowerCase().includes(prefix.toLowerCase())) {
        return typeof v === 'number' ? v : parseFloat(v) || 0;
      }
    }
    return null;
  };

  const denseProb = getProb('dense') ?? activePred?.probability ?? 0;
  const swinProb = getProb('swin') ?? activePred?.probability ?? 0;
  const resProb = getProb('res') ?? activePred?.probability ?? 0;
  const ensembleProb = activePred?.probability ?? 0;
  const threshold = activePred?.threshold ?? 0.45;

  // Find heatmaps for active disease
  const diseaseHeatmaps = useMemo(() => {
    return heatmaps.filter((h) => h.disease === currentDisease);
  }, [heatmaps, currentDisease]);

  const findCAM = (prefix) => {
    return diseaseHeatmaps.find((h) => {
      const m = (h.model_name || '').toLowerCase();
      return m.includes(prefix.toLowerCase());
    });
  };

  const ensembleCAM = findCAM('ensemble') || diseaseHeatmaps.find((h) => !h.model_name) || diseaseHeatmaps[0];
  const swinCAM = findCAM('swin');
  const denseCAM = findCAM('dense') || diseaseHeatmaps.find((h) => !h.model_name);
  const resCAM = findCAM('res');

  const modelsMeta = [
    {
      key: 'swin',
      name: 'SwinCheX',
      arch: 'Vision Transformer (Swin-Large)',
      params: '197M parameters',
      prob: swinProb,
      detected: swinProb >= threshold,
      cam: swinCAM,
      color: '#38bdf8',
      desc: 'Self-attention across shifted local windows for global contextual pattern recognition.',
    },
    {
      key: 'densenet121',
      name: 'DenseNet-121',
      arch: 'Densely Connected CNN',
      params: '7.0M parameters',
      prob: denseProb,
      detected: denseProb >= threshold,
      cam: denseCAM,
      color: '#f59e0b',
      desc: 'Iterative feature reuse across dense blocks for high spatial feature retention.',
    },
    {
      key: 'resnet50',
      name: 'ResNet-50',
      arch: 'Deep Residual Network',
      params: '25.6M parameters',
      prob: resProb,
      detected: resProb >= threshold,
      cam: resCAM,
      color: '#a855f7',
      desc: 'Deep residual bottleneck layers preventing gradient vanishing at 512x512 resolution.',
    },
  ];

  return (
    <div className="card mb-4" id="model-comparison-view">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexWrap: 'wrap', gap: 10 }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: '#f8fafc' }}>
            Tri-Model Comparative Benchmark
          </h2>
          <p style={{ fontSize: '0.82rem', color: '#94a3b8', margin: '4px 0 0 0' }}>
            Direct head-to-head comparison between DenseNet-121, SwinCheX, and ResNet-50 backbones
          </p>
        </div>

        {/* Detected pathology buttons only */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.78rem', color: '#94a3b8', fontWeight: 600, marginRight: 4 }}>
            Detected Pathology:
          </span>
          {detectedDiseases.map((d) => {
            const isSelected = currentDisease === d;
            return (
              <button
                key={d}
                onClick={() => setActiveDisease(d)}
                style={{
                  padding: '5px 12px',
                  borderRadius: 6,
                  border: `1.5px solid ${isSelected ? '#3b82f6' : '#334155'}`,
                  background: isSelected ? '#3b82f625' : '#1e293b',
                  color: isSelected ? '#60a5fa' : '#cbd5e1',
                  fontWeight: isSelected ? 700 : 500,
                  fontSize: '0.80rem',
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
              >
                {d}
              </button>
            );
          })}
        </div>
      </div>

      {/* Model Probability Comparison Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: 12,
          marginBottom: 20,
        }}
      >
        {modelsMeta.map((m) => {
          const pct = (m.prob * 100).toFixed(1);
          const isHigh = m.detected;
          return (
            <div
              key={m.key}
              style={{
                background: '#18181b',
                border: `1.5px solid ${isHigh ? m.color : '#27272a'}`,
                borderRadius: 10,
                padding: '14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 700, fontSize: '0.95rem', color: m.color }}>
                  {m.name}
                </span>
                <span
                  style={{
                    fontSize: '0.68rem',
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: 8,
                    background: isHigh ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                    color: isHigh ? '#f87171' : '#4ade80',
                    border: `1px solid ${isHigh ? '#ef444455' : '#22c55e55'}`,
                  }}
                >
                  {isHigh ? 'Detected' : 'Normal'}
                </span>
              </div>

              <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>{m.arch}</div>

              {/* Progress bar */}
              <div style={{ marginTop: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.80rem', marginBottom: 4 }}>
                  <span style={{ color: '#cbd5e1' }}>Confidence</span>
                  <span style={{ fontWeight: 700, color: isHigh ? '#f87171' : '#94a3b8' }}>
                    {pct}%
                  </span>
                </div>
                <div style={{ height: 6, background: '#27272a', borderRadius: 3, overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.min(100, Math.max(0, m.prob * 100))}%`,
                      background: m.color,
                      transition: 'width 0.4s ease',
                      borderRadius: 3,
                    }}
                  />
                </div>
              </div>

              <div style={{ fontSize: '0.68rem', color: '#71717a', marginTop: 4 }}>
                {m.params} · Cutoff: {(threshold * 100).toFixed(0)}%
              </div>
            </div>
          );
        })}

        {/* Ensemble Consensus Card */}
        <div
          style={{
            background: 'rgba(59, 130, 246, 0.08)',
            border: '1.5px solid rgba(59, 130, 246, 0.4)',
            borderRadius: 10,
            padding: '14px',
            display: 'flex',
            flexDirection: 'column',
            gap: 6,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: 700, fontSize: '0.95rem', color: '#60a5fa' }}>
              ✦ Fused Ensemble
            </span>
            <span
              style={{
                fontSize: '0.68rem',
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: 8,
                background: activePred?.detected ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                color: activePred?.detected ? '#f87171' : '#4ade80',
                border: `1px solid ${activePred?.detected ? '#ef444455' : '#22c55e55'}`,
              }}
            >
              {activePred?.detected ? 'Detected' : 'Normal'}
            </span>
          </div>

          <div style={{ fontSize: '0.72rem', color: '#93c5fd' }}>
            Consensus: <strong>{activePred?.consensus || 'N/A'} Models Agree</strong>
          </div>

          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.80rem', marginBottom: 4 }}>
              <span style={{ color: '#cbd5e1' }}>Weighted Soft-Vote</span>
              <span style={{ fontWeight: 700, color: '#60a5fa' }}>
                {(ensembleProb * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ height: 6, background: '#27272a', borderRadius: 3, overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${Math.min(100, Math.max(0, ensembleProb * 100))}%`,
                  background: 'linear-gradient(90deg, #3b82f6, #60a5fa)',
                  borderRadius: 3,
                }}
              />
            </div>
          </div>

          <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: 4 }}>
            Weights: Swin 45% · DenseNet 35% · ResNet 20%
          </div>
        </div>
      </div>

      {/* Visual LayerCAM Heatmap Comparison Grid */}
      <div style={{ marginTop: 24 }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 12, color: '#f1f5f9' }}>
          Side-by-Side Visual Explainability ({currentDisease})
        </h3>
        <p style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: 14 }}>
          Compare where each architecture focuses its attention within the thoracic lung fields
        </p>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))',
            gap: 14,
          }}
        >
          {/* Panel 1: Original X-Ray */}
          <div style={{ background: '#09090b', borderRadius: 8, overflow: 'hidden', border: '1px solid #27272a' }}>
            <div style={{ padding: '6px 10px', background: '#18181b', fontSize: '0.75rem', fontWeight: 600, color: '#cbd5e1' }}>
              1. Original Radiograph
            </div>
            <div style={{ height: 260, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {previewUrl && (
                <img src={previewUrl} alt="Original" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              )}
            </div>
          </div>

          {/* Panel 2: SwinCheX */}
          <div style={{ background: '#09090b', borderRadius: 8, overflow: 'hidden', border: '1px solid #27272a' }}>
            <div style={{ padding: '6px 10px', background: '#18181b', fontSize: '0.75rem', fontWeight: 600, color: '#38bdf8' }}>
              2. SwinCheX LayerCAM ({(swinProb * 100).toFixed(1)}%)
            </div>
            <div style={{ height: 260, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {swinCAM?.overlay ? (
                <img src={`data:image/jpeg;base64,${swinCAM.overlay}`} alt="Swin LayerCAM" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>LayerCAM not available</span>
              )}
            </div>
          </div>

          {/* Panel 3: DenseNet-121 */}
          <div style={{ background: '#09090b', borderRadius: 8, overflow: 'hidden', border: '1px solid #27272a' }}>
            <div style={{ padding: '6px 10px', background: '#18181b', fontSize: '0.75rem', fontWeight: 600, color: '#f59e0b' }}>
              3. DenseNet-121 LayerCAM ({(denseProb * 100).toFixed(1)}%)
            </div>
            <div style={{ height: 260, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {denseCAM?.overlay ? (
                <img src={`data:image/jpeg;base64,${denseCAM.overlay}`} alt="DenseNet LayerCAM" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>LayerCAM not available</span>
              )}
            </div>
          </div>

          {/* Panel 4: ResNet-50 */}
          <div style={{ background: '#09090b', borderRadius: 8, overflow: 'hidden', border: '1px solid #27272a' }}>
            <div style={{ padding: '6px 10px', background: '#18181b', fontSize: '0.75rem', fontWeight: 600, color: '#a855f7' }}>
              4. ResNet-50 LayerCAM ({(resProb * 100).toFixed(1)}%)
            </div>
            <div style={{ height: 260, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {resCAM?.overlay ? (
                <img src={`data:image/jpeg;base64,${resCAM.overlay}`} alt="ResNet LayerCAM" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>LayerCAM not available</span>
              )}
            </div>
          </div>

          {/* Panel 5: Ensemble Fused */}
          <div style={{ background: '#09090b', borderRadius: 8, overflow: 'hidden', border: '1.5px solid rgba(59, 130, 246, 0.5)' }}>
            <div style={{ padding: '6px 10px', background: '#1e3a8a30', fontSize: '0.75rem', fontWeight: 700, color: '#60a5fa' }}>
              ✦ Fused Tri-Model Ensemble ({(ensembleProb * 100).toFixed(1)}%)
            </div>
            <div style={{ height: 260, backgroundColor: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {ensembleCAM?.overlay ? (
                <img src={`data:image/jpeg;base64,${ensembleCAM.overlay}`} alt="Ensemble Fused CAM" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>LayerCAM not available</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Comprehensive Pathology Comparison Table */}
      <div style={{ marginTop: 24, overflowX: 'auto' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 10, color: '#f1f5f9' }}>
          All-Pathologies Multi-Model Matrix
        </h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: '#18181b', color: '#94a3b8', borderBottom: '1px solid #27272a' }}>
              <th style={{ padding: '8px 12px' }}>Pathology</th>
              <th style={{ padding: '8px 12px' }}>Status</th>
              <th style={{ padding: '8px 12px', color: '#38bdf8' }}>SwinCheX</th>
              <th style={{ padding: '8px 12px', color: '#f59e0b' }}>DenseNet-121</th>
              <th style={{ padding: '8px 12px', color: '#a855f7' }}>ResNet-50</th>
              <th style={{ padding: '8px 12px', color: '#60a5fa' }}>Ensemble Prob</th>
              <th style={{ padding: '8px 12px' }}>Consensus</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p) => {
              const mp = p.model_predictions || {};
              const getVal = (prefix) => {
                for (const [k, v] of Object.entries(mp)) {
                  if (k.toLowerCase().includes(prefix.toLowerCase())) {
                    return typeof v === 'number' ? v : parseFloat(v) || 0;
                  }
                }
                return p.probability;
              };
              const sP = getVal('swin');
              const dP = getVal('dense');
              const rP = getVal('res');
              const isDetected = p.detected;
              const isCurrent = p.disease === currentDisease;

              return (
                <tr
                  key={p.disease}
                  onClick={() => setActiveDisease(p.disease)}
                  style={{
                    borderBottom: '1px solid #27272a',
                    background: isCurrent ? 'rgba(59, 130, 246, 0.12)' : 'transparent',
                    cursor: 'pointer',
                    transition: 'background 0.1s',
                  }}
                >
                  <td style={{ padding: '8px 12px', fontWeight: isDetected ? 700 : 500, color: isDetected ? '#f87171' : '#e2e8f0' }}>
                    {p.disease}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span
                      style={{
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        padding: '1px 6px',
                        borderRadius: 4,
                        background: isDetected ? '#ef444420' : '#22c55e20',
                        color: isDetected ? '#f87171' : '#4ade80',
                      }}
                    >
                      {isDetected ? 'Detected' : 'Normal'}
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px', color: sP >= p.threshold ? '#38bdf8' : '#94a3b8' }}>
                    {(sP * 100).toFixed(1)}% {sP >= p.threshold ? '●' : '○'}
                  </td>
                  <td style={{ padding: '8px 12px', color: dP >= p.threshold ? '#f59e0b' : '#94a3b8' }}>
                    {(dP * 100).toFixed(1)}% {dP >= p.threshold ? '●' : '○'}
                  </td>
                  <td style={{ padding: '8px 12px', color: rP >= p.threshold ? '#a855f7' : '#94a3b8' }}>
                    {(rP * 100).toFixed(1)}% {rP >= p.threshold ? '●' : '○'}
                  </td>
                  <td style={{ padding: '8px 12px', fontWeight: 700, color: isDetected ? '#f87171' : '#10b981' }}>
                    {(p.probability * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '8px 12px', color: p.consensus?.startsWith('3/3') ? '#34d399' : p.consensus?.startsWith('2/3') ? '#fbbf24' : '#94a3b8' }}>
                    {p.consensus || 'N/A'}
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

