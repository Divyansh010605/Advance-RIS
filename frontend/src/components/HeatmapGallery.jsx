import React, { useState, useMemo } from 'react';

const MODEL_LABELS = {
  ensemble: { name: '✦ Ensemble Fused', badge: 'Optimal', desc: 'Weighted fusion across SwinCheX, DenseNet-121, and ResNet-50 with 20% background noise suppression and Gaussian refinement.' },
  swin: { name: 'SwinCheX', badge: 'Transformer', desc: 'Attention feature gradients backpropagated from Swin Transformer Stage 4 norm1 layer.' },
  densenet121: { name: 'DenseNet-121', badge: 'CNN', desc: 'Convolutional feature gradients backpropagated from DenseNet denseblock4 logits.' },
  resnet50: { name: 'ResNet-50', badge: 'Residual', desc: 'Residual bottleneck gradients backpropagated from ResNet-50 layer4 logits.' },
};

export default function HeatmapGallery({ heatmaps = [], previewUrl = '', predictions = [] }) {
  const [selectedDisease, setSelectedDisease] = useState('');
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [viewMode, setViewMode] = useState('blend'); // 'blend' | 'side-by-side' | 'heatmap-only'
  const [opacity, setOpacity] = useState(0.65);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [invert, setInvert] = useState(false);

  // Extract unique diseases — ONLY detected diseases are shown
  const uniqueDiseases = useMemo(() => {
    if (!heatmaps || heatmaps.length === 0) return [];
    const available = Array.from(new Set(heatmaps.map((h) => h.disease)));
    if (predictions && predictions.length > 0) {
      const detectedSet = new Set(predictions.filter((p) => p.detected).map((p) => p.disease));
      if (detectedSet.size > 0) {
        const filtered = available.filter((d) => detectedSet.has(d));
        if (filtered.length > 0) return filtered;
      }
    }
    return available.slice(0, 1);
  }, [heatmaps, predictions]);

  // Current active disease
  const activeDisease = selectedDisease && uniqueDiseases.includes(selectedDisease)
    ? selectedDisease
    : uniqueDiseases[0] || '';

  // Filter heatmaps for active disease
  const diseaseHeatmaps = useMemo(() => {
    if (!activeDisease) return [];
    return heatmaps.filter((h) => h.disease === activeDisease);
  }, [heatmaps, activeDisease]);

  // Available models for this disease
  const availableModels = useMemo(() => {
    if (!diseaseHeatmaps.length) return [];
    const models = Array.from(new Set(diseaseHeatmaps.map((h) => h.model_name || 'ensemble')));
    // Order: ensemble first, then swin, densenet121, resnet50, others
    const order = ['ensemble', 'swin', 'densenet121', 'resnet50'];
    return models.sort((a, b) => {
      const idxA = order.indexOf(a);
      const idxB = order.indexOf(b);
      return (idxA === -1 ? 99 : idxA) - (idxB === -1 ? 99 : idxB);
    });
  }, [diseaseHeatmaps]);

  // Active heatmap item
  const activeHeatmap = useMemo(() => {
    if (!diseaseHeatmaps.length) return heatmaps[0] || null;
    const match = diseaseHeatmaps.find((h) => (h.model_name || 'ensemble') === selectedModel);
    return match || diseaseHeatmaps[0];
  }, [diseaseHeatmaps, selectedModel, heatmaps]);

  if (!heatmaps || heatmaps.length === 0) {
    return (
      <div className="card mb-4">
        <p className="text-muted">No Grad-CAM heatmaps generated for this scan.</p>
      </div>
    );
  }

  const currentModelKey = activeHeatmap?.model_name || selectedModel || 'ensemble';
  const currentModelMeta = MODEL_LABELS[currentModelKey] || {
    name: currentModelKey,
    badge: 'Model',
    desc: 'Pathology localization gradient attribution map.',
  };

  const imgStyle = {
    filter: `brightness(${brightness}%) contrast(${contrast}%) ${invert ? 'invert(100%)' : ''}`,
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    display: 'block',
  };

  return (
    <div className="card mb-4" id="heatmap-gallery">
      {/* Header with Title and Disease Selector */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>
            Per-Disease Grad-CAM Explainability
          </h2>
          <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginTop: 2 }}>
            High-precision anatomical localization with background noise suppression
          </div>
        </div>

        {/* Disease pills */}
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          {uniqueDiseases.map((disease) => {
            const isSelected = activeDisease === disease;
            return (
              <button
                key={disease}
                id={`cam-disease-${disease.toLowerCase().replace(/\s+/g, '-')}`}
                className={`btn ${isSelected ? 'btn-primary' : 'btn-ghost'}`}
                style={{
                  padding: '4px 12px',
                  fontSize: '0.82rem',
                  fontWeight: isSelected ? 700 : 500,
                  borderRadius: 6,
                }}
                onClick={() => setSelectedDisease(disease)}
              >
                {disease}
              </button>
            );
          })}
        </div>
      </div>

      {/* Model Selector Bar */}
      <div
        style={{
          background: 'rgba(24, 24, 27, 0.8)',
          border: '1px solid #27272a',
          borderRadius: 8,
          padding: '8px 12px',
          marginBottom: 14,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.78rem', color: '#a1a1aa', fontWeight: 600 }}>
            Model CAM Backbone:
          </span>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {availableModels.map((mKey) => {
              const meta = MODEL_LABELS[mKey] || { name: mKey, badge: 'Model' };
              const isSelected = (activeHeatmap?.model_name || selectedModel) === mKey;
              return (
                <button
                  key={mKey}
                  id={`cam-model-${mKey}`}
                  onClick={() => setSelectedModel(mKey)}
                  style={{
                    padding: '3px 10px',
                    borderRadius: 6,
                    border: `1px solid ${isSelected ? '#3b82f6' : '#3f3f46'}`,
                    background: isSelected ? '#3b82f625' : '#1e1e24',
                    color: isSelected ? '#60a5fa' : '#9ca3af',
                    fontSize: '0.76rem',
                    fontWeight: isSelected ? 700 : 500,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    transition: 'all 0.15s',
                  }}
                >
                  <span>{meta.name}</span>
                  <span
                    style={{
                      fontSize: '0.62rem',
                      padding: '1px 5px',
                      borderRadius: 4,
                      background: isSelected ? '#3b82f640' : '#27272a',
                      color: isSelected ? '#93c5fd' : '#71717a',
                    }}
                  >
                    {meta.badge}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* View Mode Toggle */}
        <div style={{ display: 'flex', gap: 4 }}>
          {[
            { id: 'blend', label: 'Overlay' },
            { id: 'side-by-side', label: 'Side-by-Side' },
            { id: 'heatmap-only', label: 'Heatmap Only' },
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => setViewMode(mode.id)}
              style={{
                padding: '3px 8px',
                borderRadius: 5,
                border: `1px solid ${viewMode === mode.id ? '#6366f1' : '#27272a'}`,
                background: viewMode === mode.id ? '#6366f120' : 'transparent',
                color: viewMode === mode.id ? '#a5b4fc' : '#71717a',
                fontSize: '0.72rem',
                cursor: 'pointer',
              }}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>

      {/* Viewer controls */}
      <div
        style={{
          background: '#18181b',
          padding: '10px 16px',
          borderRadius: '6px',
          marginBottom: '14px',
          display: 'flex',
          gap: '20px',
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        {viewMode !== 'heatmap-only' && (
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span className="text-sm text-muted">CAM Opacity:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              style={{ cursor: 'pointer' }}
            />
            <span style={{ fontSize: '0.75rem', color: '#9ca3af', minWidth: 32 }}>
              {Math.round(opacity * 100)}%
            </span>
          </div>
        )}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span className="text-sm text-muted">Brightness:</span>
          <input
            type="range"
            min="50"
            max="200"
            value={brightness}
            onChange={(e) => setBrightness(Number(e.target.value))}
            style={{ cursor: 'pointer' }}
          />
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span className="text-sm text-muted">Contrast:</span>
          <input
            type="range"
            min="50"
            max="200"
            value={contrast}
            onChange={(e) => setContrast(Number(e.target.value))}
            style={{ cursor: 'pointer' }}
          />
        </div>
        <button
          className="btn btn-ghost"
          style={{ padding: '3px 8px', fontSize: '0.75rem' }}
          onClick={() => setInvert(!invert)}
        >
          {invert ? 'Normal LUT' : 'Invert (B/W)'}
        </button>
        <button
          className="btn btn-ghost"
          style={{ padding: '3px 8px', fontSize: '0.75rem', color: '#ef4444' }}
          onClick={() => {
            setOpacity(0.65);
            setBrightness(100);
            setContrast(100);
            setInvert(false);
          }}
        >
          Reset View
        </button>
      </div>

      {/* Image display container */}
      {viewMode === 'side-by-side' ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, height: '480px' }}>
          {/* Left: Original X-Ray */}
          <div style={{ backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
            <div style={{ position: 'absolute', top: 8, left: 10, zIndex: 10, background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: 4, fontSize: '0.72rem', color: '#94a3b8' }}>
              Original Radiograph
            </div>
            {previewUrl && (
              <img src={previewUrl} alt="Original X-Ray" style={imgStyle} />
            )}
          </div>
          {/* Right: Heatmap Overlay */}
          <div style={{ backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
            <div style={{ position: 'absolute', top: 8, left: 10, zIndex: 10, background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: 4, fontSize: '0.72rem', color: '#60a5fa' }}>
              Grad-CAM Overlay ({currentModelMeta.name})
            </div>
            {activeHeatmap?.overlay ? (
              <img
                src={`data:image/jpeg;base64,${activeHeatmap.overlay}`}
                alt={`${activeHeatmap.disease} Overlay`}
                style={imgStyle}
              />
            ) : (
              previewUrl && <img src={previewUrl} alt="X-Ray" style={imgStyle} />
            )}
          </div>
        </div>
      ) : viewMode === 'heatmap-only' ? (
        <div style={{ height: '480px', backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
          <div style={{ position: 'absolute', top: 8, left: 10, zIndex: 10, background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: 4, fontSize: '0.72rem', color: '#f59e0b' }}>
            Pure Heatmap Activation ({currentModelMeta.name})
          </div>
          {activeHeatmap?.heatmap && (
            <img
              src={`data:image/jpeg;base64,${activeHeatmap.heatmap}`}
              alt={`${activeHeatmap.disease} Grad-CAM Heatmap`}
              style={imgStyle}
            />
          )}
        </div>
      ) : (
        /* Blend Mode (Default) */
        <div style={{ height: '500px', backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
          <div style={{ position: 'absolute', top: 8, left: 10, zIndex: 10, background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: 4, fontSize: '0.72rem', color: '#38bdf8' }}>
            Interactive Blended View ({currentModelMeta.name})
          </div>
          {previewUrl && (
            <img
              src={previewUrl}
              alt="Original X-Ray"
              style={{ ...imgStyle, position: 'absolute', inset: 0 }}
            />
          )}

          {activeHeatmap?.heatmap && (
            <img
              src={`data:image/jpeg;base64,${activeHeatmap.heatmap}`}
              alt={`${activeHeatmap.disease} Grad-CAM`}
              style={{
                ...imgStyle,
                position: 'absolute',
                inset: 0,
                opacity: opacity,
                mixBlendMode: 'screen',
              }}
            />
          )}
        </div>
      )}

      {/* Description & Technical Footer */}
      <div style={{ marginTop: '14px', padding: '10px 14px', background: 'rgba(24, 24, 27, 0.5)', borderRadius: 6, border: '1px solid #27272a' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
          <div style={{ fontSize: '0.85rem', color: '#d1d5db' }}>
            Pathological target: <strong style={{ color: '#ffffff' }}>{activeDisease}</strong>
            {' · '}
            Model: <strong style={{ color: '#60a5fa' }}>{currentModelMeta.name}</strong>
          </div>
          <div style={{ fontSize: '0.72rem', color: '#9ca3af' }}>
            {currentModelMeta.desc}
          </div>
        </div>
      </div>
    </div>
  );
}
