import React, { useState } from 'react';

export default function HeatmapGallery({ heatmaps = [], previewUrl = '' }) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [opacity, setOpacity] = useState(0.6);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [invert, setInvert] = useState(false);

  if (!heatmaps || heatmaps.length === 0) {
    return (
      <div className="card mb-4">
        <p className="text-muted">No Grad-CAM heatmaps generated for this scan.</p>
      </div>
    );
  }

  const activeHeatmap = heatmaps[selectedIdx] || heatmaps[0];

  const imgStyle = {
    filter: `brightness(${brightness}%) contrast(${contrast}%) ${invert ? 'invert(100%)' : ''}`,
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    display: 'block',
  };

  return (
    <div className="card mb-4" id="heatmap-gallery">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>
          Per-Disease Grad-CAM Explainability
        </h2>
        <div style={{ display: 'flex', gap: '8px' }}>
          {heatmaps.map((hm, idx) => (
            <button
              key={hm.disease || idx}
              className={`btn ${selectedIdx === idx ? 'btn-primary' : 'btn-ghost'}`}
              style={{ padding: '4px 12px', fontSize: '0.85rem' }}
              onClick={() => setSelectedIdx(idx)}
            >
              {hm.disease}
            </button>
          ))}
        </div>
      </div>

      {/* Viewer controls */}
      <div style={{ background: '#18181b', padding: '12px 16px', borderRadius: '6px', marginBottom: '16px', display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span className="text-sm text-muted">Opacity:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
          />
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span className="text-sm text-muted">Brightness:</span>
          <input
            type="range"
            min="50"
            max="200"
            value={brightness}
            onChange={(e) => setBrightness(Number(e.target.value))}
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
          />
        </div>
        <button
          className="btn btn-ghost"
          style={{ padding: '4px 8px', fontSize: '0.8rem' }}
          onClick={() => setInvert(!invert)}
        >
          {invert ? 'Normal' : 'Invert'}
        </button>
      </div>

      {/* Image display container */}
      <div style={{ height: '500px', backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
        {previewUrl && (
          <img
            src={previewUrl}
            alt="Original X-Ray"
            style={{ ...imgStyle, position: 'absolute', inset: 0 }}
          />
        )}

        {activeHeatmap.heatmap && (
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

      <div style={{ marginTop: '12px', fontSize: '0.85rem', color: '#9ca3af', textAlign: 'center' }}>
        Highlighting pathological region for: <strong style={{ color: '#ffffff' }}>{activeHeatmap.disease}</strong>
      </div>
    </div>
  );
}
