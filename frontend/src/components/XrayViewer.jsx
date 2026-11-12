import React, { useState } from 'react';

/**
 * XrayViewer — Medical-grade interactive image viewport.
 *
 * Features:
 *  - Fullscreen/viewport rendering with zoom and pan
 *  - Live window/level (brightness + contrast)
 *  - Grad-CAM heatmap overlay with blend modes and opacity slider
 *  - Monochromatic inversion (LUT)
 *  - Reset controls
 */
export default function XrayViewer({
  imageUrl,
  overlayUrl,
  title = 'Chest X-Ray',
  diseases = [],
  selectedDisease = '',
  onSelectDisease,
}) {
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.65);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [invert, setInvert] = useState(false);

  function handleWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((z) => Math.max(0.4, Math.min(4.0, z + delta)));
  }

  function handleMouseDown(e) {
    e.preventDefault();
    setDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  }

  function handleMouseMove(e) {
    if (!dragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  }

  function handleMouseUp() {
    setDragging(false);
  }

  function reset() {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
    setBrightness(100);
    setContrast(100);
    setInvert(false);
    setShowOverlay(true);
    setOverlayOpacity(0.65);
  }

  const imgFilter = [
    `brightness(${brightness}%)`,
    `contrast(${contrast}%)`,
    invert ? 'invert(100%)' : '',
  ].filter(Boolean).join(' ');

  const btnStyle = (active) => ({
    padding: '4px 10px',
    border: `1px solid ${active ? '#3b82f6' : '#3f3f46'}`,
    borderRadius: 6,
    background: active ? '#3b82f625' : '#27272a',
    color: active ? '#60a5fa' : '#cbd5e1',
    cursor: 'pointer',
    fontSize: '0.75rem',
    fontWeight: 600,
    transition: 'all 0.15s',
  });

  return (
    <div
      id="xray-viewer"
      style={{
        background: '#09090b',
        borderRadius: 12,
        overflow: 'hidden',
        border: '1px solid #27272a',
        userSelect: 'none',
      }}
    >
      {/* Top Toolbar */}
      <div
        style={{
          background: '#0f172a',
          padding: '10px 14px',
          display: 'flex',
          gap: 10,
          alignItems: 'center',
          flexWrap: 'wrap',
          borderBottom: '1px solid #1e293b',
        }}
      >
        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#f8fafc', marginRight: 4 }}>
          {title}
        </span>

        {/* Disease switcher if provided */}
        {diseases.length > 0 && onSelectDisease && (
          <div style={{ display: 'flex', gap: 4, marginRight: 8 }}>
            {diseases.map((d) => (
              <button
                key={d}
                onClick={() => onSelectDisease(d)}
                style={{
                  padding: '3px 8px',
                  borderRadius: 4,
                  border: `1px solid ${selectedDisease === d ? '#3b82f6' : '#334155'}`,
                  background: selectedDisease === d ? '#3b82f630' : 'transparent',
                  color: selectedDisease === d ? '#60a5fa' : '#94a3b8',
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                {d}
              </button>
            ))}
          </div>
        )}

        <div style={{ width: 1, height: 20, background: '#334155' }} />

        {/* Zoom Controls */}
        <button style={btnStyle(false)} onClick={() => setZoom((z) => Math.min(4, z + 0.25))} title="Zoom In">+</button>
        <button style={btnStyle(false)} onClick={() => setZoom((z) => Math.max(0.4, z - 0.25))} title="Zoom Out">−</button>
        <span style={{ fontSize: '0.75rem', color: '#94a3b8', minWidth: 38 }}>
          {Math.round(zoom * 100)}%
        </span>

        <div style={{ width: 1, height: 20, background: '#334155' }} />

        {/* Window/Level Brightness */}
        <label style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 4 }}>
          ☀
          <input
            type="range"
            min={30}
            max={200}
            value={brightness}
            onChange={(e) => setBrightness(Number(e.target.value))}
            style={{ width: 65, cursor: 'pointer' }}
            title={`Brightness: ${brightness}%`}
          />
        </label>

        {/* Contrast */}
        <label style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 4 }}>
          ◑
          <input
            type="range"
            min={30}
            max={250}
            value={contrast}
            onChange={(e) => setContrast(Number(e.target.value))}
            style={{ width: 65, cursor: 'pointer' }}
            title={`Contrast: ${contrast}%`}
          />
        </label>

        <button style={btnStyle(invert)} onClick={() => setInvert((v) => !v)} title="Invert LUT">
          {invert ? 'Normal' : 'Invert'}
        </button>

        <div style={{ width: 1, height: 20, background: '#334155' }} />

        {/* Grad-CAM Overlay Toggle & Slider */}
        {overlayUrl && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <button
              style={btnStyle(showOverlay)}
              onClick={() => setShowOverlay((v) => !v)}
              title="Toggle LayerCAM Overlay"
            >
              {showOverlay ? '✓ CAM' : 'CAM Off'}
            </button>
            {showOverlay && (
              <label style={{ fontSize: '0.72rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 4 }}>
                Opacity:
                <input
                  type="range"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                  style={{ width: 60, cursor: 'pointer' }}
                />
                <span>{Math.round(overlayOpacity * 100)}%</span>
              </label>
            )}
          </div>
        )}

        <button style={{ ...btnStyle(false), marginLeft: 'auto', color: '#f87171' }} onClick={reset}>
          Reset View
        </button>
      </div>

      {/* Viewport Canvas */}
      <div
        style={{
          position: 'relative',
          height: 480,
          backgroundColor: '#000',
          cursor: dragging ? 'grabbing' : 'grab',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {imageUrl ? (
          <div
            style={{
              position: 'relative',
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
              transition: dragging ? 'none' : 'transform 0.05s ease',
            }}
          >
            {/* Base Radiograph */}
            <img
              src={imageUrl}
              alt="Chest X-ray"
              draggable={false}
              style={{
                maxWidth: '96%',
                maxHeight: '96%',
                objectFit: 'contain',
                filter: imgFilter,
                display: 'block',
                pointerEvents: 'none',
              }}
            />

            {/* Grad-CAM Heatmap Overlay */}
            {overlayUrl && showOverlay && (
              <img
                src={overlayUrl}
                alt="LayerCAM Heatmap Overlay"
                draggable={false}
                style={{
                  position: 'absolute',
                  maxWidth: '96%',
                  maxHeight: '96%',
                  objectFit: 'contain',
                  opacity: overlayOpacity,
                  mixBlendMode: 'screen',
                  pointerEvents: 'none',
                }}
              />
            )}
          </div>
        ) : (
          <div style={{ color: '#64748b', fontSize: '0.9rem' }}>No image loaded</div>
        )}
      </div>

      {/* Footer Navigation Hints */}
      <div
        style={{
          background: '#0f172a',
          padding: '6px 14px',
          display: 'flex',
          gap: 16,
          fontSize: '0.72rem',
          color: '#64748b',
          borderTop: '1px solid #1e293b',
          flexWrap: 'wrap',
        }}
      >
        <span>🖱 Mouse Wheel = Zoom ({Math.round(zoom * 100)}%)</span>
        <span>✋ Click &amp; Drag = Pan</span>
        <span>☀/◑ = Brightness &amp; Contrast</span>
        {overlayUrl && <span>✦ LayerCAM Overlay: {showOverlay ? 'Active' : 'Hidden'}</span>}
      </div>
    </div>
  );
}
