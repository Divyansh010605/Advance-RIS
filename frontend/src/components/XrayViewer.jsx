import React, { useState, useEffect } from 'react';

/**
 * XrayViewer — Phase 3 medical-grade image viewport.
 *
 * Features:
 *  - Window/Level (brightness + contrast) sliders
 *  - Zoom in/out via mouse wheel and buttons
 *  - Pan by click-drag
 *  - Overlay toggle (Grad-CAM heatmap)
 *  - Invert LUT (monochrome1 style)
 *  - Reset to defaults
 */
export default function XrayViewer({ imageUrl, overlayUrl, title = 'Chest X-Ray' }) {
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);
  const [brightness, setBrightness] = useState(100); // %
  const [contrast, setContrast] = useState(100);     // %
  const [invert, setInvert] = useState(false);

  // Scroll-to-zoom
  function handleWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((z) => Math.max(0.3, Math.min(5.0, z + delta)));
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
    setShowOverlay(false);
    setOverlayOpacity(0.6);
  }

  const imgFilter = [
    `brightness(${brightness / 100})`,
    `contrast(${contrast / 100})`,
    invert ? 'invert(1)' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const transform = `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`;

  const btnStyle = (active) => ({
    padding: '4px 10px',
    border: `1px solid ${active ? '#3b82f6' : '#3f3f46'}`,
    borderRadius: 6,
    background: active ? '#3b82f620' : '#27272a',
    color: active ? '#60a5fa' : '#9ca3af',
    cursor: 'pointer',
    fontSize: '0.75rem',
    fontWeight: 600,
    transition: 'all 0.15s',
  });

  return (
    <div
      id="xray-viewer"
      style={{
        background: '#000',
        borderRadius: 12,
        overflow: 'hidden',
        border: '1px solid #27272a',
        userSelect: 'none',
      }}
    >
      {/* Toolbar */}
      <div
        style={{
          background: '#0f172a',
          padding: '8px 12px',
          display: 'flex',
          gap: 8,
          alignItems: 'center',
          flexWrap: 'wrap',
          borderBottom: '1px solid #1e293b',
        }}
      >
        <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#e2e8f0', marginRight: 4 }}>
          {title}
        </span>

        <button style={btnStyle(false)} onClick={() => setZoom((z) => Math.min(5, z + 0.25))} title="Zoom in" aria-label="Zoom in">+</button>
        <button style={btnStyle(false)} onClick={() => setZoom((z) => Math.max(0.3, z - 0.25))} title="Zoom out" aria-label="Zoom out">−</button>
        <span style={{ fontSize: '0.72rem', color: '#64748b', minWidth: 38 }}>{Math.round(zoom * 100)}%</span>

        <div style={{ width: 1, height: 20, background: '#334155' }} />

        <label style={{ fontSize: '0.72rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 4 }}>
          ☀
          <input
            type="range" min={30} max={200} value={brightness}
            onChange={(e) => setBrightness(Number(e.target.value))}
            style={{ width: 60 }}
            aria-label="Brightness"
            title={`Brightness: ${brightness}%`}
          />
        </label>

        <label style={{ fontSize: '0.72rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 4 }}>
          ◑
          <input
            type="range" min={30} max={300} value={contrast}
            onChange={(e) => setContrast(Number(e.target.value))}
            style={{ width: 60 }}
            aria-label="Contrast"
            title={`Contrast: ${contrast}%`}
          />
        </label>

        <div style={{ width: 1, height: 20, background: '#334155' }} />

        <button style={btnStyle(invert)} onClick={() => setInvert((v) => !v)} title="Invert LUT" aria-label="Invert image">
          INV
        </button>

        {overlayUrl && (
          <>
            <button
              style={btnStyle(showOverlay)}
              onClick={() => setShowOverlay((v) => !v)}
              title="Toggle Grad-CAM overlay"
              aria-label="Toggle heatmap overlay"
            >
              CAM
            </button>
            {showOverlay && (
              <label style={{ fontSize: '0.72rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 4 }}>
                Opacity
                <input
                  type="range" min={0.1} max={1} step={0.05} value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                  style={{ width: 55 }}
                  aria-label="Overlay opacity"
                />
              </label>
            )}
          </>
        )}

        <button style={{ ...btnStyle(false), marginLeft: 'auto' }} onClick={reset} aria-label="Reset view">
          Reset
        </button>
      </div>

      {/* Viewport */}
      <div
        style={{ position: 'relative', height: 420, cursor: dragging ? 'grabbing' : 'grab', overflow: 'hidden' }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        aria-label="X-ray image viewer. Scroll to zoom, drag to pan."
        role="img"
      >
        {imageUrl ? (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform,
              transformOrigin: 'center center',
              transition: dragging ? 'none' : 'transform 0.05s ease',
              marginLeft: '-50%',
              marginTop: '-50%',
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <img
              src={imageUrl}
              alt="Chest X-ray"
              draggable={false}
              style={{ maxWidth: '100%', maxHeight: '100%', filter: imgFilter, display: 'block' }}
            />
            {overlayUrl && showOverlay && (
              <img
                src={overlayUrl}
                alt="Grad-CAM heatmap overlay"
                draggable={false}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  opacity: overlayOpacity,
                  mixBlendMode: 'screen',
                  pointerEvents: 'none',
                }}
              />
            )}
          </div>
        ) : (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: '0.9rem' }}>
            No image loaded
          </div>
        )}
      </div>

      {/* Footer info */}
      <div style={{ background: '#0f172a', padding: '5px 12px', display: 'flex', gap: 16, fontSize: '0.7rem', color: '#475569', borderTop: '1px solid #1e293b' }}>
        <span>Scroll = Zoom</span>
        <span>Drag = Pan</span>
        <span>W/L via sliders</span>
        {overlayUrl && <span>CAM = Grad-CAM toggle</span>}
      </div>
    </div>
  );
}
