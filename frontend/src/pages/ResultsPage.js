import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import MultiDiseaseResults from '../components/MultiDiseaseResults';
import HeatmapGallery from '../components/HeatmapGallery';
import ReportEditor from '../components/ReportEditor';
import XrayViewer from '../components/XrayViewer';
import LongitudinalComparison from '../components/LongitudinalComparison';

const API_BASE = process.env.REACT_APP_AI_API_URL || 'http://localhost:8000';

/**
 * ResultsPage — Phase 3 upgrade:
 *  - XrayViewer with W/L, zoom, pan, Grad-CAM overlay toggle
 *  - Enhanced MultiDiseaseResults with uncertainty badges & CI bars
 *  - LongitudinalComparison (loads prior from sessionStorage if available)
 *  - PDF Download button (calls /api/v1/report/pdf)
 *  - Ensemble/TTA metadata display
 */
export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, previewUrl } = location.state || {};

  const [selectedDisease, setSelectedDisease] = useState(null);
  const [priorResult, setPriorResult] = useState(null);
  const [showComparison, setShowComparison] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'viewer' | 'comparison'

  useEffect(() => {
    if (!result) {
      navigate('/upload');
      return;
    }
    // Persist current result for potential future longitudinal comparison
    try {
      sessionStorage.setItem('lastResult', JSON.stringify({ result, previewUrl }));
    } catch (_) {}

    // Load prior result if available
    try {
      const stored = sessionStorage.getItem('priorResult');
      if (stored) {
        const parsed = JSON.parse(stored);
        setPriorResult(parsed.result);
        setShowComparison(true);
      }
    } catch (_) {}
  }, [result, navigate, previewUrl]);

  if (!result) return null;

  const predictions = result.predictions || [];
  const heatmaps = result.heatmaps || [];
  const report = result.report || {};
  const meta = result.metadata || {};
  const scanId = result.id || result.request_id;

  const detectedCount = predictions.filter((p) => p.detected).length;
  const uncertainCount = predictions.filter((p) => p.is_uncertain).length;
  const isNormal = detectedCount === 0;

  // Find overlay for selected disease, or use first heatmap
  const activeHeatmap = selectedDisease
    ? heatmaps.find((h) => h.disease === selectedDisease)
    : heatmaps[0];
  const overlayUrl = activeHeatmap?.overlay
    ? `data:image/jpeg;base64,${activeHeatmap.overlay}`
    : null;

  async function handleDownloadPDF() {
    setPdfLoading(true);
    try {
      // Use request_id from result to fetch PDF from job or generate fresh
      const resp = await fetch(`${API_BASE}/api/v1/report/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: result._imageB64 || '',  // stored during submission
          mime_type: 'image/jpeg',
          use_ensemble: meta.ensemble_enabled || false,
          use_tta: meta.tta_enabled || false,
        }),
      });
      if (!resp.ok) {
        alert('PDF generation failed. Please try again.');
        return;
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `nih_report_${scanId || 'export'}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert('PDF download error: ' + e.message);
    } finally {
      setPdfLoading(false);
    }
  }

  function handleUseAsPrior() {
    try {
      sessionStorage.setItem('priorResult', JSON.stringify({ result, previewUrl }));
      alert('Saved as prior study. Upload a new scan to compare.');
    } catch (_) {}
  }

  const tabStyle = (tab) => ({
    padding: '6px 16px',
    border: 'none',
    background: activeTab === tab ? '#3b82f6' : 'transparent',
    color: activeTab === tab ? '#fff' : '#94a3b8',
    borderRadius: 6,
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: '0.82rem',
    transition: 'all 0.15s',
  });

  return (
    <main className="results-page container" role="main" style={{ paddingTop: 24, paddingBottom: 48 }}>
      {/* Header */}
      <header className="page-header" style={{ marginBottom: 20 }}>
        <h1>Multi-Disease Screening &amp; Report</h1>
        <p className="text-muted">
          NIH ChestX-ray14 · 14-class multi-label pathology classification with per-disease Grad-CAM
        </p>
      </header>

      {/* Primary Status Banner */}
      <div
        id="status-banner"
        className="card mb-4"
        style={{
          borderLeft: `6px solid ${isNormal ? '#22c55e' : '#ef4444'}`,
          background: isNormal ? 'rgba(34,197,94,0.05)' : 'rgba(239,68,68,0.05)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <span
              style={{
                fontSize: '0.85rem',
                fontWeight: 700,
                padding: '4px 12px',
                borderRadius: 12,
                background: isNormal ? '#22c55e' : '#ef4444',
                color: '#fff',
              }}
            >
              {isNormal ? '✓ Normal — No Acute Abnormality' : `⚠ ${detectedCount} Patholog${detectedCount > 1 ? 'ies' : 'y'} Detected`}
            </span>
            {uncertainCount > 0 && (
              <span
                style={{
                  marginLeft: 8,
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  padding: '3px 10px',
                  borderRadius: 12,
                  background: '#f59e0b22',
                  color: '#f59e0b',
                  border: '1px solid #f59e0b55',
                }}
              >
                {uncertainCount} uncertain — review required
              </span>
            )}
            <p className="text-sm text-muted mt-2 mb-0">
              Request: <strong>{scanId || 'N/A'}</strong>
              {meta.processing_time_ms && <> · Time: <strong>{meta.processing_time_ms} ms</strong></>}
              {meta.models_evaluated?.length > 1 && (
                <> · Models: <strong>{meta.models_evaluated.join(', ')}</strong></>
              )}
              {meta.total_passes > 1 && (
                <> · Passes: <strong>{meta.total_passes}</strong></>
              )}
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button
              id="download-pdf-btn"
              className="btn btn-primary"
              onClick={handleDownloadPDF}
              disabled={pdfLoading}
              style={{ fontSize: '0.82rem' }}
            >
              {pdfLoading ? 'Generating PDF…' : '⬇ Download PDF Report'}
            </button>
            <button className="btn btn-ghost" style={{ fontSize: '0.82rem' }} onClick={handleUseAsPrior}>
              Save as Prior Study
            </button>
            <button className="btn btn-ghost" onClick={() => navigate('/upload')} style={{ fontSize: '0.82rem' }}>
              New Scan
            </button>
          </div>
        </div>
      </div>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, background: '#0f172a', borderRadius: 8, padding: 4, width: 'fit-content' }}>
        <button style={tabStyle('overview')} onClick={() => setActiveTab('overview')} id="tab-overview">Overview</button>
        <button style={tabStyle('viewer')} onClick={() => setActiveTab('viewer')} id="tab-viewer">Image Viewer</button>
        {showComparison && (
          <button style={tabStyle('comparison')} onClick={() => setActiveTab('comparison')} id="tab-comparison">
            Comparison
          </button>
        )}
      </div>

      {/* ── Tab: Overview ───────────────────────────────────────────── */}
      {activeTab === 'overview' && (
        <>
          {/* Multi-Disease Pathology Cards with uncertainty & CI */}
          <MultiDiseaseResults
            predictions={predictions}
            onSelectDisease={(disease) => {
              setSelectedDisease(disease);
              setActiveTab('viewer');
            }}
          />

          {/* Per-Disease Heatmap Gallery */}
          <HeatmapGallery heatmaps={heatmaps} previewUrl={previewUrl} />

          {/* Interactive Report Editor */}
          <ReportEditor scanId={scanId} initialReport={report} initialStatus={report.status || 'DRAFT'} />
        </>
      )}

      {/* ── Tab: Medical Image Viewer ────────────────────────────────── */}
      {activeTab === 'viewer' && (
        <div className="card mb-4">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h2 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700 }}>
              Medical Image Viewer
              {selectedDisease && (
                <span style={{ marginLeft: 8, fontSize: '0.78rem', color: '#3b82f6', fontWeight: 400 }}>
                  — Grad-CAM: {selectedDisease}
                </span>
              )}
            </h2>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {heatmaps.slice(0, 5).map((h) => (
                <button
                  key={h.disease}
                  id={`select-heatmap-${h.disease.toLowerCase().replace(/\s+/g, '-')}`}
                  onClick={() => setSelectedDisease(h.disease)}
                  style={{
                    padding: '3px 10px',
                    borderRadius: 6,
                    border: `1px solid ${selectedDisease === h.disease ? '#3b82f6' : '#27272a'}`,
                    background: selectedDisease === h.disease ? '#3b82f620' : 'transparent',
                    color: selectedDisease === h.disease ? '#60a5fa' : '#94a3b8',
                    cursor: 'pointer',
                    fontSize: '0.72rem',
                  }}
                >
                  {h.disease}
                </button>
              ))}
            </div>
          </div>
          <XrayViewer
            imageUrl={previewUrl}
            overlayUrl={overlayUrl}
            title={selectedDisease ? `Grad-CAM: ${selectedDisease}` : 'Original X-Ray'}
          />
        </div>
      )}

      {/* ── Tab: Longitudinal Comparison ─────────────────────────────── */}
      {activeTab === 'comparison' && showComparison && (
        <div className="card mb-4">
          <LongitudinalComparison
            currentResult={{ ...result, imageUrl: previewUrl }}
            priorResult={{ ...priorResult, imageUrl: priorResult?._previewUrl }}
          />
        </div>
      )}

      {/* Disclaimer */}
      <div className="divider" />
      <p className="text-sm text-muted text-center" style={{ marginTop: 24 }}>
        Research and educational use only. Not a certified diagnostic tool. All findings require radiologist review.
      </p>
    </main>
  );
}
