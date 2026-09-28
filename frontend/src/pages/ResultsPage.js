import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import MultiDiseaseResults from '../components/MultiDiseaseResults';
import HeatmapGallery from '../components/HeatmapGallery';
import ReportEditor from '../components/ReportEditor';

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, previewUrl } = location.state || {};

  useEffect(() => {
    if (!result) {
      navigate('/upload');
    }
  }, [result, navigate]);

  if (!result) return null;

  const predictions = result.predictions || [];
  const heatmaps = result.heatmaps || (result.heatmap ? [{ disease: 'Tuberculosis', heatmap: result.heatmap, overlay: result.overlay }] : []);
  const report = result.report || {};
  const scanId = result.id;

  const detectedCount = predictions.filter((p) => p.detected).length;
  const isNormal = detectedCount === 0;

  return (
    <main className="results-page container" role="main" style={{ paddingTop: 24, paddingBottom: 48 }}>
      <header className="page-header" style={{ marginBottom: 24 }}>
        <h1>Multi-Disease Screening & Report</h1>
        <p className="text-muted">
          Chest X-Ray multi-label pathology classification with per-disease Grad-CAM explainability
        </p>
      </header>

      {/* Primary Status Banner */}
      <div
        className="card mb-4"
        style={{
          borderLeft: `6px solid ${isNormal ? '#22c55e' : '#ef4444'}`,
          background: isNormal ? 'rgba(34, 197, 94, 0.05)' : 'rgba(239, 68, 68, 0.05)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <span
              style={{
                fontSize: '0.85rem',
                fontWeight: 700,
                padding: '4px 10px',
                borderRadius: '12px',
                background: isNormal ? '#22c55e' : '#ef4444',
                color: '#ffffff',
                textTransform: 'uppercase',
              }}
            >
              {isNormal ? 'Normal / No Acute Abnormality' : `${detectedCount} Pathology Detected`}
            </span>
            <p className="text-sm text-muted mt-2 mb-0">
              Patient ID: <strong>{result.patientId || 'N/A'}</strong> · Processing Time:{' '}
              <strong>{result.processingTimeMs ? `${result.processingTimeMs} ms` : '—'}</strong>
            </p>
          </div>
          <button className="btn btn-ghost" onClick={() => navigate('/upload')}>
            Analyze Another Image
          </button>
        </div>
      </div>

      {/* 1. Multi-Disease Pathology Cards */}
      <MultiDiseaseResults predictions={predictions} />

      {/* 2. Per-Disease Grad-CAM Heatmaps */}
      <HeatmapGallery heatmaps={heatmaps} previewUrl={previewUrl} />

      {/* 3. Interactive Radiology Report Editor (Human-in-the-loop) */}
      <ReportEditor scanId={scanId} initialReport={report} initialStatus={report.status || 'DRAFT'} />

      {/* Disclaimer */}
      <div className="divider" />
      <p className="text-sm text-muted text-center" style={{ marginTop: 24 }}>
        This system is for research and educational purposes only and is not a certified medical diagnostic tool.
      </p>
    </main>
  );
}
