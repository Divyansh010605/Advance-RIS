import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';

const STATUS_COLORS = {
  PATHOLOGY_DETECTED: { bg: '#3d1a1a', text: '#f87171', dot: '#ef4444' },
  TB_DETECTED:        { bg: '#3d1a1a', text: '#f87171', dot: '#ef4444' },
  NORMAL:             { bg: '#0f2d1f', text: '#4ade80', dot: '#22c55e' },
  INCONCLUSIVE:       { bg: '#2d2207', text: '#fbbf24', dot: '#f59e0b' },
};

function badge(label) {
  const c = STATUS_COLORS[label] || STATUS_COLORS.INCONCLUSIVE;
  return (
    <span
      style={{
        background: c.bg,
        color: c.text,
        border: `1px solid ${c.dot}33`,
        padding: '3px 10px',
        borderRadius: 20,
        fontSize: 12,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        fontWeight: 600,
      }}
    >
      <span style={{ width: 7, height: 7, borderRadius: '50%', background: c.dot, display: 'inline-block' }} />
      {label.replace('_', ' ')}
    </span>
  );
}

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? '#ef4444' : pct >= 50 ? '#f59e0b' : '#22c55e';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <div style={{ flex: 1, height: 6, background: '#1e1e2e', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.6s ease' }} />
      </div>
      <span style={{ fontSize: 12, color: '#a0a0b8', minWidth: 34, textAlign: 'right' }}>{pct}%</span>
    </div>
  );
}

export default function HistoryPage() {
  const navigate = useNavigate();
  const [records, setRecords] = useState([]);
  const [fullRecords, setFullRecords] = useState({});
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('ALL');
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState(null);
  const limit = 10;

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({ page, limit });
      if (filter !== 'ALL') params.append('status', filter);
      if (search.trim()) params.append('search', search.trim());
      const res = await api.get(`/history?${params}`);
      setRecords(res.data.records || []);
      setTotal(res.data.total || 0);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  }, [page, filter, search]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  useEffect(() => {
    setPage(1);
  }, [filter, search]);

  const totalPages = Math.ceil(total / limit);

  const handleDownloadPDF = async (id) => {
    try {
      const response = await api.get(`/history/${id}/report/pdf`, {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Lung_Disease_Report_${id}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Failed to download report. Please try again.');
    }
  };

  const handleView = async (id) => {
    try {
      const res = await api.get(`/history/${id}`);
      const fullRecord = res.data;

      // Extract VLM findings & impressions
      const reportData = (fullRecord.finalReport?.findings && fullRecord.finalReport.findings.length > 0)
        ? {
            ...fullRecord.finalReport,
            status: fullRecord.reportStatus || 'APPROVED',
            detected_conditions: fullRecord.aiReport?.detectedConditions || [],
            model_confidence: fullRecord.aiReport?.modelConfidence,
          }
        : (fullRecord.aiReport || {});

      // Navigate to /results with complete payload so VLM report and visualizations load
      navigate('/results', {
        state: {
          result: {
            id: fullRecord._id,
            patientId: fullRecord.patientId,
            predictions: fullRecord.predictions || [],
            heatmaps: fullRecord.heatmaps || [],
            report: reportData,
            metadata: {
              model_version: fullRecord.modelVersion || 'DenseNet-121 + ResNet-50 + SwinCheX',
              ensemble_enabled: true,
              processing_time_ms: fullRecord.processingMs,
            },
            result: fullRecord.result || 'NORMAL',
            confidence: fullRecord.confidence,
            processingTimeMs: fullRecord.processingMs,
            originalImageBase64: fullRecord.originalImageBase64,
          },
          previewUrl: fullRecord.originalImageBase64
            ? `data:image/jpeg;base64,${fullRecord.originalImageBase64}`
            : null,
        },
      });
    } catch (err) {
      alert('Failed to load record details: ' + (err.response?.data?.message || err.message));
    }
  };

  const handleExpand = async (id) => {
    if (expanded === id) {
      setExpanded(null);
      return;
    }
    setExpanded(id);

    // Fetch full record if not cached
    if (!fullRecords[id]) {
      try {
        const res = await api.get(`/history/${id}`);
        setFullRecords((prev) => ({ ...prev, [id]: res.data }));
      } catch (err) {
        console.error('Failed to fetch full record details', err);
      }
    }
  };

  const styles = {
    page:    { minHeight: '100vh', background: '#0a0a0f', color: '#e0e0ef', fontFamily: "'Inter', sans-serif", padding: '32px 24px' },
    header:  { maxWidth: 1100, margin: '0 auto 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 },
    title:   { fontSize: 26, fontWeight: 700, color: '#fff', margin: 0 },
    sub:     { fontSize: 13, color: '#6b6b8a', marginTop: 4 },
    card:    { maxWidth: 1100, margin: '0 auto', background: '#111120', border: '1px solid #1e1e35', borderRadius: 16, overflow: 'hidden' },
    toolbar: { padding: '16px 24px', borderBottom: '1px solid #1e1e35', display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' },
    input:   { background: '#0a0a0f', border: '1px solid #2a2a45', borderRadius: 8, color: '#e0e0ef', padding: '8px 14px', fontSize: 13, outline: 'none', flex: 1, minWidth: 200 },
    pill: (active) => ({
      padding: '6px 14px',
      borderRadius: 20,
      fontSize: 12,
      fontWeight: 600,
      cursor: 'pointer',
      border: 'none',
      background: active ? '#3b82f6' : '#1a1a2e',
      color: active ? '#fff' : '#8080a0',
      transition: 'all 0.2s',
    }),
    table:   { width: '100%', borderCollapse: 'collapse' },
    th:      { padding: '12px 20px', textAlign: 'left', fontSize: 11, fontWeight: 600, color: '#6b6b8a', letterSpacing: '0.08em', textTransform: 'uppercase', borderBottom: '1px solid #1e1e35' },
    td:      { padding: '14px 20px', fontSize: 13, borderBottom: '1px solid #13131f', verticalAlign: 'middle' },
    btn:     { background: '#1a1a2e', border: '1px solid #2a2a45', borderRadius: 8, color: '#8080a0', padding: '6px 12px', fontSize: 12, cursor: 'pointer' },
    btnPri:  { background: '#3b82f6', border: 'none', borderRadius: 8, color: '#fff', padding: '8px 18px', fontSize: 13, fontWeight: 600, cursor: 'pointer' },
    empty:   { padding: 60, textAlign: 'center', color: '#4a4a6a' },
    expand:  { background: '#0d0d1a', borderBottom: '1px solid #1e1e35', padding: '20px 24px', display: 'grid', gridTemplateColumns: '300px 1fr', gap: 24 },
  };

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>Patient Diagnostic History</h1>
          <p style={styles.sub}>{total} study record{total !== 1 ? 's' : ''} on file</p>
        </div>
        <button style={styles.btnPri} onClick={() => navigate('/upload')}>
          + New X-Ray Screening
        </button>
      </div>

      <div style={styles.card}>
        {/* Toolbar */}
        <div style={styles.toolbar}>
          <input
            style={styles.input}
            placeholder="Search by filename or patient ID…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {['ALL', 'PATHOLOGY_DETECTED', 'NORMAL', 'INCONCLUSIVE'].map((f) => (
            <button key={f} style={styles.pill(filter === f)} onClick={() => setFilter(f)}>
              {f === 'ALL' ? 'All' : f.replace('_', ' ')}
            </button>
          ))}
          <button style={styles.btn} onClick={fetchHistory} title="Refresh">
            Refresh
          </button>
        </div>

        {/* Table */}
        {loading ? (
          <div style={styles.empty}>
            <div style={{ fontSize: 14, marginBottom: 12, opacity: 0.4 }}>...</div>
            Loading diagnostic records…
          </div>
        ) : error ? (
          <div style={{ ...styles.empty, color: '#f87171' }}>{error}</div>
        ) : records.length === 0 ? (
          <div style={styles.empty}>
            <div style={{ fontSize: 14, marginBottom: 12 }}>No Records Found</div>
            No chest radiographs found matching your criteria.
            <div style={{ marginTop: 16 }}>
              <button style={styles.btnPri} onClick={() => navigate('/upload')}>
                Upload First Radiograph
              </button>
            </div>
          </div>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                {['#', 'Date & Time', 'Filename', 'Primary Result', 'Confidence', 'Actions'].map((h) => (
                  <th key={h} style={styles.th}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.map((rec, i) => (
                <tr
                  key={rec._id}
                  style={{ cursor: 'pointer', background: expanded === rec._id ? '#0d0d1a' : 'transparent' }}
                  onClick={() => handleExpand(rec._id)}
                >
                  <td style={{ ...styles.td, color: '#4a4a6a', fontFamily: 'monospace' }}>
                    {(page - 1) * limit + i + 1}
                  </td>
                  <td style={styles.td}>
                    <div style={{ fontWeight: 500 }}>{new Date(rec.createdAt).toLocaleDateString()}</div>
                    <div style={{ fontSize: 11, color: '#6b6b8a' }}>{new Date(rec.createdAt).toLocaleTimeString()}</div>
                  </td>
                  <td style={{ ...styles.td, fontFamily: 'monospace', fontSize: 12, color: '#a0a0c0' }}>
                    {rec.filename || rec.originalName || '—'}
                  </td>
                  <td style={styles.td}>{badge(rec.result || 'INCONCLUSIVE')}</td>
                  <td style={{ ...styles.td, minWidth: 140 }}>
                    <ConfidenceBar value={rec.confidence || 0} />
                  </td>
                  <td style={styles.td}>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        style={styles.btn}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleView(rec._id);
                        }}
                      >
                        View
                      </button>
                      <button
                        style={styles.btn}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDownloadPDF(rec._id);
                        }}
                      >
                        PDF
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* Expanded Row Detail Overlay */}
        {expanded && (() => {
          const full = fullRecords[expanded];
          const firstOverlay = full?.heatmaps?.[0]?.overlay || full?.overlayBase64 || full?.gradcamBase64;
          const heatmapSrc = firstOverlay ? `data:image/jpeg;base64,${firstOverlay}` : null;
          const origSrc = full?.originalImageBase64 ? `data:image/jpeg;base64,${full.originalImageBase64}` : null;

          const report = (full?.finalReport?.findings && full.finalReport.findings.length > 0)
            ? full.finalReport
            : full?.aiReport || {};

          const findings = Array.isArray(report.findings) ? report.findings : [];
          const impression = Array.isArray(report.impression) ? report.impression : [];

          return (
            <div style={styles.expand}>
              {/* Left Column: Visual Radiograph & Heatmap */}
              <div>
                <div style={{ fontSize: 11, color: '#6b6b8a', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600 }}>
                  LayerCAM Explainability Heatmap
                </div>
                {!full ? (
                  <div style={{ color: '#6b6b8a', fontSize: 13 }}>Loading study images...</div>
                ) : heatmapSrc ? (
                  <img
                    src={heatmapSrc}
                    alt="LayerCAM Heatmap"
                    style={{ width: '100%', maxHeight: 260, objectFit: 'contain', borderRadius: 8, border: '1px solid #2a2a45' }}
                  />
                ) : origSrc ? (
                  <img
                    src={origSrc}
                    alt="Chest Radiograph"
                    style={{ width: '100%', maxHeight: 260, objectFit: 'contain', borderRadius: 8, border: '1px solid #2a2a45' }}
                  />
                ) : (
                  <div style={{ color: '#4a4a6a', fontSize: 13 }}>No radiograph image on record.</div>
                )}
                <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
                  <button style={styles.btnPri} onClick={() => handleView(expanded)}>
                    Open Full Results & Editor
                  </button>
                  <button style={styles.btn} onClick={() => handleDownloadPDF(expanded)}>
                    Download PDF
                  </button>
                </div>
              </div>

              {/* Right Column: VLM Radiology Findings & Impressions */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>
                    Automated Radiology Report (VLM Grounded)
                  </div>
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      padding: '2px 8px',
                      borderRadius: 6,
                      background: full?.reportStatus === 'APPROVED' ? '#16a34a25' : '#3b82f625',
                      color: full?.reportStatus === 'APPROVED' ? '#4ade80' : '#60a5fa',
                      border: `1px solid ${full?.reportStatus === 'APPROVED' ? '#16a34a' : '#3b82f6'}`,
                    }}
                  >
                    {full?.reportStatus || 'DRAFT'}
                  </span>
                </div>

                {/* Findings Box */}
                <div style={{ background: '#121224', padding: '12px 16px', borderRadius: 8, border: '1px solid #1e1e35' }}>
                  <div style={{ fontSize: 11, color: '#60a5fa', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                    Radiological Findings
                  </div>
                  {findings.length > 0 ? (
                    <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
                      {findings.map((f, idx) => (
                        <li key={idx}>{f}</li>
                      ))}
                    </ul>
                  ) : (
                    <div style={{ fontSize: 12, color: '#64748b' }}>No focal acute pathology identified. Clear bilateral lung fields.</div>
                  )}
                </div>

                {/* Impressions Box */}
                <div style={{ background: '#121224', padding: '12px 16px', borderRadius: 8, border: '1px solid #1e1e35' }}>
                  <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 600, textTransform: 'uppercase', marginBottom: 6 }}>
                    Clinical Impression
                  </div>
                  {impression.length > 0 ? (
                    <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
                      {impression.map((imp, idx) => (
                        <li key={idx} style={{ fontWeight: 600 }}>{imp}</li>
                      ))}
                    </ul>
                  ) : (
                    <div style={{ fontSize: 12, color: '#64748b' }}>No acute cardiopulmonary disease.</div>
                  )}
                </div>

                {/* Details Footer */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, paddingTop: 4 }}>
                  <Detail label="Scan ID" value={expanded} mono />
                  <Detail label="Patient ID" value={full?.patientId || 'N/A'} />
                  <Detail label="Model Pipeline" value={full?.modelVersion || 'Tri-Model Ensemble'} />
                  <Detail label="Inference Time" value={full?.processingMs ? `${full.processingMs} ms` : '—'} />
                </div>
              </div>
            </div>
          );
        })()}

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ padding: '16px 24px', borderTop: '1px solid #1e1e35', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 12, color: '#6b6b8a' }}>
              Page {page} of {totalPages} · {total} total
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button style={styles.btn} disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
                Prev
              </button>
              {Array.from({ length: Math.min(5, totalPages) }, (_, k) => {
                const p = Math.max(1, Math.min(page - 2, totalPages - 4)) + k;
                return (
                  <button
                    key={p}
                    style={{
                      ...styles.btn,
                      background: p === page ? '#3b82f6' : '#1a1a2e',
                      color: p === page ? '#fff' : '#8080a0',
                      border: 'none',
                    }}
                    onClick={() => setPage(p)}
                  >
                    {p}
                  </button>
                );
              })}
              <button style={styles.btn} disabled={page === totalPages} onClick={() => setPage((p) => p + 1)}>
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Detail({ label, value, mono }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: '#6b6b8a', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 12, color: '#c0c0d8', fontFamily: mono ? 'monospace' : 'inherit', wordBreak: 'break-all' }}>{value}</div>
    </div>
  );
}
