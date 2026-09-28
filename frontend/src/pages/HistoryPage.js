import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';

const STATUS_COLORS = {
  TB_DETECTED:    { bg: '#3d1a1a', text: '#f87171', dot: '#ef4444' },
  NORMAL:         { bg: '#0f2d1f', text: '#4ade80', dot: '#22c55e' },
  INCONCLUSIVE:   { bg: '#2d2207', text: '#fbbf24', dot: '#f59e0b' },
};

function badge(label) {
  const c = STATUS_COLORS[label] || STATUS_COLORS.INCONCLUSIVE;
  return (
    <span style={{
      background: c.bg, color: c.text,
      border: `1px solid ${c.dot}33`,
      padding: '3px 10px', borderRadius: 20, fontSize: 12,
      display: 'inline-flex', alignItems: 'center', gap: 6, fontWeight: 600,
    }}>
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
  const [records, setRecords]     = useState([]);
  const [fullRecords, setFullRecords] = useState({});
  const [total, setTotal]         = useState(0);
  const [page, setPage]           = useState(1);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState('');
  const [filter, setFilter]       = useState('ALL');
  const [search, setSearch]       = useState('');
  const [expanded, setExpanded]   = useState(null);
  const limit = 10;

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({ page, limit });
      if (filter !== 'ALL') params.append('status', filter);
      if (search.trim())    params.append('search', search.trim());
      const res = await api.get(`/history?${params}`);
      setRecords(res.data.records || []);
      setTotal(res.data.total || 0);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  }, [page, filter, search]);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);
  useEffect(() => { setPage(1); }, [filter, search]);

  const totalPages = Math.ceil(total / limit);
  
  const handleDownloadPDF = async (id) => {
    try {
      const response = await api.get(`/history/${id}/report/pdf`, {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Lung_Disease_Report_${id}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      alert('Failed to download report. Please try again.');
    }
  };

  const handleView = async (id) => {
    try {
      const res = await api.get(`/history/${id}`);
      const fullRecord = res.data;
      navigate('/results', {
        state: {
          result: {
            id: fullRecord._id,
            prediction: fullRecord.result === 'TB_DETECTED' ? fullRecord.confidence : 1 - fullRecord.confidence,
            confidence: Math.round(fullRecord.confidence * 100),
            processingTimeMs: fullRecord.processingMs,
            heatmap: fullRecord.gradcamBase64,
            overlay: fullRecord.overlayBase64,
            label: fullRecord.result === 'TB_DETECTED' ? 'TB Positive' : 'TB Negative',
          },
          previewUrl: fullRecord.originalImageBase64 ? `data:image/jpeg;base64,${fullRecord.originalImageBase64}` : null,
        }
      });
    } catch (err) {
      alert('Failed to load record details.');
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
        setFullRecords(prev => ({ ...prev, [id]: res.data }));
      } catch (err) {
        console.error("Failed to fetch full record details", err);
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
    pill:    (active) => ({
      padding: '6px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600, cursor: 'pointer', border: 'none',
      background: active ? '#6366f1' : '#1a1a2e', color: active ? '#fff' : '#8080a0', transition: 'all 0.2s',
    }),
    table:   { width: '100%', borderCollapse: 'collapse' },
    th:      { padding: '12px 20px', textAlign: 'left', fontSize: 11, fontWeight: 600, color: '#6b6b8a', letterSpacing: '0.08em', textTransform: 'uppercase', borderBottom: '1px solid #1e1e35' },
    td:      { padding: '14px 20px', fontSize: 13, borderBottom: '1px solid #13131f', verticalAlign: 'middle' },
    btn:     { background: '#1a1a2e', border: '1px solid #2a2a45', borderRadius: 8, color: '#8080a0', padding: '6px 12px', fontSize: 12, cursor: 'pointer' },
    btnPri:  { background: '#6366f1', border: 'none', borderRadius: 8, color: '#fff', padding: '8px 18px', fontSize: 13, fontWeight: 600, cursor: 'pointer' },
    empty:   { padding: 60, textAlign: 'center', color: '#4a4a6a' },
    expand:  { background: '#0d0d1a', borderBottom: '1px solid #1e1e35', padding: '16px 24px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 },
  };

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>Patient History</h1>
          <p style={styles.sub}>{total} scan{total !== 1 ? 's' : ''} on record</p>
        </div>
        <button style={styles.btnPri} onClick={() => navigate('/upload')}>+ New Scan</button>
      </div>

      <div style={styles.card}>
        {/* Toolbar */}
        <div style={styles.toolbar}>
          <input
            style={styles.input}
            placeholder="Search by patient ID or filename…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          {['ALL', 'TB_DETECTED', 'NORMAL', 'INCONCLUSIVE'].map(f => (
            <button key={f} style={styles.pill(filter === f)} onClick={() => setFilter(f)}>
              {f === 'ALL' ? 'All' : f.replace('_', ' ')}
            </button>
          ))}
          <button style={styles.btn} onClick={fetchHistory} title="Refresh">Refresh</button>
        </div>

        {/* Table */}
        {loading ? (
          <div style={styles.empty}>
            <div style={{ fontSize: 14, marginBottom: 12, opacity: 0.4 }}>...</div>
            Loading scans…
          </div>
        ) : error ? (
          <div style={{ ...styles.empty, color: '#f87171' }}>{error}</div>
        ) : records.length === 0 ? (
          <div style={styles.empty}>
            <div style={{ fontSize: 14, marginBottom: 12 }}>No Data</div>
            No scans found
            <div style={{ marginTop: 16 }}>
              <button style={styles.btnPri} onClick={() => navigate('/upload')}>Upload first X-Ray</button>
            </div>
          </div>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                {['#', 'Date & Time', 'Filename', 'Result', 'Confidence', 'Actions'].map(h => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.map((rec, i) => (
                <>
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
                          onClick={e => { e.stopPropagation(); handleView(rec._id); }}
                        >View</button>
                        <button
                          style={styles.btn}
                          onClick={e => { e.stopPropagation(); handleDownloadPDF(rec._id); }}
                        >PDF</button>
                      </div>
                    </td>
                  </tr>

                  {/* Expanded detail row */}
                  {expanded === rec._id && (() => {
                    const full = fullRecords[rec._id];
                    const heatmapSrc = full?.gradcamBase64
                      ? `data:image/jpeg;base64,${full.gradcamBase64}`
                      : null;
                    return (
                      <tr key={`${rec._id}-exp`}>
                        <td colSpan={6} style={{ padding: 0, background: '#0d0d1a' }}>
                          <div style={styles.expand}>
                            <div>
                              <div style={{ fontSize: 11, color: '#6b6b8a', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Grad-CAM Heatmap</div>
                              {!full ? (
                                <div style={{ color: '#6b6b8a', fontSize: 13 }}>Loading...</div>
                              ) : heatmapSrc ? (
                                <img src={heatmapSrc} alt="Grad-CAM" style={{ width: '100%', maxWidth: 280, borderRadius: 8, border: '1px solid #2a2a45' }} />
                              ) : (
                                <div style={{ color: '#4a4a6a', fontSize: 13 }}>No heatmap available</div>
                              )}
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                              <Detail label="Scan ID"        value={rec._id} mono />
                              <Detail label="Patient ID"     value={full?.patientId || rec.patientId || '—'} />
                              <Detail label="Model version"  value={full?.modelVersion || rec.modelVersion || 'DenseNet-121 v1.0'} />
                              <Detail label="Processing time" value={rec.processingMs ? `${rec.processingMs} ms` : '—'} />
                              <Detail label="Top class"      value={rec.result || '—'} />
                              <Detail label="Raw confidence" value={rec.confidence != null ? `${(rec.confidence * 100).toFixed(2)}%` : '—'} />
                            </div>
                          </div>
                        </td>
                      </tr>
                    );
                  })()}
                </>
              ))}
            </tbody>
          </table>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ padding: '16px 24px', borderTop: '1px solid #1e1e35', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: 12, color: '#6b6b8a' }}>
              Page {page} of {totalPages} · {total} total
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button style={styles.btn} disabled={page === 1} onClick={() => setPage(p => p - 1)}>Prev</button>
              {Array.from({ length: Math.min(5, totalPages) }, (_, k) => {
                const p = Math.max(1, Math.min(page - 2, totalPages - 4)) + k;
                return (
                  <button key={p} style={{ ...styles.btn, background: p === page ? '#6366f1' : '#1a1a2e', color: p === page ? '#fff' : '#8080a0', border: 'none' }} onClick={() => setPage(p)}>{p}</button>
                );
              })}
              <button style={styles.btn} disabled={page === totalPages} onClick={() => setPage(p => p + 1)}>Next</button>
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