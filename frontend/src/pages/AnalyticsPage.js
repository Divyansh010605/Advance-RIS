import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';

/* ── Tiny Donut Chart ─────────────────────────────────────────── */
function DonutChart({ pathologyCount, normalCount, total }) {
  const path = total ? pathologyCount / total : 0;
  const nor = total ? normalCount / total : 0;
  const r = 52, cx = 70, cy = 70, stroke = 14;
  const circ = 2 * Math.PI * r;
  const pathArc = circ * path;
  const norArc = circ * nor;
  const incArc = circ * Math.max(0, 1 - path - nor);

  return (
    <svg width="140" height="140" viewBox="0 0 140 140">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#1e1e35" strokeWidth={stroke} />
      {/* Inconclusive */}
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke="#f59e0b" strokeWidth={stroke}
        strokeDasharray={`${incArc} ${circ}`}
        strokeDashoffset={-circ * (path + nor)}
        style={{ transition: 'stroke-dasharray 0.8s ease', transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px` }}
      />
      {/* Normal */}
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke="#22c55e" strokeWidth={stroke}
        strokeDasharray={`${norArc} ${circ}`}
        strokeDashoffset={-circ * path}
        style={{ transition: 'stroke-dasharray 0.8s ease', transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px` }}
      />
      {/* Pathology Detected */}
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke="#ef4444" strokeWidth={stroke}
        strokeDasharray={`${pathArc} ${circ}`}
        style={{ transition: 'stroke-dasharray 0.8s ease', transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px` }}
      />
      <text x={cx} y={cy - 6} textAnchor="middle" fill="#fff" fontSize="20" fontWeight="700">{total}</text>
      <text x={cx} y={cy + 14} textAnchor="middle" fill="#6b6b8a" fontSize="10">Total Scans</text>
    </svg>
  );
}

/* ── Horizontal Pathology Breakdown Bar ────────────────────────── */
function PathologyBarChart({ distribution = [] }) {
  if (!distribution || distribution.length === 0) {
    return (
      <div style={{ color: '#6b6b8a', fontSize: 13, padding: '24px 0', textAlign: 'center' }}>
        No pathologies detected in this time window.
      </div>
    );
  }

  const maxCount = Math.max(...distribution.map((d) => d.count), 1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
      {distribution.map((item) => {
        const pct = Math.round((item.count / maxCount) * 100);
        return (
          <div key={item.disease} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
              <span style={{ fontWeight: 600, color: '#f1f5f9' }}>{item.disease.replace('_', ' ')}</span>
              <span style={{ color: '#94a3b8' }}>
                <strong style={{ color: '#ef4444' }}>{item.count}</strong> scan{item.count !== 1 ? 's' : ''} · Avg Conf: {item.avgConfidence}%
              </span>
            </div>
            <div style={{ height: 8, background: '#1e1e2e', borderRadius: 4, overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${pct}%`,
                  background: 'linear-gradient(90deg, #ef4444, #f87171)',
                  borderRadius: 4,
                  transition: 'width 0.6s ease',
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Timeline Bar Chart ────────────────────────────────────────── */
function TimelineChart({ data }) {
  if (!data || data.length === 0) {
    return <div style={{ color: '#4a4a6a', fontSize: 13, padding: 20 }}>No scans recorded in this window.</div>;
  }
  const max = Math.max(...data.map((d) => d.count), 1);
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 120, padding: '0 4px' }}>
      {data.map((d, i) => (
        <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
          <div style={{ fontSize: 10, color: '#6b6b8a' }}>{d.count}</div>
          <div
            style={{
              width: '100%',
              height: `${(d.count / max) * 90}px`,
              background: 'linear-gradient(180deg, #3b82f6, #1d4ed8)',
              borderRadius: '3px 3px 0 0',
              minHeight: 2,
              transition: 'height 0.6s ease',
            }}
          />
          <div style={{ fontSize: 9, color: '#6b6b8a', textAlign: 'center', whiteSpace: 'nowrap', overflow: 'hidden', maxWidth: '100%' }}>
            {d.label}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Confidence Buckets ────────────────────────────────────────── */
function ConfidenceBuckets({ buckets }) {
  if (!buckets) return null;
  const labels = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'];
  const colors = ['#6366f1', '#8b5cf6', '#3b82f6', '#f59e0b', '#22c55e'];
  const max = Math.max(...buckets, 1);
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end', height: 100 }}>
      {buckets.map((v, i) => (
        <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
          <div style={{ fontSize: 10, color: '#6b6b8a' }}>{v}</div>
          <div
            style={{
              width: '100%',
              height: `${(v / max) * 80}px`,
              background: colors[i],
              borderRadius: '3px 3px 0 0',
              minHeight: 2,
              transition: 'height 0.6s',
            }}
          />
          <div style={{ fontSize: 9, color: '#6b6b8a', textAlign: 'center' }}>{labels[i]}</div>
        </div>
      ))}
    </div>
  );
}

/* ── Stat Card Component ───────────────────────────────────────── */
function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: '#111120', border: '1px solid #1e1e35', borderRadius: 12, padding: '20px 24px', flex: 1, minWidth: 150 }}>
      <div style={{ fontSize: 11, color: '#6b6b8a', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>{label}</div>
      <div style={{ fontSize: 30, fontWeight: 700, color: accent || '#fff', lineHeight: 1 }}>{value ?? '—'}</div>
      {sub && <div style={{ fontSize: 12, color: '#6b6b8a', marginTop: 6 }}>{sub}</div>}
    </div>
  );
}

/* ── Main Analytics Page ───────────────────────────────────────── */
export default function AnalyticsPage() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [range, setRange] = useState('30d');

  useEffect(() => {
    setLoading(true);
    api
      .get(`/analytics?range=${range}`)
      .then((r) => {
        setStats(r.data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.response?.data?.message || 'Failed to load analytics');
        setLoading(false);
      });
  }, [range]);

  const s = {
    page:    { minHeight: '100vh', background: '#0a0a0f', color: '#e0e0ef', fontFamily: "'Inter', sans-serif", padding: '32px 24px' },
    wrap:    { maxWidth: 1100, margin: '0 auto' },
    heading: { fontSize: 26, fontWeight: 700, color: '#fff', marginBottom: 4 },
    sub:     { fontSize: 13, color: '#6b6b8a', marginBottom: 32 },
    row:     { display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 },
    panel:   { background: '#111120', border: '1px solid #1e1e35', borderRadius: 12, padding: '20px 24px' },
    pTitle:  { fontSize: 14, fontWeight: 700, color: '#f1f5f9', marginBottom: 16 },
    pill: (active) => ({
      padding: '5px 14px',
      borderRadius: 20,
      fontSize: 12,
      fontWeight: 600,
      cursor: 'pointer',
      border: 'none',
      background: active ? '#3b82f6' : '#1a1a2e',
      color: active ? '#fff' : '#8080a0',
    }),
  };

  if (loading) {
    return (
      <div style={{ ...s.page, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: '#6b6b8a', fontSize: 14 }}>Loading Multi-Disease screening metrics…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ ...s.page, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: '#f87171', fontSize: 14 }}>{error}</div>
      </div>
    );
  }

  const total = stats?.totalScans || 0;
  const pathCount = stats?.pathologyCount || 0;
  const normalCount = stats?.normalCount || 0;
  const pathPct = total ? Math.round((pathCount / total) * 100) : 0;
  const norPct = total ? Math.round((normalCount / total) * 100) : 0;

  return (
    <div style={s.page}>
      <div style={s.wrap}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 32 }}>
          <div>
            <h1 style={s.heading}>Multi-Disease Screening Analytics</h1>
            <p style={{ ...s.sub, marginBottom: 0 }}>
              NIH ChestX-ray14 · Tri-Model Consensus (SwinCheX + DenseNet-121 + ResNet-50) · Performance & Trends
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {['7d', '30d', '90d', 'all'].map((r) => (
              <button key={r} style={s.pill(range === r)} onClick={() => setRange(r)}>
                {r.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* KPI Row */}
        <div style={s.row}>
          <StatCard label="Total Scans" value={total} sub={`Evaluated in last ${range}`} />
          <StatCard
            label="Pathology Detected"
            value={pathCount}
            sub={`${pathPct}% of total scans`}
            accent="#ef4444"
          />
          <StatCard
            label="Normal Scans"
            value={normalCount}
            sub={`${norPct}% clear lungs`}
            accent="#22c55e"
          />
          <StatCard
            label="Consensus Rate"
            value={`${stats?.consensusRate ?? 100}%`}
            sub="≥2 models agreement"
            accent="#38bdf8"
          />
          <StatCard
            label="Avg Confidence"
            value={stats?.avgConfidence != null ? `${Math.round(stats.avgConfidence * 100)}%` : '—'}
            sub="Across all evaluated classes"
            accent="#a855f7"
          />
          <StatCard
            label="Avg Latency"
            value={stats?.avgProcessingMs != null ? `${Math.round(stats.avgProcessingMs)} ms` : '—'}
            sub="AI inference + LayerCAM"
          />
        </div>

        {/* Row 2: Donut + Tri-Model Consensus Breakdown */}
        <div style={s.row}>
          {/* Classification Donut */}
          <div style={{ ...s.panel, flex: '1 1 320px' }}>
            <div style={s.pTitle}>Primary Screening Classification</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
              <DonutChart pathologyCount={pathCount} normalCount={normalCount} total={total} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, flex: 1 }}>
                {[
                  { color: '#ef4444', label: 'Pathology Detected', pct: pathPct, n: pathCount },
                  { color: '#22c55e', label: 'Normal / Clear Lung', pct: norPct, n: normalCount },
                  {
                    color: '#f59e0b',
                    label: 'Inconclusive / Review',
                    pct: Math.max(0, 100 - pathPct - norPct),
                    n: stats?.inconclusiveCount || 0,
                  },
                ].map(({ color, label, pct, n }) => (
                  <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }} />
                    <span style={{ fontSize: 12, color: '#a0a0c0' }}>{label}</span>
                    <span style={{ fontSize: 12, color: '#fff', marginLeft: 'auto' }}>
                      {n ?? 0} <span style={{ color: '#6b6b8a' }}>({pct}%)</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Tri-Model Consensus Card */}
          <div style={{ ...s.panel, flex: '1 1 320px' }}>
            <div style={s.pTitle}>Tri-Model Consensus Distribution</div>
            <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 14 }}>
              Agreement across SwinCheX (ViT), DenseNet-121 (CNN), and ResNet-50
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                {
                  label: '3/3 Models Agree (Unanimous)',
                  count: stats?.consensusBreakdown?.unanimous3 || 0,
                  color: '#22c55e',
                  desc: 'High clinical certainty',
                },
                {
                  label: '2/3 Models Agree (Consensus Majority)',
                  count: stats?.consensusBreakdown?.majority2 || 0,
                  color: '#38bdf8',
                  desc: 'Declared pathology (≥2 models)',
                },
                {
                  label: '1/3 Model Flagged (Borderline)',
                  count: stats?.consensusBreakdown?.single1 || 0,
                  color: '#f59e0b',
                  desc: 'Sub-threshold attenuation',
                },
              ].map((c) => (
                <div key={c.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#181826', padding: '8px 12px', borderRadius: 8 }}>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: c.color }}>{c.label}</div>
                    <div style={{ fontSize: 10, color: '#6b6b8a' }}>{c.desc}</div>
                  </div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#f1f5f9' }}>{c.count}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Row 3: Pathology Distribution across 14 NIH findings */}
        <div style={{ ...s.panel, marginBottom: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={s.pTitle}>Top Detected Pathologies (NIH ChestX-ray14)</div>
            <span style={{ fontSize: 12, color: '#6b6b8a' }}>Ranked by prevalence</span>
          </div>
          <PathologyBarChart distribution={stats?.pathologyDistribution || []} />
        </div>

        {/* Row 4: Confidence distribution + Scans over time */}
        <div style={s.row}>
          <div style={{ ...s.panel, flex: 1, minWidth: 260 }}>
            <div style={s.pTitle}>Confidence Score Distribution</div>
            <ConfidenceBuckets buckets={stats?.confidenceBuckets} />
            <div style={{ marginTop: 12, fontSize: 11, color: '#6b6b8a' }}>
              Ensemble soft-vote confidence score distribution across all scanned cases.
            </div>
          </div>

          <div style={{ ...s.panel, flex: 1, minWidth: 260 }}>
            <div style={s.pTitle}>Daily Screening Volume</div>
            <TimelineChart data={stats?.timeline || []} />
          </div>
        </div>

        {/* Row 5: Recent Multi-Disease Studies Table */}
        <div style={s.panel}>
          <div style={s.pTitle}>Recent Clinical Studies</div>
          {stats?.recentScans?.length ? (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Date & Time', 'Filename', 'Primary Result', 'Detected Pathologies', 'Confidence', 'Action'].map((h) => (
                    <th
                      key={h}
                      style={{
                        textAlign: 'left',
                        fontSize: 11,
                        color: '#6b6b8a',
                        padding: '0 8px 10px',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        letterSpacing: '0.07em',
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stats.recentScans.map((r) => {
                  const isPath = r.result === 'PATHOLOGY_DETECTED' || r.result === 'TB_DETECTED';
                  return (
                    <tr key={r._id} style={{ borderTop: '1px solid #1e1e35' }}>
                      <td style={{ fontSize: 12, color: '#6b6b8a', padding: '10px 8px' }}>
                        {new Date(r.createdAt).toLocaleString()}
                      </td>
                      <td style={{ fontSize: 12, color: '#cbd5e1', padding: '10px 8px', fontFamily: 'monospace' }}>
                        {r.filename}
                      </td>
                      <td style={{ padding: '10px 8px' }}>
                        <span
                          style={{
                            fontSize: 11,
                            fontWeight: 700,
                            padding: '3px 8px',
                            borderRadius: 12,
                            background: isPath ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                            color: isPath ? '#f87171' : '#4ade80',
                            border: `1px solid ${isPath ? '#ef444455' : '#22c55e55'}`,
                          }}
                        >
                          {r.result.replace('_', ' ')}
                        </span>
                      </td>
                      <td style={{ padding: '10px 8px' }}>
                        {r.detectedPathologies && r.detectedPathologies.length > 0 ? (
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                            {r.detectedPathologies.map((d, idx) => (
                              <span
                                key={idx}
                                style={{
                                  fontSize: 11,
                                  fontWeight: 600,
                                  background: '#27273a',
                                  color: '#e2e8f0',
                                  padding: '2px 6px',
                                  borderRadius: 4,
                                }}
                              >
                                {d.disease} ({d.prob}%)
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span style={{ fontSize: 11, color: '#64748b' }}>No focal pathology</span>
                        )}
                      </td>
                      <td style={{ fontSize: 12, color: '#a0a0c0', padding: '10px 8px' }}>
                        {r.confidence}%
                      </td>
                      <td style={{ padding: '10px 8px' }}>
                        <button
                          style={{
                            background: '#1e293b',
                            border: '1px solid #334155',
                            color: '#93c5fd',
                            padding: '4px 10px',
                            borderRadius: 6,
                            fontSize: 11,
                            cursor: 'pointer',
                          }}
                          onClick={() => navigate('/history')}
                        >
                          View in History
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <div style={{ color: '#4a4a6a', fontSize: 13, padding: '20px 0' }}>No screening studies recorded yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}
