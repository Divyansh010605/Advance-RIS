import { useState, useEffect } from 'react';
import api from '../services/api';

/* ── tiny chart helpers (no extra lib needed) ─────────────────── */
function DonutChart({ tbCount, normalCount, total }) {
  const tb  = total ? tbCount / total  : 0;
  const nor = total ? normalCount / total : 0;
  const r = 52, cx = 70, cy = 70, stroke = 14;
  const circ = 2 * Math.PI * r;
  const tbArc  = circ * tb;
  const norArc = circ * nor;
  const incArc = circ * Math.max(0, 1 - tb - nor);
  return (
    <svg width="140" height="140" viewBox="0 0 140 140">
      {/* bg ring */}
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#1e1e35" strokeWidth={stroke} />
      {/* inconclusive */}
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#f59e0b" strokeWidth={stroke}
        strokeDasharray={`${incArc} ${circ}`}
        strokeDashoffset={-circ * (tb + nor)}
        style={{ transition: 'stroke-dasharray 0.8s ease', transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px` }} />
      {/* normal */}
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#22c55e" strokeWidth={stroke}
        strokeDasharray={`${norArc} ${circ}`}
        strokeDashoffset={-circ * tb}
        style={{ transition: 'stroke-dasharray 0.8s ease', transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px` }} />
      {/* tb */}
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#ef4444" strokeWidth={stroke}
        strokeDasharray={`${tbArc} ${circ}`}
        style={{ transition: 'stroke-dasharray 0.8s ease', transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px` }} />
      <text x={cx} y={cy - 6} textAnchor="middle" fill="#fff" fontSize="20" fontWeight="700">{total}</text>
      <text x={cx} y={cy + 14} textAnchor="middle" fill="#6b6b8a" fontSize="10">Total scans</text>
    </svg>
  );
}

function BarChart({ data }) {
  if (!data || data.length === 0) return <div style={{ color: '#4a4a6a', fontSize: 13, padding: 20 }}>No data yet</div>;
  const max = Math.max(...data.map(d => d.count), 1);
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 120, padding: '0 4px' }}>
      {data.map((d, i) => (
        <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
          <div style={{ fontSize: 10, color: '#6b6b8a' }}>{d.count}</div>
          <div style={{
            width: '100%', height: `${(d.count / max) * 90}px`,
            background: 'linear-gradient(180deg, #6366f1, #4f46e5)',
            borderRadius: '3px 3px 0 0', minHeight: 2,
            transition: 'height 0.6s ease',
          }} />
          <div style={{ fontSize: 9, color: '#6b6b8a', textAlign: 'center', whiteSpace: 'nowrap', overflow: 'hidden', maxWidth: '100%' }}>{d.label}</div>
        </div>
      ))}
    </div>
  );
}

function ConfidenceBuckets({ buckets }) {
  if (!buckets) return null;
  const labels = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'];
  const colors  = ['#6366f1', '#8b5cf6', '#f59e0b', '#f97316', '#ef4444'];
  const max = Math.max(...buckets, 1);
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end', height: 100 }}>
      {buckets.map((v, i) => (
        <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
          <div style={{ fontSize: 10, color: '#6b6b8a' }}>{v}</div>
          <div style={{ width: '100%', height: `${(v / max) * 80}px`, background: colors[i], borderRadius: '3px 3px 0 0', minHeight: 2, transition: 'height 0.6s' }} />
          <div style={{ fontSize: 9, color: '#6b6b8a', textAlign: 'center' }}>{labels[i]}</div>
        </div>
      ))}
    </div>
  );
}

/* ── StatCard ─────────────────────────────────────────────────── */
function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: '#111120', border: '1px solid #1e1e35', borderRadius: 12, padding: '20px 24px', flex: 1, minWidth: 140 }}>
      <div style={{ fontSize: 11, color: '#6b6b8a', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>{label}</div>
      <div style={{ fontSize: 32, fontWeight: 700, color: accent || '#fff', lineHeight: 1 }}>{value ?? '—'}</div>
      {sub && <div style={{ fontSize: 12, color: '#6b6b8a', marginTop: 6 }}>{sub}</div>}
    </div>
  );
}

/* ── main ─────────────────────────────────────────────────────── */
export default function AnalyticsPage() {
  const [stats, setStats]       = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');
  const [range, setRange]       = useState('30d');

  useEffect(() => {
    setLoading(true);
    api.get(`/analytics?range=${range}`)
      .then(r => { setStats(r.data); setLoading(false); })
      .catch(e => { setError(e.response?.data?.message || 'Failed to load analytics'); setLoading(false); });
  }, [range]);

  const s = {
    page:    { minHeight: '100vh', background: '#0a0a0f', color: '#e0e0ef', fontFamily: "'Inter', sans-serif", padding: '32px 24px' },
    wrap:    { maxWidth: 1100, margin: '0 auto' },
    heading: { fontSize: 26, fontWeight: 700, color: '#fff', marginBottom: 4 },
    sub:     { fontSize: 13, color: '#6b6b8a', marginBottom: 32 },
    row:     { display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 },
    panel:   { background: '#111120', border: '1px solid #1e1e35', borderRadius: 12, padding: '20px 24px' },
    pTitle:  { fontSize: 13, fontWeight: 600, color: '#a0a0c0', marginBottom: 16 },
    pill:    (active) => ({
      padding: '5px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600, cursor: 'pointer', border: 'none',
      background: active ? '#6366f1' : '#1a1a2e', color: active ? '#fff' : '#8080a0',
    }),
  };

  if (loading) return (
    <div style={{ ...s.page, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ color: '#6b6b8a', fontSize: 14 }}>Loading analytics…</div>
    </div>
  );

  if (error) return (
    <div style={{ ...s.page, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ color: '#f87171', fontSize: 14 }}>{error}</div>
    </div>
  );

  const total = (stats?.tbCount || 0) + (stats?.normalCount || 0) + (stats?.inconclusiveCount || 0);
  const tbPct  = total ? Math.round((stats.tbCount / total) * 100) : 0;
  const norPct = total ? Math.round((stats.normalCount / total) * 100) : 0;

  return (
    <div style={s.page}>
      <div style={s.wrap}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 32 }}>
          <div>
            <h1 style={s.heading}>Analytics Dashboard</h1>
            <p style={{ ...s.sub, marginBottom: 0 }}>Model performance & scan trends</p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {['7d', '30d', '90d', 'all'].map(r => (
              <button key={r} style={s.pill(range === r)} onClick={() => setRange(r)}>{r}</button>
            ))}
          </div>
        </div>

        {/* KPI row */}
        <div style={s.row}>
          <StatCard label="Total scans"       value={total}                      sub={`in last ${range}`} />
          <StatCard label="TB detected"       value={`${stats?.tbCount ?? 0}`}   sub={`${tbPct}% of scans`}   accent="#ef4444" />
          <StatCard label="Normal"            value={`${stats?.normalCount ?? 0}`} sub={`${norPct}% of scans`} accent="#22c55e" />
          <StatCard label="Avg confidence"    value={stats?.avgConfidence != null ? `${Math.round(stats.avgConfidence * 100)}%` : '—'} sub="across all scans" accent="#6366f1" />
          <StatCard label="Avg process time"  value={stats?.avgProcessingMs != null ? `${Math.round(stats.avgProcessingMs)} ms` : '—'} sub="ai-service latency" />
        </div>

        {/* Row 2: donut + confidence buckets */}
        <div style={{ ...s.row }}>
          {/* Donut */}
          <div style={{ ...s.panel, flex: '0 0 auto' }}>
            <div style={s.pTitle}>Result distribution</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
              <DonutChart tbCount={stats?.tbCount || 0} normalCount={stats?.normalCount || 0} total={total} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {[
                  { color: '#ef4444', label: 'TB Detected',  pct: tbPct,  n: stats?.tbCount },
                  { color: '#22c55e', label: 'Normal',       pct: norPct, n: stats?.normalCount },
                  { color: '#f59e0b', label: 'Inconclusive', pct: 100 - tbPct - norPct, n: stats?.inconclusiveCount },
                ].map(({ color, label, pct, n }) => (
                  <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }} />
                    <span style={{ fontSize: 12, color: '#a0a0c0' }}>{label}</span>
                    <span style={{ fontSize: 12, color: '#fff', marginLeft: 'auto', paddingLeft: 12 }}>{n ?? 0} <span style={{ color: '#6b6b8a' }}>({Math.max(0, pct)}%)</span></span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Confidence distribution */}
          <div style={{ ...s.panel, flex: 1, minWidth: 220 }}>
            <div style={s.pTitle}>Confidence distribution</div>
            <ConfidenceBuckets buckets={stats?.confidenceBuckets} />
            <div style={{ marginTop: 12, fontSize: 11, color: '#6b6b8a' }}>
              Higher confidence = model is more certain of its prediction
            </div>
          </div>
        </div>

        {/* Row 3: scans over time bar chart */}
        <div style={{ ...s.panel, marginBottom: 24 }}>
          <div style={s.pTitle}>Scans over time</div>
          <BarChart data={stats?.timeline || []} />
        </div>

        {/* Row 4: recent activity table */}
        <div style={s.panel}>
          <div style={s.pTitle}>Recent scans (last 5)</div>
          {stats?.recentScans?.length ? (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Time', 'File', 'Result', 'Confidence'].map(h => (
                    <th key={h} style={{ textAlign: 'left', fontSize: 11, color: '#6b6b8a', padding: '0 0 10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stats.recentScans.map((r, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: 12, color: '#6b6b8a', padding: '8px 0', borderTop: '1px solid #1e1e35' }}>{new Date(r.createdAt).toLocaleString()}</td>
                    <td style={{ fontSize: 12, color: '#a0a0c0', padding: '8px 0', borderTop: '1px solid #1e1e35', fontFamily: 'monospace' }}>{r.filename || '—'}</td>
                    <td style={{ padding: '8px 0', borderTop: '1px solid #1e1e35' }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: r.result === 'TB_DETECTED' ? '#ef4444' : r.result === 'NORMAL' ? '#22c55e' : '#f59e0b' }}>
                        {(r.result || 'INCONCLUSIVE').replace('_', ' ')}
                      </span>
                    </td>
                    <td style={{ fontSize: 12, color: '#a0a0c0', padding: '8px 0', borderTop: '1px solid #1e1e35' }}>
                      {r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={{ color: '#4a4a6a', fontSize: 13 }}>No scans yet</div>
          )}
        </div>
      </div>
    </div>
  );
}