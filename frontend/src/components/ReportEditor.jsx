import React, { useState } from 'react';
import api from '../services/api';

export default function ReportEditor({ scanId, initialReport = {}, initialStatus = 'DRAFT' }) {
  const [findings, setFindings] = useState(
    Array.isArray(initialReport.findings) ? initialReport.findings.join('\n') : ''
  );
  const [impression, setImpression] = useState(
    Array.isArray(initialReport.impression) ? initialReport.impression.join('\n') : ''
  );
  const [status, setStatus] = useState(initialStatus);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState(null);

  const handleSave = async (approve = false) => {
    if (!scanId) return;
    setSaving(true);
    setMsg(null);

    const findingsArray = findings.split('\n').filter((line) => line.trim().length > 0);
    const impressionArray = impression.split('\n').filter((line) => line.trim().length > 0);

    try {
      const res = await api.put(`/history/${scanId}/report`, {
        findings: findingsArray,
        impression: impressionArray,
        approve: approve,
      });

      setStatus(res.data.record?.reportStatus || (approve ? 'APPROVED' : 'EDITED'));
      setMsg({
        type: 'success',
        text: approve ? 'Report approved by clinician!' : 'Report draft saved successfully.',
      });
    } catch (err) {
      setMsg({ type: 'error', text: 'Failed to update report. Please try again.' });
    } finally {
      setSaving(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!scanId) return;
    try {
      const response = await api.get(`/history/${scanId}/report/pdf`, {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Lung_Disease_Report_${scanId}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      alert('Failed to download PDF report.');
    }
  };

  return (
    <div className="card mb-4" id="report-editor">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>
            Automated Radiology Report (Human-in-the-Loop)
          </h2>
          <span style={{ fontSize: '0.8rem', color: '#9ca3af' }}>
            Status: <strong style={{ color: status === 'APPROVED' ? '#22c55e' : '#f59e0b' }}>{status}</strong>
          </span>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className="btn btn-ghost"
            style={{ padding: '6px 12px', fontSize: '0.85rem' }}
            onClick={() => handleSave(false)}
            disabled={saving}
          >
            {saving ? 'Saving...' : 'Save Draft'}
          </button>
          <button
            className="btn btn-primary"
            style={{ padding: '6px 14px', fontSize: '0.85rem', background: '#22c55e', borderColor: '#22c55e' }}
            onClick={() => handleSave(true)}
            disabled={saving}
          >
            Approve Report
          </button>
          <button
            className="btn btn-primary"
            style={{ padding: '6px 14px', fontSize: '0.85rem' }}
            onClick={handleDownloadPDF}
          >
            Generate PDF
          </button>
        </div>
      </div>

      {msg && (
        <div
          style={{
            padding: '10px 14px',
            borderRadius: '6px',
            marginBottom: '16px',
            fontSize: '0.85rem',
            background: msg.type === 'success' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
            color: msg.type === 'success' ? '#4ade80' : '#f87171',
            border: `1px solid ${msg.type === 'success' ? '#22c55e' : '#ef4444'}`,
          }}
        >
          {msg.text}
        </div>
      )}

      {/* Findings Section */}
      <div style={{ marginBottom: 16 }}>
        <label style={{ display: 'block', fontWeight: 600, fontSize: '0.9rem', marginBottom: 6 }}>
          Findings [Editable]
        </label>
        <textarea
          rows={5}
          style={{
            width: '100%',
            background: '#18181b',
            color: '#f4f4f5',
            border: '1px solid #27272a',
            borderRadius: '6px',
            padding: '10px',
            fontFamily: 'inherit',
            fontSize: '0.9rem',
          }}
          value={findings}
          onChange={(e) => setFindings(e.target.value)}
        />
      </div>

      {/* Impression Section */}
      <div style={{ marginBottom: 16 }}>
        <label style={{ display: 'block', fontWeight: 600, fontSize: '0.9rem', marginBottom: 6 }}>
          Impression [Editable]
        </label>
        <textarea
          rows={3}
          style={{
            width: '100%',
            background: '#18181b',
            color: '#f4f4f5',
            border: '1px solid #27272a',
            borderRadius: '6px',
            padding: '10px',
            fontFamily: 'inherit',
            fontSize: '0.9rem',
          }}
          value={impression}
          onChange={(e) => setImpression(e.target.value)}
        />
      </div>

      {/* Warning banner */}
      <div style={{ fontSize: '0.75rem', color: '#a1a1aa', borderTop: '1px solid #27272a', paddingTop: '12px' }}>
        <strong>DISCLAIMER:</strong> This radiology report is an AI-generated draft provided for research and educational purposes. Clinical review and authorization by a certified radiologist is required before diagnostic decision-making.
      </div>
    </div>
  );
}
