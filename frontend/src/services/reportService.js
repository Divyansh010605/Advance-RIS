import api from './api';

/**
 * Download a PDF report for a given scan record ID.
 * Uses the JWT token via the existing Axios interceptor in api.js.
 */
const inFlight = new Set();

/**
 * Exactly one PDF per request: repeated clicks while a download for the same
 * scan is still running (or just finished) are ignored.
 */
export async function downloadReport(recordId, filename = `Lung_Disease_Report_${recordId}.pdf`) {
  if (!recordId || inFlight.has(recordId)) return;
  inFlight.add(recordId);
  try {
    const response = await api.get(`/history/${recordId}/report/pdf`, {
      responseType: 'blob',
    });

    const url  = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
    const link = document.createElement('a');
    link.href  = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
    window.URL.revokeObjectURL(url);
  } finally {
    // short cool-down so a double-click can't trigger a second file
    setTimeout(() => inFlight.delete(recordId), 1500);
  }
}

/**
 * Fetch analytics stats for a given time range.
 * range: '7d' | '30d' | '90d' | 'all'
 */
export async function fetchAnalytics(range = '30d') {
  const res = await api.get(`/analytics?range=${range}`);
  return res.data;
}