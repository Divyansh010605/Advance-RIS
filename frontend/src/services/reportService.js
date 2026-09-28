import api from './api';

/**
 * Download a PDF report for a given scan record ID.
 * Uses the JWT token via the existing Axios interceptor in api.js.
 */
export async function downloadReport(recordId, filename = 'tb-report.pdf') {
  const response = await api.get(`/history/${recordId}/report`, {
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
}

/**
 * Fetch analytics stats for a given time range.
 * range: '7d' | '30d' | '90d' | 'all'
 */
export async function fetchAnalytics(range = '30d') {
  const res = await api.get(`/analytics?range=${range}`);
  return res.data;
}