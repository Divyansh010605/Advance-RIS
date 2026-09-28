import api from './api';

/**
 * Uploads a File object to the backend and returns prediction results.
 * @param {File} file
 * @returns {Promise<{prediction, heatmap, overlay, label, confidence, processingTimeMs}>}
 */
export async function uploadImage(file) {
  const formData = new FormData();
  formData.append('image', file);

  const res = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000, // 2 min for large images
  });

  return res.data.data;
}
