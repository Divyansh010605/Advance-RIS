import { useState, useCallback } from 'react';
import { uploadImage } from '../services/uploadService';

/**
 * Hook that manages the full prediction lifecycle.
 * Returns: { predict, loading, result, error, reset }
 */
export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const predict = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await uploadImage(file);
      setResult(data);
      return data;
    } catch (err) {
      const msg = err.response?.data?.error || err.message || 'Prediction failed.';
      setError(msg);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { predict, loading, result, error, reset };
}
