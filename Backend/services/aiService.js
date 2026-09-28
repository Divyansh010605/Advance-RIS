const axios = require('axios');
const CircuitBreaker = require('opossum');
const logger = require('../config/logger');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://ai-service:8000';
const TIMEOUT_MS = parseInt(process.env.AI_TIMEOUT_MS || '30000', 10);

/**
 * Core function calling AI service multi-disease endpoint /api/v1/analyze.
 */
async function callAIServiceRaw(imageBase64, mimeType = 'image/jpeg') {
  try {
    const response = await axios.post(
      `${AI_SERVICE_URL}/api/v1/analyze`,
      { image: imageBase64, mime_type: mimeType },
      {
        timeout: TIMEOUT_MS,
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
      }
    );
    return response.data;
  } catch (err) {
    logger.warn(`Multi-disease AI endpoint failed: ${err.message}. Trying legacy /predict endpoint...`);
    // Fallback to legacy /predict
    const fallbackResponse = await axios.post(
      `${AI_SERVICE_URL}/predict`,
      { image: imageBase64, mime_type: mimeType },
      {
        timeout: TIMEOUT_MS,
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
      }
    );
    const legacy = fallbackResponse.data;
    // Map legacy response to multi-disease format
    return {
      request_id: 'legacy-req',
      predictions: [
        {
          disease: 'Infiltration',
          probability: legacy.prediction,
          threshold: 0.45,
          detected: legacy.prediction > 0.45,
        },
      ],
      heatmaps: [
        {
          disease: 'Infiltration',
          heatmap: legacy.heatmap,
          overlay: legacy.overlay,
        },
      ],
      report: {
        status: 'draft',
        findings: [
          legacy.prediction > 0.45
            ? 'Focal pulmonary opacities and infiltrates detected.'
            : 'No active lung infiltrates or consolidations observed.',
        ],
        impression: [
          legacy.prediction > 0.45 ? 'Positive for lung disease pathology.' : 'Negative for significant lung disease pathology.',
        ],
        detected_conditions: legacy.prediction > 0.45 ? [{ name: 'Infiltration', confidence: legacy.prediction }] : [],
        model_confidence: { overall: legacy.prediction, per_disease: { Infiltration: legacy.prediction } },
        review_required: true,
        disclaimer: 'AI-generated draft for research and educational purposes. Human review required.',
      },
      metadata: { model_version: '1.0.0-legacy', processing_time_ms: 0 },
    };
  }
}

// Circuit Breaker setup
const cbOptions = {
  timeout: TIMEOUT_MS,
  errorThresholdPercentage: 50,
  resetTimeout: 30000,
  name: 'ai-service-breaker',
  rollingCountTimeout: 10000,
  rollingCountBuckets: 10,
};

const breaker = new CircuitBreaker(callAIServiceRaw, cbOptions);

breaker.on('open', () => logger.warn('⚡ Circuit breaker OPEN — AI service calls blocked'));
breaker.on('halfOpen', () => logger.info('⚡ Circuit breaker HALF-OPEN — Testing AI service'));
breaker.on('close', () => logger.info('✅ Circuit breaker CLOSED — AI service healthy'));
breaker.fallback(() => {
  const err = new Error('AI service is temporarily unavailable (circuit open).');
  err.code = 'EOPENBREAKER';
  throw err;
});

async function analyzeXRay(imageBase64, mimeType) {
  logger.info('Forwarding image to AI service for multi-disease analysis...');
  const result = await breaker.fire(imageBase64, mimeType);
  logger.info(`AI analysis complete. Request ID: ${result.request_id}`);
  return result;
}

// Backward compatibility alias
async function predictTB(imageBase64, mimeType) {
  const res = await analyzeXRay(imageBase64, mimeType);
  const topPred = res.predictions.find((p) => p.detected) || res.predictions[0];
  const firstHeatmap = res.heatmaps[0] || {};
  return {
    prediction: topPred ? topPred.probability : 0,
    heatmap: firstHeatmap.heatmap || '',
    overlay: firstHeatmap.overlay || '',
    raw: res,
  };
}

function getBreakerStats() {
  return breaker.stats;
}

module.exports = { analyzeXRay, predictTB, getBreakerStats };
