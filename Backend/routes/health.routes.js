const express = require('express');
const mongoose = require('mongoose');
const { getBreakerStats } = require('../services/aiService');

const router = express.Router();

// GET /health
router.get('/', async (req, res) => {
  const dbState = mongoose.connection.readyState;
  const dbStatus = dbState === 1 ? 'connected' : 'disconnected';
  const breakerStats = getBreakerStats();

  const status = dbState === 1 ? 'healthy' : 'degraded';

  res.status(status === 'healthy' ? 200 : 503).json({
    status,
    timestamp: new Date().toISOString(),
    services: {
      database: dbStatus,
      circuitBreaker: {
        state: breakerStats?.state || 'unknown',
        failures: breakerStats?.failures || 0,
        successes: breakerStats?.successes || 0,
      },
    },
  });
});

module.exports = router;
