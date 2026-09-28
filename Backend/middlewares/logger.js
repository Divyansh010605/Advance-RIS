const logger = require('../config/logger');

/**
 * Logs all incoming HTTP requests with method, path, IP, and duration.
 */
function requestLogger(req, res, next) {
  const start = Date.now();
  const requestId = require('uuid').v4();
  req.requestId = requestId;

  res.on('finish', () => {
    const duration = Date.now() - start;
    logger.info({
      requestId,
      method: req.method,
      path: req.originalUrl,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip || req.connection.remoteAddress,
      userAgent: req.headers['user-agent'],
    });
  });

  next();
}

module.exports = requestLogger;
