const AuditLog = require('../models/AuditLog');
const logger = require('../config/logger');

/**
 * Creates an audit log entry.
 * Silently handles errors to avoid blocking the main request flow.
 */
async function createAuditLog({ userId, username, action, endpoint, method, statusCode, ipAddress, userAgent, metadata, duration }) {
  try {
    await AuditLog.create({
      userId,
      username,
      action,
      endpoint,
      method,
      statusCode,
      ipAddress: ipAddress || 'unknown',
      userAgent: userAgent || 'unknown',
      metadata: metadata || {},
      duration,
    });
  } catch (err) {
    const errMsg = err?.message || (typeof err === 'object' ? JSON.stringify(err) : String(err));
    logger.error(`Failed to write audit log: ${errMsg}`);
  }
}

/**
 * Retrieves audit logs with pagination.
 */
async function getAuditLogs({ page = 1, limit = 20, userId, action } = {}) {
  const filter = {};
  if (userId) filter.userId = userId;
  if (action) filter.action = action;

  const skip = (page - 1) * limit;
  const [logs, total] = await Promise.all([
    AuditLog.find(filter).sort({ createdAt: -1 }).skip(skip).limit(limit).populate('userId', 'username email'),
    AuditLog.countDocuments(filter),
  ]);

  return { logs, total, page, pages: Math.ceil(total / limit) };
}

module.exports = { createAuditLog, getAuditLogs };
