const { getAuditLogs } = require('../services/auditService');

/**
 * GET /api/audit
 * Returns paginated audit logs. Admin only.
 */
async function listAuditLogs(req, res, next) {
  try {
    const { page = 1, limit = 20, action } = req.query;
    const result = await getAuditLogs({
      page: parseInt(page, 10),
      limit: Math.min(parseInt(limit, 10), 100),
      action,
    });
    res.status(200).json({ success: true, ...result });
  } catch (err) {
    next(err);
  }
}

module.exports = { listAuditLogs };
