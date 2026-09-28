const express = require('express');
const { listAuditLogs } = require('../controllers/auditController');
const { authenticate, authorize } = require('../middlewares/auth');

const router = express.Router();

// GET /api/audit  (admin only)
router.get('/', authenticate, authorize('admin'), listAuditLogs);

module.exports = router;
