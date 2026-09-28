const express  = require('express');
const router   = express.Router();
const { authenticate } = require('../middlewares/auth');
const {
  getHistory,
  getRecordById,
  getReport,
  updateReport,
} = require('../controllers/historyController');

// GET /api/history — paginated scan history for the logged-in user
router.get('/', authenticate, getHistory);

// GET /api/history/:id — fetch a single scan record
router.get('/:id', authenticate, getRecordById);

// PUT /api/history/:id/report — update/edit/approve radiology report
router.put('/:id/report', authenticate, updateReport);

// GET /api/history/:id/report/pdf — stream a PDF diagnostic report
router.get('/:id/report/pdf', authenticate, getReport);
router.get('/:id/report', authenticate, getReport);

module.exports = router;