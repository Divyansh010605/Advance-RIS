const express  = require('express');
const router   = express.Router();
const { authenticate } = require('../middlewares/auth');
const { getAnalytics } = require('../controllers/analyticsController');

// GET /api/analytics?range=30d
router.get('/', authenticate, getAnalytics);

module.exports = router;