const express = require('express');
const { signup, login, getMe } = require('../controllers/authController');
const { authenticate } = require('../middlewares/auth');

const router = express.Router();

// POST /api/auth/signup
router.post('/signup', signup);

// POST /api/auth/login
router.post('/login', login);

// GET /api/auth/me  (protected)
router.get('/me', authenticate, getMe);

module.exports = router;
