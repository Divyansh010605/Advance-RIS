const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { createAuditLog } = require('../services/auditService');
const logger = require('../config/logger');

const JWT_SECRET = process.env.JWT_SECRET || 'supersecretjwtkey123';

/**
 * Generates a signed JWT for a given user ID.
 */
function signToken(userId) {
  return jwt.sign(
    { id: userId },
    JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );
}

/**
 * POST /api/auth/signup
 */
async function signup(req, res, next) {
  try {
    const { username, email, password } = req.body;

    if (!username || !email || !password) {
      return res.status(400).json({ success: false, error: 'Username, email, and password are required.' });
    }

    const cleanUsername = username.trim();
    const cleanEmail = email.trim().toLowerCase();

    if (cleanUsername.length < 3 || cleanUsername.length > 30) {
      return res.status(400).json({ success: false, error: 'Username must be between 3 and 30 characters.' });
    }

    if (password.length < 8) {
      return res.status(400).json({ success: false, error: 'Password must be at least 8 characters long.' });
    }

    // Check if user already exists
    const existingUser = await User.findOne({
      $or: [{ email: cleanEmail }, { username: cleanUsername }],
    });

    if (existingUser) {
      if (existingUser.email === cleanEmail) {
        return res.status(409).json({ success: false, error: 'An account with this email address already exists.' });
      }
      if (existingUser.username === cleanUsername) {
        return res.status(409).json({ success: false, error: 'Username is already taken. Please choose another.' });
      }
    }

    const user = await User.create({
      username: cleanUsername,
      email: cleanEmail,
      password,
    });
    const token = signToken(user._id);

    await createAuditLog({
      userId: user._id,
      username: user.username,
      action: 'SIGNUP',
      endpoint: '/api/auth/signup',
      method: 'POST',
      statusCode: 201,
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    logger.info(`New user registered: ${cleanEmail} (${cleanUsername})`);

    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        role: user.role,
      },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * POST /api/auth/login
 * Accepts email OR username (in email, username, or loginInput field).
 */
async function login(req, res, next) {
  try {
    const rawIdentifier = req.body.email || req.body.username || req.body.identifier || req.body.loginInput;
    const password = req.body.password;

    if (!rawIdentifier || !password) {
      return res.status(400).json({ success: false, error: 'Username/Email and password are required.' });
    }

    const identifier = rawIdentifier.trim();

    // Look up user by either email or username
    const user = await User.findOne({
      $or: [
        { email: identifier.toLowerCase() },
        { username: identifier },
      ],
    }).select('+password');

    if (!user || !(await user.comparePassword(password))) {
      await createAuditLog({
        username: identifier,
        action: 'AUTH_FAILURE',
        endpoint: '/api/auth/login',
        method: 'POST',
        statusCode: 401,
        ipAddress: req.ip,
        userAgent: req.headers['user-agent'],
      });
      return res.status(401).json({ success: false, error: 'Invalid username/email or password.' });
    }

    if (!user.isActive) {
      return res.status(403).json({ success: false, error: 'Account has been deactivated.' });
    }

    user.lastLogin = new Date();
    await user.save({ validateBeforeSave: false });

    const token = signToken(user._id);

    await createAuditLog({
      userId: user._id,
      username: user.username,
      action: 'LOGIN',
      endpoint: '/api/auth/login',
      method: 'POST',
      statusCode: 200,
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    res.status(200).json({
      success: true,
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        role: user.role,
      },
    });
  } catch (err) {
    next(err);
  }
}

/**
 * GET /api/auth/me
 */
async function getMe(req, res) {
  res.status(200).json({
    success: true,
    user: {
      id: req.user._id,
      username: req.user.username,
      email: req.user.email,
      role: req.user.role,
    },
  });
}

module.exports = { signup, login, getMe };
