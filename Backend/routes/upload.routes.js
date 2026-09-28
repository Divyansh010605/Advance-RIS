const express = require('express');
const multer = require('multer');
const { uploadImage } = require('../controllers/uploadController');
const { authenticate } = require('../middlewares/auth');

const router = express.Router();

// Store file in memory for processing (no disk I/O)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
  fileFilter(req, file, cb) {
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file type.'));
    }
  },
});

// POST /api/upload  (protected)
router.post('/', authenticate, upload.single('image'), uploadImage);

module.exports = router;
