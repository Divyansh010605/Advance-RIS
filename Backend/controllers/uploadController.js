const sharp = require('sharp');
const { analyzeXRay } = require('../services/aiService');
const { createAuditLog } = require('../services/auditService');
const logger = require('../config/logger');
const ScanRecord = require('../models/ScanRecord');

/**
 * POST /api/upload
 * Accepts a multipart/form-data image upload (or DICOM).
 * Strips EXIF metadata, sends to AI service, returns multi-disease analysis and draft report.
 */
async function uploadImage(req, res, next) {
  const start = Date.now();
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image file uploaded.' });
    }

    const { mimetype, buffer, size, originalname } = req.file;
    const allowedMimes = [
      'image/jpeg', 'image/png', 'image/webp', 'image/bmp',
      'application/dicom', 'application/octet-stream'
    ];

    if (!allowedMimes.includes(mimetype) && !originalname.toLowerCase().endsWith('.dcm')) {
      return res.status(415).json({
        success: false,
        error: `Unsupported format. Allowed: JPEG, PNG, WebP, BMP, DICOM`,
      });
    }

    if (size > 15 * 1024 * 1024) {
      return res.status(413).json({ success: false, error: 'File must be smaller than 15 MB.' });
    }

    logger.info(`Processing upload: ${originalname} (${size} bytes, type: ${mimetype})`);

    let processedBuffer;
    let mimeTypeForAI = 'image/jpeg';

    if (originalname.toLowerCase().endsWith('.dcm') || mimetype === 'application/dicom') {
      // Basic DICOM handling: pass buffer as base64 with application/dicom or convert
      processedBuffer = buffer;
      mimeTypeForAI = 'application/dicom';
    } else {
      // Strip EXIF metadata and convert to high quality JPEG via sharp
      processedBuffer = await sharp(buffer)
        .removeAlpha()
        .toFormat('jpeg', { quality: 90 })
        .withMetadata({ exif: {} })
        .toBuffer();
    }

    const imageBase64 = processedBuffer.toString('base64');

    // Call AI multi-disease screening service
    const aiResult = await analyzeXRay(imageBase64, mimeTypeForAI);

    const duration = Date.now() - start;

    const detectedConditions = aiResult.predictions.filter((p) => p.detected);
    const hasPathology = detectedConditions.length > 0;
    const resultLabel = hasPathology ? 'PATHOLOGY_DETECTED' : 'NORMAL';

    const maxProb = aiResult.predictions.length
      ? Math.max(...aiResult.predictions.map((p) => p.probability))
      : 0;

    const firstHeatmap = aiResult.heatmaps && aiResult.heatmaps.length ? aiResult.heatmaps[0] : {};

    const newRecord = await ScanRecord.create({
      userId: req.user?._id,
      patientId: req.body.patientId || `P-${Math.random().toString(36).substring(2, 8).toUpperCase()}`,
      filename: originalname || 'xray_scan.jpg',
      result: resultLabel,
      confidence: maxProb,
      predictions: aiResult.predictions,
      originalImageBase64: imageBase64,
      heatmaps: aiResult.heatmaps,
      gradcamBase64: firstHeatmap.heatmap || null,
      overlayBase64: firstHeatmap.overlay || null,
      aiReport: {
        status: aiResult.report?.status || 'draft',
        findings: aiResult.report?.findings || [],
        impression: aiResult.report?.impression || [],
        detectedConditions: aiResult.report?.detected_conditions || [],
        modelConfidence: aiResult.report?.model_confidence || { overall: maxProb, perDisease: {} },
        unsupportedFindings: aiResult.report?.unsupported_findings || [],
        generatedAt: new Date(),
      },
      reportStatus: 'DRAFT',
      modelVersion: aiResult.metadata?.model_version || 'DenseNet-121 Multi-Label v2.0',
      processingMs: duration,
      rawResponse: aiResult,
    });

    await createAuditLog({
      userId: req.user?._id,
      username: req.user?.username || 'anonymous',
      action: 'MULTI_DISEASE_ANALYSIS',
      endpoint: '/api/upload',
      method: 'POST',
      statusCode: 200,
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
      metadata: { detectedCount: detectedConditions.length, maxProb },
      duration,
    });

    res.status(200).json({
      success: true,
      data: {
        id: newRecord._id,
        patientId: newRecord.patientId,
        predictions: aiResult.predictions,
        heatmaps: aiResult.heatmaps,
        report: aiResult.report,
        result: resultLabel,
        confidence: maxProb,
        processingTimeMs: duration,
      },
    });
  } catch (err) {
    next(err);
  }
}

module.exports = { uploadImage };