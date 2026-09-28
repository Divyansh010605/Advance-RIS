const mongoose = require('mongoose');

/**
 * ScanRecord — Multi-Disease Chest X-Ray diagnostic record schema.
 */
const scanRecordSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },
    patientId: { type: String, default: null },
    filename:  { type: String, required: true },
    result: {
      type: String,
      enum: ['PATHOLOGY_DETECTED', 'NORMAL', 'TB_DETECTED', 'INCONCLUSIVE'],
      default: 'NORMAL',
    },
    confidence: { type: Number, min: 0, max: 1, required: true },
    
    // Multi-disease predictions array
    predictions: [
      {
        disease: { type: String, required: true },
        probability: { type: Number, required: true },
        threshold: { type: Number, required: true },
        detected: { type: Boolean, required: true },
      }
    ],

    originalImageBase64: { type: String, default: null },
    
    // Per-disease heatmaps
    heatmaps: [
      {
        disease: { type: String },
        heatmap: { type: String }, // base64
        overlay: { type: String }, // base64
      }
    ],

    // Backward-compatible single heatmap properties
    gradcamBase64: { type: String, default: null },
    overlayBase64: { type: String, default: null },

    // AI Generated Report Draft
    aiReport: {
      status: { type: String, default: 'draft' },
      findings: [{ type: String }],
      impression: [{ type: String }],
      detectedConditions: [
        {
          name: String,
          confidence: Number
        }
      ],
      modelConfidence: {
        overall: Number,
        perDisease: mongoose.Schema.Types.Mixed
      },
      unsupportedFindings: [{ type: String }],
      generatedAt: { type: Date, default: Date.now },
    },

    // Human Approved / Edited Report
    finalReport: {
      findings: [{ type: String }],
      impression: [{ type: String }],
      approvedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
      approvedAt: { type: Date },
    },

    reportStatus: {
      type: String,
      enum: ['DRAFT', 'EDITED', 'APPROVED'],
      default: 'DRAFT',
    },

    modelVersion: { type: String, default: 'DenseNet-121 Multi-Label v2.0' },
    processingMs: { type: Number, default: null },
    rawResponse:  { type: mongoose.Schema.Types.Mixed, default: null },
  },
  { timestamps: true }
);

scanRecordSchema.index({ userId: 1, createdAt: -1 });

module.exports = mongoose.model('ScanRecord', scanRecordSchema);