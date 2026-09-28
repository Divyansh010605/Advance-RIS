const mongoose = require('mongoose');

const auditLogSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: false, // allow anonymous audit entries
    },
    username: {
      type: String,
      default: 'anonymous',
    },
    action: {
      type: String,
      required: true,
      enum: [
        'LOGIN',
        'LOGOUT',
        'SIGNUP',
        'IMAGE_UPLOAD',
        'PREDICTION_REQUEST',
        'PREDICTION_RESPONSE',
        'AUTH_FAILURE',
        'UNAUTHORIZED_ACCESS',
        'MULTI_DISEASE_ANALYSIS',
        'REPORT_EDIT',
        'PDF_GENERATE',
        'ANALYSIS_SUCCESS',
        'ANALYSIS_FAILURE',
      ],
    },
    endpoint: {
      type: String,
      required: true,
    },
    method: {
      type: String,
      enum: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    },
    statusCode: {
      type: Number,
    },
    ipAddress: {
      type: String,
      default: 'unknown',
    },
    userAgent: {
      type: String,
      default: 'unknown',
    },
    metadata: {
      type: mongoose.Schema.Types.Mixed,
      default: {},
    },
    duration: {
      type: Number, // milliseconds
    },
  },
  {
    timestamps: true,
    capped: { size: 50 * 1024 * 1024, max: 10000 }, // 50 MB capped collection
  }
);

auditLogSchema.index({ userId: 1 });
auditLogSchema.index({ action: 1 });
auditLogSchema.index({ createdAt: -1 });

module.exports = mongoose.model('AuditLog', auditLogSchema);
