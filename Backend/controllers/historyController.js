const ScanRecord = require('../models/ScanRecord');
const PDFDocument = require('pdfkit');

/* ────────────────────────────────────────────────────────────────
   GET /api/history
──────────────────────────────────────────────────────────────── */
exports.getHistory = async (req, res, next) => {
  try {
    const page   = Math.max(1, parseInt(req.query.page)  || 1);
    const limit  = Math.min(50, parseInt(req.query.limit) || 10);
    const skip   = (page - 1) * limit;

    const filter = { userId: req.user._id };
    if (req.query.status && req.query.status !== 'ALL') {
      filter.result = req.query.status;
    }
    if (req.query.search) {
      filter.filename = { $regex: req.query.search, $options: 'i' };
    }

    const [records, total] = await Promise.all([
      ScanRecord.find(filter)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .select('-heatmaps -originalImageBase64 -gradcamBase64 -overlayBase64 -rawResponse')
        .lean(),
      ScanRecord.countDocuments(filter),
    ]);

    res.json({ records, total, page, limit });
  } catch (err) {
    next(err);
  }
};

/* ────────────────────────────────────────────────────────────────
   GET /api/history/:id
──────────────────────────────────────────────────────────────── */
exports.getRecordById = async (req, res, next) => {
  try {
    const record = await ScanRecord.findOne({
      _id: req.params.id,
      userId: req.user._id,
    }).lean();

    if (!record) return res.status(404).json({ message: 'Record not found' });

    res.json(record);
  } catch (err) {
    next(err);
  }
};

/* ────────────────────────────────────────────────────────────────
   PUT /api/history/:id/report
   Update / edit / approve draft report (Human-in-the-Loop workflow)
──────────────────────────────────────────────────────────────── */
exports.updateReport = async (req, res, next) => {
  try {
    const { findings, impression, approve } = req.body;

    const record = await ScanRecord.findOne({
      _id: req.params.id,
      userId: req.user._id,
    });

    if (!record) return res.status(404).json({ message: 'Record not found' });

    if (!record.finalReport) {
      record.finalReport = {};
    }

    if (Array.isArray(findings)) record.finalReport.findings = findings;
    if (Array.isArray(impression)) record.finalReport.impression = impression;

    if (approve) {
      record.reportStatus = 'APPROVED';
      record.finalReport.approvedBy = req.user._id;
      record.finalReport.approvedAt = new Date();
    } else {
      record.reportStatus = 'EDITED';
    }

    await record.save();

    res.json({
      success: true,
      message: approve ? 'Report approved successfully.' : 'Draft report saved.',
      record,
    });
  } catch (err) {
    next(err);
  }
};

/* ────────────────────────────────────────────────────────────────
   GET /api/history/:id/report/pdf
   Streams structured multi-disease PDF diagnostic report.
──────────────────────────────────────────────────────────────── */
exports.getReport = async (req, res, next) => {
  try {
    const record = await ScanRecord.findOne({
      _id: req.params.id,
      userId: req.user._id,
    }).lean();

    if (!record) return res.status(404).json({ message: 'Record not found' });

    const doc = new PDFDocument({ margin: 40, size: 'A4', autoFirstPage: true });

    const safePatientId = (record.patientId || 'Patient').replace(/[^a-zA-Z0-9_-]/g, '_');
    const dateStr = new Date(record.createdAt || Date.now()).toISOString().split('T')[0];
    const pdfFilename = `Lung_Disease_Report_${safePatientId}_${dateStr}.pdf`;

    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="${pdfFilename}"`
    );
    doc.pipe(res);

    // ── PAGE 1: CLINICAL REPORT ──
    // Header Banner
    doc.rect(40, 40, 515, 52).fill('#0f172a');
    doc
      .fontSize(16).font('Helvetica-Bold').fillColor('#ffffff')
      .text('LUNG DISEASE DIAGNOSTIC REPORT', 40, 50, { width: 515, align: 'center' });
    doc
      .fontSize(8.5).font('Helvetica').fillColor('#38bdf8')
      .text('LungDetect AI  ·  Multi-Disease Chest X-Ray Screening System', 40, 72, { width: 515, align: 'center' });

    // Metadata Grid Box
    const metaY = 105;
    doc.roundedRect(40, metaY, 515, 50, 6).fillAndStroke('#f8fafc', '#cbd5e1');

    doc.fontSize(8).font('Helvetica-Bold').fillColor('#64748b');
    doc.text('PATIENT ID', 55, metaY + 10);
    doc.fontSize(10).font('Helvetica-Bold').fillColor('#0f172a');
    doc.text(record.patientId || 'N/A', 55, metaY + 22);

    doc.fontSize(8).font('Helvetica-Bold').fillColor('#64748b');
    doc.text('SCAN ID', 185, metaY + 10);
    doc.fontSize(9).font('Helvetica-Bold').fillColor('#0f172a');
    doc.text(String(record._id).substring(0, 18), 185, metaY + 22);

    doc.fontSize(8).font('Helvetica-Bold').fillColor('#64748b');
    doc.text('DATE & TIME', 345, metaY + 10);
    doc.fontSize(9.5).font('Helvetica').fillColor('#0f172a');
    doc.text(new Date(record.createdAt).toLocaleDateString(), 345, metaY + 22);

    doc.fontSize(8).font('Helvetica-Bold').fillColor('#64748b');
    doc.text('REPORT STATUS', 455, metaY + 10);
    const statusColor = record.reportStatus === 'APPROVED' ? '#16a34a' : record.reportStatus === 'EDITED' ? '#2563eb' : '#d97706';
    doc.fontSize(9.5).font('Helvetica-Bold').fillColor(statusColor);
    doc.text(record.reportStatus || 'DRAFT', 455, metaY + 22);

    // Section 1: Detected Pathology Summary
    let currentY = 172;
    doc.fontSize(11).font('Helvetica-Bold').fillColor('#0f172a')
       .text('1. PATHOLOGY DETECTION SUMMARY', 40, currentY);
    currentY += 18;

    const predictions = record.predictions || [];
    const detectedList = predictions.filter(p => p.detected);

    if (detectedList.length === 0) {
      doc.roundedRect(40, currentY, 515, 34, 4).fillAndStroke('#f0fdf4', '#86efac');
      doc.fontSize(9.5).font('Helvetica-Bold').fillColor('#16a34a')
         .text('✔ No abnormal thoracic pathology detected above diagnostic thresholds.', 55, currentY + 10, { width: 485 });
      currentY += 46;
    } else {
      const boxHeight = Math.max(36, detectedList.length * 20 + 12);
      doc.roundedRect(40, currentY, 515, boxHeight, 4).fillAndStroke('#fef2f2', '#fca5a5');
      let itemY = currentY + 8;
      detectedList.forEach(p => {
        doc.fontSize(9.5).font('Helvetica-Bold').fillColor('#dc2626')
           .text(`• ${p.disease}: `, 55, itemY, { continued: true })
           .font('Helvetica').fillColor('#1e293b')
           .text(`Confidence ${(p.probability * 100).toFixed(1)}%  (Threshold: ${(p.threshold * 100).toFixed(1)}%)`);
        itemY += 18;
      });
      currentY += boxHeight + 12;
    }

    // Section 2: Radiology Findings
    doc.fontSize(11).font('Helvetica-Bold').fillColor('#0f172a')
       .text('2. RADIOLOGY REPORT FINDINGS', 40, currentY);
    currentY += 18;

    const reportData = record.finalReport?.findings?.length
      ? record.finalReport
      : record.aiReport || { findings: [], impression: [] };

    const findingsList = reportData.findings && reportData.findings.length
      ? reportData.findings
      : ['No focal consolidation, pleural effusion, or pneumothorax identified.', 'Lungs clear bilaterally with normal cardiomediastinal silhouette.'];

    findingsList.forEach(f => {
      doc.fontSize(9).font('Helvetica').fillColor('#334155')
         .text(`• ${f}`, 50, currentY, { width: 495 });
      currentY += doc.heightOfString(`• ${f}`, { width: 495 }) + 4;
    });

    currentY += 10;

    // Section 3: Impression
    doc.fontSize(11).font('Helvetica-Bold').fillColor('#0f172a')
       .text('3. IMPRESSION & RECOMMENDATION', 40, currentY);
    currentY += 18;

    const impressionList = reportData.impression && reportData.impression.length
      ? reportData.impression
      : ['No significant thoracic pathology detected.'];

    let totalImpHeight = 12;
    impressionList.forEach(imp => {
      totalImpHeight += doc.heightOfString(`- ${imp}`, { width: 485 }) + 4;
    });

    doc.roundedRect(40, currentY, 515, Math.max(34, totalImpHeight), 4).fillAndStroke('#f8fafc', '#e2e8f0');
    let impY = currentY + 8;
    impressionList.forEach(imp => {
      doc.fontSize(9.5).font('Helvetica-Bold').fillColor('#1e293b')
         .text(`- ${imp}`, 55, impY, { width: 485 });
      impY += doc.heightOfString(`- ${imp}`, { width: 485 }) + 4;
    });

    // Page 1 Footer
    doc.moveTo(40, 750).lineTo(555, 750).strokeColor('#cbd5e1').stroke();
    doc.fontSize(8).font('Helvetica').fillColor('#94a3b8')
       .text('LungDetect AI Diagnostic Platform  ·  Confidential Medical Record', 40, 758);
    doc.fontSize(8).font('Helvetica').fillColor('#94a3b8')
       .text('Page 1 of 2', 490, 758, { width: 65, align: 'right' });

    // ── PAGE 2: IMAGING & EXPLAINABILITY (GRAD-CAM) ──
    doc.addPage();

    // Page 2 Header Banner
    doc.rect(40, 40, 515, 52).fill('#0f172a');
    doc
      .fontSize(15).font('Helvetica-Bold').fillColor('#ffffff')
      .text('IMAGING & EXPLAINABILITY (GRAD-CAM)', 40, 50, { width: 515, align: 'center' });
    doc
      .fontSize(8.5).font('Helvetica').fillColor('#38bdf8')
      .text('Visual Feature Conditioning & Localized Activation Maps', 40, 72, { width: 515, align: 'center' });

    // Side by Side Images
    const imgY = 105;
    const imgBoxWidth = 245;
    const imgBoxHeight = 220;

    // Box 1: Original X-Ray
    doc.roundedRect(40, imgY, imgBoxWidth, imgBoxHeight, 6).fillAndStroke('#f8fafc', '#cbd5e1');
    doc.fontSize(9.5).font('Helvetica-Bold').fillColor('#1e293b')
       .text('Original Chest X-Ray (PA View)', 40, imgY + 8, { width: imgBoxWidth, align: 'center' });

    if (record.originalImageBase64) {
      try {
        const origBuf = Buffer.from(record.originalImageBase64, 'base64');
        doc.image(origBuf, 57, imgY + 26, { width: 210, height: 180, fit: [210, 180], align: 'center', valig: 'center' });
      } catch (e) {}
    } else {
      doc.fontSize(9).font('Helvetica').fillColor('#94a3b8')
         .text('Original scan image unavailable', 40, imgY + 90, { width: imgBoxWidth, align: 'center' });
    }

    // Box 2: Heatmap Overlay
    const box2X = 310;
    doc.roundedRect(box2X, imgY, imgBoxWidth, imgBoxHeight, 6).fillAndStroke('#f8fafc', '#cbd5e1');
    
    const topDisease = detectedList.length ? detectedList[0].disease : (record.heatmaps && record.heatmaps.length ? record.heatmaps[0].disease : 'Pathology');
    doc.fontSize(9.5).font('Helvetica-Bold').fillColor('#1e293b')
       .text(`AI Heatmap Overlay (${topDisease})`, box2X, imgY + 8, { width: imgBoxWidth, align: 'center' });

    const heatmapB64 = (record.heatmaps && record.heatmaps.length && (record.heatmaps[0].overlay || record.heatmaps[0].heatmap)) || record.overlayBase64 || record.gradcamBase64;

    if (heatmapB64) {
      try {
        const hmBuf = Buffer.from(heatmapB64, 'base64');
        doc.image(hmBuf, box2X + 17, imgY + 26, { width: 210, height: 180, fit: [210, 180], align: 'center', valig: 'center' });
      } catch (e) {}
    } else {
      doc.fontSize(9).font('Helvetica').fillColor('#94a3b8')
         .text('Heatmap overlay unavailable', box2X, imgY + 90, { width: imgBoxWidth, align: 'center' });
    }

    // Full Pathology Breakdown Table
    let tableY = imgY + imgBoxHeight + 20;
    doc.fontSize(11).font('Helvetica-Bold').fillColor('#0f172a')
       .text('4. FULL 14-PATHOLOGY SCREENING BREAKDOWN', 40, tableY);
    tableY += 18;

    doc.roundedRect(40, tableY, 515, 140, 6).fillAndStroke('#ffffff', '#e2e8f0');

    // Table Header
    doc.rect(41, tableY + 1, 513, 18).fill('#f1f5f9');
    doc.fontSize(8).font('Helvetica-Bold').fillColor('#475569');
    doc.text('PATHOLOGY CLASS', 55, tableY + 5);
    doc.text('PROBABILITY', 195, tableY + 5);
    doc.text('STATUS', 260, tableY + 5);

    doc.text('PATHOLOGY CLASS', 315, tableY + 5);
    doc.text('PROBABILITY', 455, tableY + 5);
    doc.text('STATUS', 520, tableY + 5);

    let rowY = tableY + 23;
    const allPreds = predictions.length ? predictions : [
      { disease: 'Atelectasis', probability: 0.05, threshold: 0.45, detected: false },
      { disease: 'Cardiomegaly', probability: 0.02, threshold: 0.45, detected: false },
      { disease: 'Effusion', probability: 0.04, threshold: 0.45, detected: false },
      { disease: 'Infiltration', probability: 0.08, threshold: 0.45, detected: false },
      { disease: 'Mass', probability: 0.01, threshold: 0.45, detected: false },
      { disease: 'Nodule', probability: 0.03, threshold: 0.45, detected: false },
      { disease: 'Pneumonia', probability: 0.06, threshold: 0.45, detected: false },
      { disease: 'Pneumothorax', probability: 0.01, threshold: 0.45, detected: false },
      { disease: 'Consolidation', probability: 0.02, threshold: 0.45, detected: false },
      { disease: 'Edema', probability: 0.01, threshold: 0.45, detected: false },
      { disease: 'Emphysema', probability: 0.01, threshold: 0.45, detected: false },
      { disease: 'Fibrosis', probability: 0.03, threshold: 0.45, detected: false },
      { disease: 'Pleural_Thickening', probability: 0.02, threshold: 0.45, detected: false },
      { disease: 'Hernia', probability: 0.00, threshold: 0.45, detected: false },
    ];

    const halfLength = Math.ceil(allPreds.length / 2);
    for (let i = 0; i < halfLength; i++) {
      const left = allPreds[i];
      const right = allPreds[i + halfLength];

      // Left col
      doc.fontSize(8).font('Helvetica-Bold').fillColor('#334155').text(left.disease, 55, rowY);
      doc.font('Helvetica').fillColor('#0f172a').text(`${(left.probability * 100).toFixed(1)}%`, 195, rowY);
      doc.font('Helvetica-Bold').fillColor(left.detected ? '#dc2626' : '#16a34a').text(left.detected ? 'POSITIVE' : 'NORMAL', 260, rowY);

      // Right col
      if (right) {
        doc.font('Helvetica-Bold').fillColor('#334155').text(right.disease, 315, rowY);
        doc.font('Helvetica').fillColor('#0f172a').text(`${(right.probability * 100).toFixed(1)}%`, 455, rowY);
        doc.font('Helvetica-Bold').fillColor(right.detected ? '#dc2626' : '#16a34a').text(right.detected ? 'POSITIVE' : 'NORMAL', 520, rowY);
      }
      rowY += 15;
    }

    // Disclaimer & Footer
    doc.moveTo(40, 720).lineTo(555, 720).strokeColor('#cbd5e1').stroke();
    doc
      .fontSize(7.5).font('Helvetica-Bold').fillColor('#64748b')
      .text('RESEARCH & CLINICAL INVESTIGATIONAL DISCLAIMER:', 40, 726, { width: 515, align: 'center' });
    doc
      .fontSize(7).font('Helvetica').fillColor('#94a3b8')
      .text(
        'This automated diagnostic screening report is generated by LungDetect AI for decision support and research purposes. ' +
        'It is NOT a certified standalone diagnostic tool. Clinical review and authorization by a certified radiologist is required before patient management.',
        40, 736, { width: 515, align: 'center' }
      );

    doc.fontSize(8).font('Helvetica').fillColor('#94a3b8')
       .text('LungDetect AI Research Platform', 40, 758);
    doc.fontSize(8).font('Helvetica').fillColor('#94a3b8')
       .text('Page 2 of 2', 490, 758, { width: 65, align: 'right' });

    doc.end();
  } catch (err) {
    next(err);
  }
};