const ScanRecord = require('../models/ScanRecord');

/* ────────────────────────────────────────────────────────────────
   GET /api/analytics?range=30d
   Aggregated metrics for NIH ChestX-ray14 Multi-Disease Screening
   and Tri-Model Consensus RIS Platform.
──────────────────────────────────────────────────────────────── */
exports.getAnalytics = async (req, res, next) => {
  try {
    const range  = req.query.range || '30d';
    const userId = req.user._id;

    // Compute start date based on range filter
    const now = new Date();
    let startDate = null;
    if (range !== 'all') {
      const days = range === '7d' ? 7 : range === '90d' ? 90 : 30;
      startDate = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);
    }

    const dateFilter = startDate ? { createdAt: { $gte: startDate } } : {};
    const baseMatch  = { userId, ...dateFilter };

    /* ── 1. Primary Classification Result Counts ── */
    const resultAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      { $group: { _id: '$result', count: { $sum: 1 } } },
    ]);

    const counts = {
      PATHOLOGY_DETECTED: 0,
      NORMAL: 0,
      INCONCLUSIVE: 0,
      TB_DETECTED: 0,
    };
    resultAgg.forEach((r) => {
      if (r._id && counts[r._id] !== undefined) {
        counts[r._id] = r.count;
      }
    });

    const pathologyCount = (counts.PATHOLOGY_DETECTED || 0) + (counts.TB_DETECTED || 0);
    const normalCount = counts.NORMAL || 0;
    const inconclusiveCount = counts.INCONCLUSIVE || 0;
    const totalScans = pathologyCount + normalCount + inconclusiveCount;

    /* ── 2. Pathology Prevalence Breakdown across 14 NIH Diseases ── */
    const pathologyAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      { $unwind: '$predictions' },
      {
        $match: {
          'predictions.disease': { $ne: 'No_Finding' },
          $or: [
            { 'predictions.detected': true },
            { $expr: { $gte: [{ $size: { $ifNull: ['$predictions.models_agreeing', []] } }, 2] } },
          ],
        },
      },
      {
        $group: {
          _id: '$predictions.disease',
          count: { $sum: 1 },
          avgProb: { $avg: '$predictions.probability' },
        },
      },
      { $sort: { count: -1 } },
      { $limit: 14 },
    ]);

    const pathologyDistribution = pathologyAgg.map((p) => ({
      disease: p._id,
      count: p.count,
      avgConfidence: Math.round((p.avgProb || 0) * 100),
    }));

    /* ── 3. Tri-Model Agreement & Consensus Breakdown ── */
    const consensusAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      { $unwind: '$predictions' },
      { $match: { 'predictions.disease': { $ne: 'No_Finding' } } },
      {
        $project: {
          numAgreeing: { $size: { $ifNull: ['$predictions.models_agreeing', []] } },
        },
      },
      {
        $group: {
          _id: '$numAgreeing',
          count: { $sum: 1 },
        },
      },
    ]);

    const consensusBreakdown = {
      unanimous3: 0, // Swin + DenseNet + ResNet all 3 agree
      majority2: 0,  // Any 2 models agree
      single1: 0,    // Only 1 model flagged
      none0: 0,      // 0 models flagged
    };

    consensusAgg.forEach((c) => {
      if (c._id === 3) consensusBreakdown.unanimous3 = c.count;
      else if (c._id === 2) consensusBreakdown.majority2 = c.count;
      else if (c._id === 1) consensusBreakdown.single1 = c.count;
      else consensusBreakdown.none0 += c.count;
    });

    const highConsensusTotal = consensusBreakdown.unanimous3 + consensusBreakdown.majority2;
    const totalPathologyChecks = highConsensusTotal + consensusBreakdown.single1;
    const consensusRate = totalPathologyChecks > 0
      ? Math.round((highConsensusTotal / totalPathologyChecks) * 100)
      : 100;

    /* ── 4. Avg Confidence & Processing Latency ── */
    const avgAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      {
        $group: {
          _id: null,
          avgConfidence:   { $avg: '$confidence' },
          avgProcessingMs: { $avg: '$processingMs' },
        },
      },
    ]);
    const avgConfidence   = avgAgg[0]?.avgConfidence   != null ? avgAgg[0].avgConfidence : 0;
    const avgProcessingMs = avgAgg[0]?.avgProcessingMs != null ? avgAgg[0].avgProcessingMs : 0;

    /* ── 5. Confidence Buckets [0–20%, 20–40%, 40–60%, 60–80%, 80–100%] ── */
    const bucketAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      {
        $bucket: {
          groupBy: '$confidence',
          boundaries: [0, 0.2, 0.4, 0.6, 0.8, 1.01],
          default: 'other',
          output: { count: { $sum: 1 } },
        },
      },
    ]);
    const bucketMap = {};
    bucketAgg.forEach((b) => {
      if (typeof b._id === 'number') bucketMap[b._id] = b.count;
    });
    const confidenceBuckets = [0, 0.2, 0.4, 0.6, 0.8].map((k) => bucketMap[k] || 0);

    /* ── 6. Scan Volume Timeline (Daily Grouping) ── */
    const maxDays = range === '7d' ? 7 : range === '90d' ? 90 : range === 'all' ? 90 : 30;
    const timelineAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      {
        $group: {
          _id: {
            $dateToString: {
              format: '%Y-%m-%d',
              date: '$createdAt',
            },
          },
          count: { $sum: 1 },
        },
      },
      { $sort: { _id: 1 } },
      { $limit: maxDays },
    ]);

    const timeline = timelineAgg.map((t) => ({
      label: t._id.slice(5), // "MM-DD"
      count: t.count,
    }));

    /* ── 7. Recent Scans with Detected Pathologies ── */
    const recentScans = await ScanRecord.find(baseMatch)
      .sort({ createdAt: -1 })
      .limit(6)
      .select('filename patientId result confidence predictions createdAt processingMs')
      .lean();

    const formattedRecent = recentScans.map((s) => {
      const detected = (s.predictions || [])
        .filter(
          (p) =>
            p.disease !== 'No_Finding' &&
            (p.detected || (Array.isArray(p.models_agreeing) && p.models_agreeing.length >= 2))
        )
        .map((p) => ({
          disease: p.disease,
          prob: Math.round(p.probability * 100),
          consensus: p.consensus || `${p.models_agreeing?.length || 0}/3`,
        }));

      return {
        _id: s._id,
        patientId: s.patientId || 'N/A',
        filename: s.filename || 'radiograph.jpg',
        result: s.result || 'NORMAL',
        confidence: s.confidence != null ? Math.round(s.confidence * 100) : 0,
        detectedPathologies: detected,
        createdAt: s.createdAt,
        processingMs: s.processingMs || 0,
      };
    });

    res.json({
      totalScans,
      pathologyCount,
      normalCount,
      inconclusiveCount,
      consensusRate,
      avgConfidence,
      avgProcessingMs,
      confidenceBuckets,
      pathologyDistribution,
      consensusBreakdown,
      timeline,
      recentScans: formattedRecent,
      // Backward compatibility fields
      tbCount: pathologyCount,
    });
  } catch (err) {
    next(err);
  }
};
