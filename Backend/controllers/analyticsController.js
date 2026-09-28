const ScanRecord = require('../models/ScanRecord');

/* ────────────────────────────────────────────────────────────────
   GET /api/analytics?range=30d
   Returns aggregated stats for the dashboard.
──────────────────────────────────────────────────────────────── */
exports.getAnalytics = async (req, res, next) => {
  try {
    const range   = req.query.range || '30d';
    const userId  = req.user._id;

    // compute start date
    const now   = new Date();
    let startDate = null;
    if (range !== 'all') {
      const days = range === '7d' ? 7 : range === '90d' ? 90 : 30;
      startDate  = new Date(now - days * 24 * 60 * 60 * 1000);
    }

    const dateFilter = startDate ? { createdAt: { $gte: startDate } } : {};
    const baseMatch  = { userId, ...dateFilter };

    /* ── 1. Result counts ── */
    const resultAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      { $group: { _id: '$result', count: { $sum: 1 } } },
    ]);
    const counts = { TB_DETECTED: 0, NORMAL: 0, INCONCLUSIVE: 0 };
    resultAgg.forEach(r => { if (r._id) counts[r._id] = r.count; });

    /* ── 2. Avg confidence + avg processing time ── */
    const avgAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      {
        $group: {
          _id:             null,
          avgConfidence:   { $avg: '$confidence' },
          avgProcessingMs: { $avg: '$processingMs' },
        },
      },
    ]);
    const avgConfidence   = avgAgg[0]?.avgConfidence   ?? null;
    const avgProcessingMs = avgAgg[0]?.avgProcessingMs ?? null;

    /* ── 3. Confidence buckets (0-20, 20-40, 40-60, 60-80, 80-100) ── */
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
    // Fill all 5 buckets (some may be missing if count is 0)
    const bucketMap = {};
    bucketAgg.forEach(b => { if (typeof b._id === 'number') bucketMap[b._id] = b.count; });
    const confidenceBuckets = [0, 0.2, 0.4, 0.6, 0.8].map(k => bucketMap[k] || 0);

    /* ── 4. Scans per day timeline ── */
    const days = range === '7d' ? 7 : range === '90d' ? 90 : range === 'all' ? 90 : 30;
    const timelineAgg = await ScanRecord.aggregate([
      { $match: baseMatch },
      {
        $group: {
          _id: {
            $dateToString: {
              format: '%Y-%m-%d',
              date:   '$createdAt',
            },
          },
          count: { $sum: 1 },
        },
      },
      { $sort: { _id: 1 } },
      { $limit: days },
    ]);

    const timeline = timelineAgg.map(t => ({
      label: t._id.slice(5),   // "MM-DD"
      count: t.count,
    }));

    /* ── 5. Recent 5 scans ── */
    const recentScans = await ScanRecord.find(baseMatch)
      .sort({ createdAt: -1 })
      .limit(5)
      .select('filename result confidence createdAt')
      .lean();

    res.json({
      tbCount:           counts.TB_DETECTED,
      normalCount:       counts.NORMAL,
      inconclusiveCount: counts.INCONCLUSIVE,
      avgConfidence,
      avgProcessingMs,
      confidenceBuckets,
      timeline,
      recentScans,
    });
  } catch (err) {
    next(err);
  }
};