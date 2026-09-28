/**
 * Privacy filter middleware.
 * Strips potentially sensitive metadata from incoming requests
 * before they are processed by controllers.
 *
 * Removes: x-forwarded-for details, unnecessary headers,
 *          and metadata fields from body that should not be stored.
 */
function privacyFilter(req, res, next) {
  // Remove identifying headers that are not needed downstream
  const headersToRemove = [
    'x-real-ip',
    'x-forwarded-for',
    'x-forwarded-host',
    'x-forwarded-proto',
    'cf-connecting-ip',
    'true-client-ip',
  ];

  headersToRemove.forEach((header) => {
    delete req.headers[header];
  });

  // Strip EXIF-style metadata fields from body if present
  if (req.body && typeof req.body === 'object') {
    const sensitiveBodyFields = ['gps', 'location', 'coordinates', 'exif', 'metadata', 'deviceInfo'];
    sensitiveBodyFields.forEach((field) => {
      delete req.body[field];
    });
  }

  next();
}

module.exports = privacyFilter;
