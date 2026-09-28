require('dotenv').config();
const app = require('./app');
const connectDB = require('./config/database');
const seedAdminUser = require('./config/seedAdmin');
const logger = require('./config/logger');

const PORT = process.env.PORT || 5000;

async function startServer() {
  try {
    await connectDB();
    await seedAdminUser();
    const server = app.listen(PORT, '0.0.0.0', () => {
      logger.info(`TB Detection Backend running on port ${PORT}`);
    });

    // Graceful shutdown
    const shutdown = (signal) => {
      logger.info(`${signal} received. Shutting down gracefully...`);
      server.close(() => {
        logger.info('HTTP server closed.');
        process.exit(0);
      });
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
  } catch (err) {
    logger.error('Failed to start server:', err);
    process.exit(1);
  }
}

startServer();
