const User = require('../models/User');
const logger = require('./logger');

/**
 * Seeds the default admin user into the database if not present or ensures credentials match.
 * Default admin credentials:
 *   username: admin
 *   email: admin01@example.com
 *   password: admin123
 *   role: admin
 */
async function seedAdminUser() {
  try {
    const adminEmail = 'admin01@example.com';
    const adminUsername = 'admin';
    const adminPassword = 'admin123';

    // Find existing user by username or email
    let admin = await User.findOne({
      $or: [{ email: adminEmail }, { username: adminUsername }],
    }).select('+password');

    if (!admin) {
      admin = await User.create({
        username: adminUsername,
        email: adminEmail,
        password: adminPassword,
        role: 'admin',
        isActive: true,
      });
      logger.info(`Default admin user created successfully (username: ${adminUsername}, email: ${adminEmail})`);
    } else {
      let isModified = false;

      if (admin.role !== 'admin') {
        admin.role = 'admin';
        isModified = true;
      }
      if (!admin.isActive) {
        admin.isActive = true;
        isModified = true;
      }

      // Check if password matches admin123
      const passwordMatches = await admin.comparePassword(adminPassword);
      if (!passwordMatches) {
        admin.password = adminPassword;
        isModified = true;
      }

      if (isModified) {
        await admin.save();
        logger.info(`Default admin user credentials/role updated (username: ${adminUsername}, email: ${adminEmail}).`);
      } else {
        logger.info(`Default admin user ready (username: ${adminUsername}, email: ${adminEmail}).`);
      }
    }
  } catch (err) {
    logger.error('Error seeding default admin user:', err.message);
  }
}

module.exports = seedAdminUser;
