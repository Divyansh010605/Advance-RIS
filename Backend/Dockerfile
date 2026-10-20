# syntax=docker/dockerfile:1
# ── Stage 1: Build Dependencies ───────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev || npm install --omit=dev

# ── Stage 2: Production Runtime ───────────────────────────────────────────────
FROM node:20-alpine

RUN apk add --no-cache dumb-init wget

ENV NODE_ENV=production
WORKDIR /app

# Copy production node_modules from builder
COPY --from=builder /app/node_modules ./node_modules
COPY . .

# Ensure upload and log directories exist with node ownership
RUN mkdir -p /app/uploads /app/logs && chown -R node:node /app

EXPOSE 5000

USER node

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "server.js"]
