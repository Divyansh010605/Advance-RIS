# syntax=docker/dockerfile:1
# ── Stage 1: Build React App ──────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies (utilize layer caching)
COPY package*.json ./
RUN npm ci || npm install

# Copy source
COPY . .

# Inject API URLs at build time
ARG REACT_APP_API_URL=/api
ARG REACT_APP_AI_API_URL=/ai-api
ENV REACT_APP_API_URL=${REACT_APP_API_URL}
ENV REACT_APP_AI_API_URL=${REACT_APP_AI_API_URL}

RUN npm run build

# ── Stage 2: Serve via Nginx ──────────────────────────────────────────────────
FROM nginx:1.25-alpine

# Copy built static files
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom Nginx configuration (with /api and /ai-api reverse proxying)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
