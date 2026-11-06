# Frontend image: build the static bundle with node, serve it with nginx.

# --- stage 1: build ---
FROM node:20-alpine AS build
WORKDIR /app

# Install from the lockfile first so the dependency layer is cached across source edits.
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .

# Vite inlines VITE_* values at BUILD time, so they are build args, not runtime env.
# The API URL is what the *browser* calls (the published host port), not the compose
# service name.
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_API_KEY=change-me
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_API_KEY=$VITE_API_KEY
RUN npm run build

# --- stage 2: serve ---
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1
