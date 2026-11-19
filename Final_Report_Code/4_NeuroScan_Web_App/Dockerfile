# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:22-slim AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend with embedded frontend ────────────────────────────
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Embed the React build so FastAPI can serve it as static files
COPY --from=frontend-build /frontend/dist ./static

ENV PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
