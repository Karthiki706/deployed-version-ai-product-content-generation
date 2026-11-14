FROM python:3.12-slim

WORKDIR /workspace

# Install essential system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements (copied from app/requirements.txt)
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY . .

# Expose FastAPI backend (8000) and Streamlit frontend (8501)
EXPOSE 8000
EXPOSE 8501

# Default runtime entrypoint (overrideable via Docker Compose)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
