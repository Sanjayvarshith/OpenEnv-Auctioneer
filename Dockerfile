# ─────────────────────────────────────────────────────────────────────────────
# OpenEnv Creative Auctioneer — Docker Image
# ─────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t openenv-auctioneer .
#
# Run (FastAPI server — default, used by inference.py & HF Space):
#   docker run --rm -p 7860:7860 openenv-auctioneer
#
# Run (inference agent directly):
#   docker run --rm -e HF_TOKEN=<key> openenv-auctioneer python inference.py
#
# Run (single task):
#   docker run --rm -e HF_TOKEN=<key> -e AUCTIONEER_TASK=easy_headline openenv-auctioneer python inference.py
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System deps for torch / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model so the container is self-contained
RUN python -<<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer cached ✓")
EOF

# Copy source files
COPY models.py environment.py inference.py app.py openenv.yaml ./

COPY Datasets.zip ./

RUN apt-get update && apt-get install -y unzip && unzip Datasets.zip -d ./ && rm Datasets.zip && rm -rf /var/lib/apt/lists/*

# Environment variable defaults (DATA_DIR removed so it falls back to the local folder)
ENV TASK=all \
    USE_LLM_SIMULATOR=0

EXPOSE 7860

# Default: run the FastAPI server (used by inference.py and HF Space)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
