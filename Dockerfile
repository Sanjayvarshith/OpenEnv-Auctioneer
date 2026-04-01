# ─────────────────────────────────────────────────────────────────────────────
# OpenEnv Creative Auctioneer — Docker Image
# ─────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t openenv-auctioneer .
#
# Run (baseline agent, all tasks):
#   docker run --rm \
#     -e OPENAI_API_KEY=<your_key> \
#     -v /local/path/to/data:/data \     # optional: mount calibration datasets
#     openenv-auctioneer
#
# Run (single task):
#   docker run --rm -e OPENAI_API_KEY=<key> -e TASK=easy_headline openenv-auctioneer
#
# Run with LLM User Simulator (requires GPU + Llama-3 weights):
#   docker run --rm --gpus all \
#     -e OPENAI_API_KEY=<key> \
#     -e USE_LLM_SIMULATOR=1 \
#     -e LLM_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct \
#     -v /local/path/to/data:/data \
#     openenv-auctioneer
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
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer cached ✓")
EOF

# Copy source
COPY models.py environment.py baseline.py openenv.yaml ./

# Dataset mount point (read-only at runtime; all datasets are optional)
# Expected layout inside /data:
#   /data/ipinyou/              ← iPinYou RTB CSV files
#   /data/criteo/sampled_clicks.csv
#   /data/pitt_ads/ads_metadata.json
#   /data/vogue_dialogue/personas.json
VOLUME ["/data"]

# Environment variable defaults
ENV DATA_DIR=/data \
    TASK=all \
    USE_LLM_SIMULATOR=0

CMD ["python", "baseline.py"]
