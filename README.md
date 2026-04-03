---
title: OpenEnv Creative Auctioneer
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# OpenEnv Creative Auctioneer

A **privacy-native real-time bidding (RTB) ad auction** environment where an RL
agent acts as an autonomous Account Manager — navigating a 24-hour campaign
cycle, selecting ad creatives, pacing budgets, and assembling viral captions to
maximise Return on Ad Spend (ROAS) — all **without individual user identifiers**.

## Motivation

Programmatic advertising is a \$500 B+ industry where split-second bidding
decisions determine campaign success.  Existing RL benchmarks either use toy
grid-worlds or require proprietary data.  **OpenEnv-Auctioneer** fills this gap
with a fully open, dataset-calibrated simulation grounded in:

| Dataset | Role |
|---------|------|
| [MIND](https://msnews.github.io/) (Microsoft News) | CTR calibration + headline catalog |
| [iPinYou RTB](https://contest.ipinyou.com/) | Competitor bid distributions (Lognormal/hour) |
| [Vogue Dialogue](https://github.com/aimagelab/Vogue-Dialogue) | User persona bank |

All datasets are **optional** — the environment falls back to published
statistics so it runs out-of-the-box with zero downloads.

---

## Action Space

```python
class Action(BaseModel):
    bid_price: float          # USD bid for the RTB auction (≥ 0)
    headline_id: int          # Index into the 6-slot headlines catalog (0–5)
    creative_id: int          # Index into the 6-slot creatives catalog (0–5)
    generated_caption: str | None  # Free-text caption (hard_assembly only)
```

## Observation Space

```python
class Observation(BaseModel):
    hour_of_day: int          # Current hour (0–23)
    remaining_budget: float   # Remaining budget in USD
    spent_so_far: float       # Cumulative spend
    current_context: str      # "Fitness" | "Tech" | "Fashion" | "Gaming"
    news_category: str        # Fine-grained MIND subcategory
    viral_trend: str          # Current cultural trend token
    market_pressure: float    # Auction competitiveness [0, 1]
    ads_shown_this_session: int
    fatigue_level: float      # User fatigue [0, 1]
    carryover_boost: float    # Brand-recall CTR boost [0, 0.30]
    last_ctr: float           # Previous step CTR
    cumulative_revenue: float # Total revenue earned
```

## Reward Signal

| Outcome | Reward |
|---------|--------|
| Auction **won** | `adjusted_ctr × $15 − clearing_price` |
| Auction **lost** | `−$0.10` (missed opportunity) |
| Over-pacing (medium only) | `−$1.00` penalty |

Rewards are **per-step** (not sparse), providing continuous gradient signal.

---

## Tasks

### Level 1 — `easy_headline` (Easy)
**Objective:** Select the headline with the highest CTR for each context.
**Budget:** $100 | **Grader:** `mean(CTR_selected / CTR_oracle)` | **Target:** 0.75

### Level 2 — `medium_pacing` (Medium)
**Objective:** Pace $50 across 24 hours; retain ≥ 20% for peak hours (18–22).
**Budget:** $50 | **Grader:** `0.3×smoothness + 0.3×peak_survival + 0.4×revenue` | **Target:** 0.70

### Level 3 — `hard_assembly` (Hard)
**Objective:** Generate captions aligned with the viral trend AND win auctions.
**Budget:** $100 | **Grader:** `0.6×cosine_sim + 0.4×revenue_factor` | **Target:** 0.65

### Level 4 — `hard_sequencing` (Hard)
**Objective:** Plan 24-hour ad placements with carry-over brand-recall boosts.
Winning triggers +15%/+10%/+5% CTR for the next 3 hours. Cover ≥ 3 contexts for
a 20% diversity bonus.
**Budget:** $100 | **Grader:** `min(1.0, agent_conv/oracle_conv × diversity_mult)` | **Target:** 0.60

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerised execution)

### Local Development

```bash
pip install -r requirements.txt
python -c "from environment import OpenEnvAuctioneer; e = OpenEnvAuctioneer(); print(e.reset())"
```

### Docker Build & Run

```bash
# Build the image
docker build -t openenv-auctioneer .

# Run the FastAPI server (default)
docker run --rm -p 7860:7860 openenv-auctioneer

# Run inference directly inside the container
docker run --rm \
  -e HF_TOKEN=<your_key> \
  openenv-auctioneer python inference.py
```

### Inference Script

```bash
# Build image first, then run inference
docker build -t openenv-auctioneer .

LOCAL_IMAGE_NAME=openenv-auctioneer \
HF_TOKEN=<your_key> \
python inference.py
```

The inference script emits standardised `[START]`/`[STEP]`/`[END]` logs to stdout.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (inference) | API key for the LLM service |
| `API_BASE_URL` | No | LLM endpoint (default: HuggingFace router) |
| `MODEL_NAME` | No | Model identifier (default: Qwen/Qwen2.5-72B-Instruct) |
| `LOCAL_IMAGE_NAME` | Yes (inference) | Docker image name |
| `AUCTIONEER_TASK` | No | Task to run (default: `all`) |
| `MIND_SOURCE` | No | `local` / `huggingface` / `azure` |
| `USE_LLM_SIMULATOR` | No | Set `1` to enable Llama-3 User Simulator |

---

## Baseline Scores (Expected Ranges)

| Task | Expected Range | Notes |
|------|---------------|-------|
| `easy_headline` | 0.55 – 0.80 | Context→headline matching is learnable |
| `medium_pacing` | 0.45 – 0.70 | Requires budget discipline |
| `hard_assembly` | 0.40 – 0.65 | Caption quality + auction wins |
| `hard_sequencing` | 0.35 – 0.60 | Compared against DP oracle |

Scores depend on LLM quality and market stochasticity.  Run multiple episodes
for stable estimates.

---

## Project Structure

```
├── models.py          # Pydantic models: Action, Observation, Reward, Info
├── environment.py     # OpenEnvAuctioneer + graders + dataset layers
├── app.py             # FastAPI server (runs inside Docker)
├── inference.py       # Baseline inference script (mandatory format)
├── openenv.yaml       # OpenEnv metadata & task definitions
├── Dockerfile         # Container build
├── requirements.txt   # Python dependencies
├── test_sequencing.py # Unit tests for DP oracle grader
└── Datasets/          # Optional dataset mount point
```

## License

MIT
