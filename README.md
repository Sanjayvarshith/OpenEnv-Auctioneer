---
title: OpenEnv Creative Auctioneer
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
| [MS-COCO Captions 2017](https://cocodataset.org/) | Ad + caption pool for `hard_assembly` |
| [Google Trends](https://github.com/GeneralMills/pytrends) / [Reddit](https://www.reddit.com/) | Live viral hashtag scraping |

All datasets are **optional** — the environment falls back to published
statistics so it runs out-of-the-box with zero downloads.

---

## Action Space

```python
class Action(BaseModel):
    bid_price: float          # USD bid for the RTB auction (≥ 0)
    headline_id: int          # Index into the 6-slot headlines catalog (0–5)
    creative_id: int          # Index into the 6-slot creatives catalog (0–5)
    generated_caption: str | None    # [hard_assembly] Rewritten caption with viral hashtags
    generated_hashtags: list[str] | None  # [hard_assembly] Chosen hashtags (e.g. ["#QuietLuxury", "#OOTD"])
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

    # hard_assembly only:
    live_hashtags: list[str]      # Real-time scraped viral hashtags
    image_description: str        # Source ad image description
    base_caption: str             # Base caption to rewrite
```

## Reward Signal

| Outcome | Reward |
|---------|--------|
| Auction **won** | `adjusted_ctr × $15 − clearing_price` |
| Auction **lost** | `−$0.10` (missed opportunity) |
| Over-pacing (medium only) | `−$1.00` penalty |
| Assembly bonus (hard_assembly) | `+composite_score × $8.00` |

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
**Objective:** Given an ad image description + base caption + live viral hashtags,
**generate a new caption** that is simultaneously viral, coherent with the image,
and creatively novel — while also winning auctions profitably.

**Budget:** $120 | **Target:** 0.65

**The RL loop (what the LLM agent does each step):**
```
1. Agent receives: image_description, base_caption, live_hashtags[], viral_trend
2. Agent must:
   a. Select 2–4 relevant hashtags from live_hashtags (scraped from Google Trends / Reddit)
   b. Rewrite the base caption to weave those hashtags into natural ad copy
   c. Add its own creative words (target 30–50% novel vocabulary)
   d. Keep the caption coherent with the source image
   e. Set a profitable bid price
3. Grader scores the assembled caption on 4 axes:
   • 35% — Hashtag relevance  (cosine_sim of each hashtag vs viral_trend)
   • 35% — Caption-trend alignment  (cosine_sim of caption vs viral_trend)
   • 20% — Caption-image coherence  (cosine_sim of caption vs image_description)
   • 10% — Novelty  (fraction of new words vs base_caption, target ~40%)
4. Reward = auction_reward + composite_score × $8.00 bonus
```

**Data sources for hard_assembly:**
- **Ad creatives**: MS-COCO Captions 2017 (val annotations) bucketed into Fitness/Tech/Fashion/Gaming by keyword matching. Falls back to 30-entry built-in seed pool.
- **Viral hashtags**: `ViralHashtagScraper` queries Google Trends (via `pytrends`) and Reddit `/r/popular/hot.json` (public, no auth). Blends with static seed hashtags per context and trend. Cached for 1 hour.

### Level 4 — `hard_sequencing` (Hard)
**Objective:** Plan 24-hour ad placements with carry-over brand-recall boosts.
Winning triggers +15%/+10%/+5% CTR for the next 3 hours. Cover ≥ 3 contexts for
a 20% diversity bonus.
**Budget:** $100 | **Grader:** `min(1.0, agent_conv/oracle_conv × diversity_mult)` | **Target:** 0.60

---

## Grading Details

### `EasyHeadlineGrader`
```
step_score  = CTR_selected / CTR_oracle
final_score = mean(step_scores)                         // [0.0, 1.0]
```

### `MediumPacingGrader`
```
smoothness     = 1 − mean(|hourly_spend − ideal_spend| / ideal_spend)
peak_survival  = 1.0 if remaining_budget ≥ 20% at hour 18, else 0.0
revenue_factor = min(1.0, total_revenue / $30)

final_score = 0.30 × smoothness + 0.30 × peak_survival + 0.40 × revenue_factor
```

### `HardAssemblyGrader` — 4-Axis Composite

| Axis | Weight | Metric |
|------|--------|--------|
| Hashtag Relevance | 0.35 | `mean(cosine_sim(hashtag, viral_trend))` |
| Caption-Trend Alignment | 0.35 | `cosine_sim(caption, viral_trend)` |
| Caption-Image Coherence | 0.20 | `cosine_sim(caption, image_description)` |
| Novelty | 0.10 | `1 − |novel_fraction − 0.40| / 0.60` |

```
composite = Σ (weight × axis_score)

final_score = 0.60 × mean(composite_scores)
            + 0.40 × min(1.0, total_revenue / $55)
```

### `HardSequencingGrader`
```
agent_conversions  = Σ [CTR_t × (1 + carryover_boost_t) × $15]
oracle_conversions = DP-optimal bid/skip sequence with carry-over

diversity_mult = 1.20 if ≥3 distinct contexts won, else 1.0

final_score = min(1.0, agent_conv / oracle_conv × diversity_mult)
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  OpenEnvAuctioneer (Gym-style environment)                │
│                                                           │
│  ┌──────────────────┐   ┌───────────────────────────────┐ │
│  │  Market Engine    │   │   User Simulator              │ │
│  │  (Statistical)    │   │   (Semantic / LLM)            │ │
│  │                   │   │                               │ │
│  │  iPinYou RTB logs │   │  SentenceTransformer          │ │
│  │  → Lognormal per  │   │  all-MiniLM-L6-v2            │ │
│  │    hour bucket    │   │  + optional Llama-3-8B        │ │
│  └──────────────────┘   └───────────────────────────────┘ │
│                                                           │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  MIND Dataset Layer  (Microsoft News Dataset)         │ │
│  │  behaviours.tsv  →  CTRCalibrator                     │ │
│  │  news.tsv        →  MINDCreativePool (headlines)      │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Ad + Caption Dataset  (MS-COCO Captions 2017)        │ │
│  │  → image_description + base_caption per step          │ │
│  │  → ViralHashtagScraper (pytrends + Reddit + seeds)    │ │
│  │  → agent rewrites caption with viral hashtags         │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Grader (task-specific, deterministic 0.0–1.0)        │ │
│  │   Level 1: easy_headline  → headline CTR lookup       │ │
│  │   Level 2: medium_pacing  → pacing + survival         │ │
│  │   Level 3: hard_assembly  → 4-axis composite score    │ │
│  │   Level 4: hard_sequencing→ DP oracle comparison      │ │
│  └───────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

---

## Models

| Model | Role | Always Active? |
|-------|------|----------------|
| `all-MiniLM-L6-v2` (SentenceTransformer) | Semantic CTR scoring + grader cosine similarity | ✅ Yes |
| `Meta-Llama-3-8B-Instruct` (4-bit) | Richer LLM-based CTR scoring | ❌ Optional (`USE_LLM_SIMULATOR=1`) |

When the LLM simulator is active: `final_ctr = 0.60 × llm_ctr + 0.40 × semantic_ctr`

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
| `COCO_SOURCE` | No | `local` / `url` (auto-download COCO annotations) |
| `USE_LLM_SIMULATOR` | No | Set `1` to enable Llama-3 User Simulator |

---

## Baseline Scores (Expected Ranges)

| Task | Expected Range | Notes |
|------|---------------|-------|
| `easy_headline` | 0.55 – 0.80 | Context→headline matching is learnable |
| `medium_pacing` | 0.45 – 0.70 | Requires budget discipline |
| `hard_assembly` | 0.40 – 0.65 | Caption quality + hashtag matching + auction wins |
| `hard_sequencing` | 0.35 – 0.60 | Compared against DP oracle |

Scores depend on LLM quality and market stochasticity.  Run multiple episodes
for stable estimates.

---

## Project Structure

```
├── models.py          # Pydantic models: Action, Observation, Reward, Info
├── environment.py     # OpenEnvAuctioneer + graders + dataset layers
│   ├── MINDLoader          # MIND dataset loader (HF / Azure / local)
│   ├── MarketCalibrator    # iPinYou-based auction price simulator
│   ├── CTRCalibrator       # MIND-based CTR lookup tables
│   ├── MINDCreativePool    # 6-slot headline/creative catalog from news.tsv
│   ├── PersonaBank         # Vogue Dialogue persona sampling
│   ├── ViralHashtagScraper # Live hashtag scraping (pytrends + Reddit)
│   ├── AdCaptionDataset    # COCO-based ad image+caption pool
│   ├── UserSimulator       # Semantic + optional LLM CTR scoring
│   ├── EasyHeadlineGrader  # Level 1 grader
│   ├── MediumPacingGrader  # Level 2 grader
│   ├── HardAssemblyGrader  # Level 3 grader (4-axis composite)
│   ├── HardSequencingGrader# Level 4 grader (DP oracle)
│   └── OpenEnvAuctioneer   # Main Gym-style env class
├── app.py             # FastAPI server (runs inside Docker)
├── inference.py       # Baseline inference script (mandatory format)
├── openenv.yaml       # OpenEnv metadata & task definitions
├── Dockerfile         # Container build
├── requirements.txt   # Python dependencies
├── test_sequencing.py # Unit tests for DP oracle grader
└── Datasets/          # Optional dataset mount point
```

## References

1. **MIND**: Wu et al. (2020) — *"MIND: A Large-scale Dataset for News Recommendation"*, ACL 2020. [msnews.github.io](https://msnews.github.io/)
2. **iPinYou RTB**: Zhang et al. (2014) — *"Real-Time Bidding Benchmarking with iPinYou Dataset"*. [contest.ipinyou.com](https://contest.ipinyou.com/)
3. **MS-COCO Captions**: Lin et al. (2014) — *"Microsoft COCO: Common Objects in Context"*. [cocodataset.org](https://cocodataset.org/)
4. **SentenceTransformers**: Reimers & Gurevych (2019) — *"Sentence-BERT"*. [sbert.net](https://www.sbert.net/)

## License

MIT
