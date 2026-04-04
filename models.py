"""
models.py — Typed data contracts for the OpenEnv Creative Auctioneer.

All tensors / vectors are represented as plain Python types so the environment
stays framework-agnostic (no hard dependency on PyTorch at this layer).

Dataset provenance (v0.4):
  CTR calibration  → MIND (Microsoft News Dataset)  behaviours.tsv + news.tsv
  Market engine    → iPinYou Global RTB logs         (Lognormal per hour)
  Persona bank     → Vogue Dialogue Dataset
  Ad+Caption pool  → MS-COCO Captions  OR  Google Conceptual Captions CC3M
  Viral hashtags   → Pytrends / Hashtagify / static fallback table
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reward  (typed wrapper so step() signature is explicit)
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    value: float = Field(...,
        description="Scalar step reward. "
                    "Positive = profitable auction win; negative = missed bid penalty.")


# ---------------------------------------------------------------------------
# Observation  (what the agent *sees* each step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    # ── Temporal ──────────────────────────────────────────────────────────
    hour_of_day: int = Field(..., ge=0, le=23,
        description="Current hour of the 24-hour campaign cycle (0–23).")

    # ── Budget / Pacing ────────────────────────────────────────────────────
    remaining_budget: float = Field(...,
        description="Remaining daily budget in USD.")
    spent_so_far: float = Field(default=0.0,
        description="Cumulative spend in USD up to this step.")

    # ── Contextual Signals (Privacy-Native — no user IDs) ──────────────────
    current_context: str = Field(...,
        description="Content category derived from MIND news.tsv taxonomy "
                    "(e.g. 'Fitness', 'Tech', 'Fashion', 'Gaming').")
    news_category: str = Field(default="",
        description="Fine-grained MIND subcategory (e.g. 'nfl', 'gadgets'). "
                    "Provides richer signal than coarse context alone.")
    viral_trend: str = Field(...,
        description="Current cultural viral token surfaced from Reels "
                    "(e.g. 'Quiet Luxury', 'Eco-Friendly', 'Cyberpunk', 'Minimalism').")

    # ── hard_assembly: live scraped hashtags + source creative ─────────────
    live_hashtags: List[str] = Field(default_factory=list,
        description="[hard_assembly] Real-time scraped viral hashtags from "
                    "Google Trends / Reddit.  The agent selects which to use "
                    "and weaves them into generated_caption. "
                    "Example: ['#QuietLuxury', '#OOTD', '#SlowFashion'].")
    image_description: str = Field(default="",
        description="[hard_assembly] Text description of the source ad image "
                    "from AdCaptionDataset (COCO or seed pool). "
                    "Agent caption must stay coherent with this.")
    base_caption: str = Field(default="",
        description="[hard_assembly] Base caption from AdCaptionDataset. "
                    "Agent rewrites this to incorporate viral hashtags.")

    # ── Market Signals ─────────────────────────────────────────────────────
    market_pressure: float = Field(default=0.5, ge=0.0, le=1.0,
        description="Normalised indicator of how competitive the auction is "
                    "this hour (0 = cheap, 1 = very expensive).")

    # ── Session State ──────────────────────────────────────────────────────
    ads_shown_this_session: int = Field(default=0,
        description="Number of ads already shown; drives the fatigue penalty.")
    fatigue_level: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Accumulated user-fatigue penalty (0 = fresh, 1 = fully fatigued).")
    carryover_boost: float = Field(default=0.0, ge=0.0, le=1.0,
        description="[hard_sequencing] Carry-over CTR boost from winning prior auctions.")

    # ── Performance Feedback (delayed by 1 step) ───────────────────────────
    last_ctr: float = Field(default=0.0, ge=0.0, le=1.0,
        description="CTR returned by the User Simulator on the previous step.")
    cumulative_revenue: float = Field(default=0.0,
        description="Total revenue earned so far.")


# ---------------------------------------------------------------------------
# Action  (what the agent *does* each step)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    # Continuous: auction bid
    bid_price: float = Field(..., ge=0.0,
        description="Bid submitted to the RTB auction in USD.")

    # Discrete: creative selection
    headline_id: int = Field(..., ge=0, le=5,
        description="Index into the Headlines Catalog (0–5).")
    creative_id: int = Field(..., ge=0, le=5,
        description="Index into the Creatives Catalog (0–5).")

    # ── hard_assembly fields ────────────────────────────────────────────────
    generated_caption: Optional[str] = Field(default=None,
        description="[hard_assembly] Final assembled caption — should incorporate "
                    "viral hashtags and remain coherent with the source image. "
                    "Leave None for easy/medium tasks.")

    generated_hashtags: Optional[List[str]] = Field(default=None,
        description="[hard_assembly] List of hashtag strings (with #) that the agent "
                    "chose to include. The agent must scrape these from ViralHashtagScraper "
                    "and select which ones to weave into generated_caption. "
                    "Example: ['#QuietLuxury', '#OOTD', '#SlowFashion']. "
                    "Leave None for easy/medium/sequencing tasks.")


# ---------------------------------------------------------------------------
# Per-step Info  (returned alongside reward; not part of observation)
# ---------------------------------------------------------------------------

class Info(BaseModel):
    task_id: str
    current_step: int
    total_revenue: float
    clearing_price: float = Field(default=0.0,
        description="The winning competitor bid price this step.")
    auction_won: bool = Field(default=False,
        description="Whether the agent won the auction this step.")
    raw_ctr: float = Field(default=0.0,
        description="CTR before fatigue penalty applied.")
    adjusted_ctr: float = Field(default=0.0,
        description="CTR after fatigue penalty.")

    # ── Per-task grader scores ────────────────────────────────────────────
    task_score: float = Field(..., ge=0.0, le=1.0,
        description="Final 0.0–1.0 task-completion score.")

    # Level 1 sub-score
    headline_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="[easy_headline] CTR_selected / CTR_best for this context.")

    # Level 2 sub-score
    pacing_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="[medium_pacing] Budget-smoothness and peak-hour survival bonus.")

    # Level 3 sub-scores (all three axes)
    clip_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="[hard_assembly] Composite grader score (0.35×hashtag + 0.35×align + 0.30×coherence).")
    hashtag_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="[hard_assembly] Mean cosine_sim(chosen_hashtag, viral_trend).")
    caption_trend_alignment: float = Field(default=0.0, ge=0.0, le=1.0,
        description="[hard_assembly] cosine_sim(final_caption, viral_trend).")
    caption_image_coherence: float = Field(default=0.0, ge=0.0, le=1.0,
        description="[hard_assembly] cosine_sim(final_caption, image_description).")
    chosen_hashtags: List[str] = Field(default_factory=list,
        description="[hard_assembly] Hashtags the agent chose this step.")
    assembly_reward_bonus: float = Field(default=0.0,
        description="[hard_assembly] Extra reward granted for viral alignment quality.")

    # Level 4 sub-scores
    sequencing_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="[hard_sequencing] agent_conversions / oracle_conversions × diversity.")
    contexts_covered: int = Field(default=0,
        description="[hard_sequencing] Number of distinct contexts won at least once.")
    diversity_multiplier: float = Field(default=1.0,
        description="[hard_sequencing] Bonus multiplier for covering ≥3 contexts.")