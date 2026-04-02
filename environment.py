#TODO: add specific headlines for easy task and mak eit to output htat headline
#TODO:download datasets/hso them elsewhere and source them here and check the compatibility of the datasets
"""
environment.py — OpenEnv Creative Auctioneer
============================================

Architecture
------------
┌─────────────────────────────────────────────────────┐
│  OpenEnvAuctioneer (Gym-style environment)           │
│                                                       │
│  ┌──────────────────┐   ┌─────────────────────────┐  │
│  │  Market Engine    │   │   User Simulator         │  │
│  │  (Statistical)    │   │   (Semantic / LLM)       │  │
│  │                   │   │                          │  │
│  │  iPinYou RTB logs │   │  SentenceTransformer     │  │
│  │  → Lognormal per  │   │  all-MiniLM-L6-v2        │  │
│  │    hour bucket    │   │  + optional Llama-3-8B   │  │
│  └──────────────────┘   └─────────────────────────┘  │
│                                                       │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Grader (task-specific, deterministic 0.0–1.0)   │ │
│  │   Level 1: easy_headline  → headline CTR lookup  │ │
│  │   Level 2: medium_pacing  → pacing + survival    │ │
│  │   Level 3: hard_assembly  → CLIP cosine sim      │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘

Dataset Calibration (lazy-loaded at first reset)
-------------------------------------------------
• iPinYou RTB logs   → per-hour Lognormal (μ, σ) for competitor bids
• Criteo Click Logs  → per-context baseline CTR lookup table
• Pitt Image Ads     → creative pool metadata (image path + caption)
• Vogue Dialogue     → user persona bank for User Simulator prompts

If the datasets are absent (default dev mode) the environment falls back to
analytic approximations so that the code runs out-of-the-box without downloads.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

from models import Action, Info, Observation, Reward

# ---------------------------------------------------------------------------
# Paths — datasets should be mounted here inside Docker; absent → fallback
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(os.environ.get("DATA_DIR", "/data"))

IPINYOU_PATH = DATA_DIR / "ipinyou"          # .csv files per campaign
CRITEO_PATH  = DATA_DIR / "criteo"           # sampled_clicks.csv
PITT_PATH    = DATA_DIR / "pitt_ads"         # ads_metadata.json
VOGUE_PATH   = DATA_DIR / "vogue_dialogue"   # personas.json


# ===========================================================================
# Helper: Dataset Calibration Layer
# ===========================================================================

class MarketCalibrator:
    """
    Reads iPinYou RTB logs and computes per-hour Lognormal (mu, sigma)
    parameters for the competitor clearing-price distribution.

    Column layout assumed (from contest.ipinyou.com):
        bid_id, timestamp, ... , bidprice, payprice, ...

    If the dataset is missing, falls back to analytic approximations
    derived from published iPinYou statistics (Zhang et al., 2014).
    """

    # Published analytic fallback: (mu, sigma) per 6-hour bucket
    # Source: iPinYou Global RTB Bidding Algorithm Competition paper
    _ANALYTIC_PARAMS: Dict[int, Tuple[float, float]] = {
        # hour → (mu, sigma)   (lognormal of payprice in ¥, rescaled to $)
        0:  (-0.80, 0.30),
        1:  (-0.85, 0.30),
        2:  (-0.90, 0.28),
        3:  (-0.92, 0.28),
        4:  (-0.88, 0.29),
        5:  (-0.75, 0.30),
        6:  (-0.55, 0.32),
        7:  (-0.40, 0.33),
        8:  (-0.20, 0.35),
        9:  (-0.10, 0.36),
        10: (-0.05, 0.37),
        11: ( 0.00, 0.38),
        12: ( 0.10, 0.40),   # noon peak
        13: ( 0.05, 0.39),
        14: ( 0.00, 0.38),
        15: (-0.05, 0.37),
        16: ( 0.00, 0.38),
        17: ( 0.05, 0.39),
        18: ( 0.12, 0.41),   # 6 PM peak
        19: ( 0.08, 0.40),
        20: ( 0.02, 0.38),
        21: (-0.10, 0.36),
        22: (-0.30, 0.33),
        23: (-0.60, 0.31),
    }

    def __init__(self) -> None:
        self._params: Dict[int, Tuple[float, float]] = {}
        self._calibrated = False

    def calibrate(self) -> None:
        """Try to load from iPinYou CSV files; fall back to analytic params."""
        if not IPINYOU_PATH.exists():
            print("[MarketCalibrator] iPinYou dataset not found → using analytic fallback.")
            self._params = dict(self._ANALYTIC_PARAMS)
            self._calibrated = True
            return

        hour_prices: Dict[int, List[float]] = {h: [] for h in range(24)}
        for csv_file in IPINYOU_PATH.glob("*.csv"):
            try:
                with open(csv_file, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # iPinYou timestamp: YYYYMMDDHHMMSS
                        ts  = row.get("timestamp", "")
                        pay = row.get("payprice", "") or row.get("bidprice", "")
                        if len(ts) >= 10 and pay:
                            hour = int(ts[8:10])
                            price_usd = float(pay) / 1000.0  # ¥ → rough $ rescale
                            hour_prices[hour].append(price_usd)
            except Exception as exc:
                print(f"[MarketCalibrator] Skipping {csv_file.name}: {exc}")

        for h in range(24):
            prices = hour_prices[h]
            if len(prices) >= 10:
                log_prices = np.log(np.clip(prices, 1e-6, None))
                self._params[h] = (float(np.mean(log_prices)),
                                   float(np.std(log_prices)) or 0.30)
            else:
                self._params[h] = self._ANALYTIC_PARAMS[h]

        self._calibrated = True
        print(f"[MarketCalibrator] Calibrated from iPinYou data ✓")

    def sample_clearing_price(self, hour: int) -> float:
        if not self._calibrated:
            self.calibrate()
        mu, sigma = self._params.get(hour, (-0.5, 0.30))
        price = np.random.lognormal(mean=mu, sigma=sigma)
        return round(float(np.clip(price, 0.01, 10.0)), 2)

    def market_pressure(self, hour: int) -> float:
        """Normalised [0,1] indicator of how expensive this hour is."""
        mu, _ = self._params.get(hour, (-0.5, 0.30))
        # Map mu from [-1, 0.2] → [0, 1]
        return float(np.clip((mu + 1.0) / 1.2, 0.0, 1.0))


# --------------------------------------------------------
#This one is used becuase we want to lookup baseline_ctr for each of the contexts-------------------
#Drops the int features
#

class CTRCalibrator:
    """
    Reads the Criteo sampled click log and builds a lookup table:
        context_category → baseline CTR

    Criteo column layout (1TB log, sampled):
        label, int_feat×13, cat_feat×26

    We bucket samples into our 4 contexts by hash(cat_feat_0) % 4.
    If dataset absent, uses published industry CTR benchmarks.
    """

    _BENCHMARK_CTR: Dict[str, float] = {
        "Fitness":  0.045,
        "Tech":     0.038,
        "Fashion":  0.052,
        "Gaming":   0.060,
    }
    _CONTEXTS = ["Fitness", "Tech", "Fashion", "Gaming"]

    def __init__(self) -> None:
        self._ctr_table: Dict[str, float] = {}

    def calibrate(self) -> None:
        sampled = CRITEO_PATH / "sampled_clicks.csv"
        if not sampled.exists():
            print("[CTRCalibrator] Criteo dataset not found → using benchmark CTR.")
            self._ctr_table = dict(self._BENCHMARK_CTR)
            return

        bucket_clicks   = {c: 0 for c in self._CONTEXTS}
        bucket_total    = {c: 0 for c in self._CONTEXTS}

        try:
            with open(sampled, newline="", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 2:
                        continue
                    label = int(row[0])
                    # Use first categorical feature as context proxy
                    cat_hash = hash(row[14]) if len(row) > 14 else random.randint(0, 3)
                    context = self._CONTEXTS[abs(cat_hash) % 4]
                    bucket_total[context]  += 1
                    bucket_clicks[context] += label
            for ctx in self._CONTEXTS:
                if bucket_total[ctx] > 0:
                    self._ctr_table[ctx] = bucket_clicks[ctx] / bucket_total[ctx]
                else:
                    self._ctr_table[ctx] = self._BENCHMARK_CTR[ctx]
            print("[CTRCalibrator] Calibrated from Criteo data ✓")
        except Exception as exc:
            print(f"[CTRCalibrator] Error reading Criteo data: {exc} → benchmark CTR.")
            self._ctr_table = dict(self._BENCHMARK_CTR)

    def baseline_ctr(self, context: str) -> float:
        if not self._ctr_table:
            self.calibrate()
        return self._ctr_table.get(context, 0.04)


# ---------------------------------------------------------------------------

class CreativePool:
    """
    Loads Pitt Image Ads metadata (ads_metadata.json) to build a richer
    creative catalog.  Falls back to the hard-coded 6-item catalog if
    the dataset is absent.
    """

    _FALLBACK_HEADLINES: Dict[int, str] = {
        0: "Push your limits every single day.",
        1: "Next-generation processing power.",
        2: "Elevate your everyday style.",
        3: "Level up your competitive play.",
        4: "Sustainable choices for a better tomorrow.",
        5: "Uncompromising quality and elegance.",
    }
    _FALLBACK_CREATIVES: Dict[int, str] = {
        0: "A runner silhouetted against a mountain sunrise.",
        1: "A glowing silicon microchip on a motherboard.",
        2: "A model wearing a tailored coat on a city street.",
        3: "An RGB mechanical keyboard in a dark room.",
        4: "Product packaging made from recycled kraft paper.",
        5: "A gold watch resting on black velvet.",
    }
    # Map each headline/creative to its best-fit context
    _CONTEXT_AFFINITY: Dict[int, str] = {
        0: "Fitness",
        1: "Tech",
        2: "Fashion",
        3: "Gaming",
        4: "Fitness",   # eco-fitness overlap
        5: "Fashion",
    }

    def __init__(self) -> None:
        self.headlines  = dict(self._FALLBACK_HEADLINES)
        self.creatives  = dict(self._FALLBACK_CREATIVES)
        self.context_affinity = dict(self._CONTEXT_AFFINITY)

    def load(self) -> None:
        meta_path = PITT_PATH / "ads_metadata.json"
        if not meta_path.exists():
            print("[CreativePool] Pitt Ads dataset not found → using fallback catalog.")
            return
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            # Expect list of {"id": int, "headline": str, "image_desc": str,
            #                  "category": str}
            for item in meta[:6]:    # keep catalog size = 6 for compatibility
                idx = int(item["id"])
                self.headlines[idx]       = item.get("headline", self.headlines[idx])
                self.creatives[idx]       = item.get("image_desc", self.creatives[idx])
                self.context_affinity[idx]= item.get("category", self.context_affinity.get(idx, "Fitness"))
            print("[CreativePool] Loaded Pitt Ads metadata ✓")
        except Exception as exc:
            print(f"[CreativePool] Error: {exc} → fallback catalog.")

    def best_headline_for_context(self, context: str) -> int:
        """Return the headline_id with the highest affinity for this context."""
        matches = [idx for idx, ctx in self.context_affinity.items() if ctx == context]
        return matches[0] if matches else 0


# ---------------------------------------------------------------------------

class PersonaBank:
    """
    Loads Vogue Dialogue personas and serves one per context/trend combo.
    Falls back to template strings if dataset is absent.
    """

    _FALLBACK = {
        "Fitness":  "A health-conscious millennial tracking their morning run.",
        "Tech":     "An early-adopter software engineer browsing product specs.",
        "Fashion":  "A style-forward Gen-Z creator researching outfit inspo.",
        "Gaming":   "A competitive gamer watching esports highlight reels.",
    }

    def __init__(self) -> None:
        self._personas: List[str] = []

    def load(self) -> None:
        persona_file = VOGUE_PATH / "personas.json"
        if not persona_file.exists():
            print("[PersonaBank] Vogue dataset not found → using template personas.")
            return
        try:
            with open(persona_file, encoding="utf-8") as f:
                data = json.load(f)
            self._personas = [p["description"] for p in data if "description" in p]
            print(f"[PersonaBank] Loaded {len(self._personas)} personas ✓")
        except Exception as exc:
            print(f"[PersonaBank] Error: {exc} → template personas.")

    def sample(self, context: str, trend: str) -> str:
        if self._personas:
            return random.choice(self._personas)
        base = self._FALLBACK.get(context,
               "A social media user scrolling through their feed.")
        return f"{base} Currently interested in the '{trend}' aesthetic."


# ===========================================================================
# User Simulator
# ===========================================================================

class UserSimulator:
    """
    Computes click-through probability using:

      1. SentenceTransformer (all-MiniLM-L6-v2) — always available.
         Maps persona × ad → cosine similarity → CTR.

      2. Optional Llama-3-8B (4-bit via bitsandbytes) for richer scoring.
         Activated if USE_LLM_SIMULATOR=1 env var is set and GPU is available.

    The two signals are blended:  ctr = α·llm_ctr + (1-α)·semantic_ctr
    where α=0.6 when LLM is available, else α=0 (pure semantic).
    """

    _CTR_MIN  = 0.02
    _CTR_MAX  = 0.55

    def __init__(self, persona_bank: PersonaBank,
                 creative_pool: CreativePool,
                 ctr_calibrator: CTRCalibrator) -> None:
        self.persona_bank    = persona_bank
        self.creative_pool   = creative_pool
        self.ctr_calibrator  = ctr_calibrator

        print("[UserSimulator] Loading SentenceTransformer (all-MiniLM-L6-v2)…")
        self._smodel = SentenceTransformer("all-MiniLM-L6-v2")
        self._llm    = None

        if os.environ.get("USE_LLM_SIMULATOR", "0") == "1":
            self._try_load_llm()

    # ------------------------------------------------------------------
    def _try_load_llm(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)
            model_id = os.environ.get("LLM_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
            print(f"[UserSimulator] Loading LLM: {model_id} (4-bit)…")
            tok   = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                        model_id, quantization_config=bnb_cfg, device_map="auto")
            self._llm = (tok, model)
            print("[UserSimulator] LLM loaded ✓")
        except Exception as exc:
            print(f"[UserSimulator] LLM load failed ({exc}) → semantic-only mode.")

    # ------------------------------------------------------------------
    def _semantic_ctr(self, persona: str, ad_text: str,
                      context: str) -> float:
        baseline = self.ctr_calibrator.baseline_ctr(context)
        u_emb = self._smodel.encode(persona,  convert_to_tensor=True)
        a_emb = self._smodel.encode(ad_text,  convert_to_tensor=True)
        cos   = float(util.cos_sim(u_emb, a_emb).item())   # -1 … +1
        # Scale: baseline ± 50% based on cosine similarity
        ctr   = baseline * (1.0 + cos)
        return float(np.clip(ctr, self._CTR_MIN, self._CTR_MAX))

    # ------------------------------------------------------------------
    def _llm_ctr(self, persona: str, ad_text: str) -> float:
        """Ask Llama-3 to return a click probability in [0,1]."""
        if self._llm is None:
            return 0.0
        tok, model = self._llm
        prompt = (
            f"You are a {persona}.\n"
            f"You are shown this advertisement:\n\"{ad_text}\"\n"
            "On a scale from 0.00 to 1.00, what is the probability you would "
            "click this ad? Reply with ONLY a decimal number, nothing else."
        )
        try:
            import torch
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, temperature=0.1)
            reply = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
            val = float(reply.split()[0])
            return float(np.clip(val, 0.0, 1.0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def compute_ctr(self, action: Action, context: str, trend: str,
                    fatigue: float) -> Tuple[float, float]:
        """
        Returns (raw_ctr, adjusted_ctr) where adjusted_ctr accounts for
        per-session user fatigue.
        """
        persona  = self.persona_bank.sample(context, trend)
        headline = self.creative_pool.headlines.get(action.headline_id, "")
        creative = self.creative_pool.creatives.get(action.creative_id, "")

        # If hard_assembly caption provided, use it in place of headline
        if action.generated_caption:
            ad_text = f"{action.generated_caption} {creative}"
        else:
            ad_text = f"{headline} {creative}"

        sem_ctr = self._semantic_ctr(persona, ad_text, context)

        if self._llm is not None:
            llm_ctr  = self._llm_ctr(persona, ad_text)
            raw_ctr  = 0.6 * llm_ctr + 0.4 * sem_ctr
        else:
            raw_ctr  = sem_ctr

        # Fatigue: each previously shown ad reduces CTR by 5%
        adjusted_ctr = float(np.clip(raw_ctr * (1.0 - fatigue), 0.0, 1.0))
        return round(raw_ctr, 4), round(adjusted_ctr, 4)


# ===========================================================================
# Graders  (deterministic, 0.0 – 1.0)
# ===========================================================================

class EasyHeadlineGrader:
    """
    Level 1: Headline Selection
    Score = CTR_selected / CTR_best_possible

    'Best possible' is the CTR achieved by the oracle headline for the
    current context (highest semantic alignment in the creative pool).
    """

    def __init__(self, user_sim: UserSimulator,
                 creative_pool: CreativePool) -> None:
        self._sim   = user_sim
        self._pool  = creative_pool
        self._episode_scores: List[float] = []

    def score_step(self, action: Action, context: str, trend: str,
                   adjusted_ctr: float) -> float:
        # Compute oracle CTR (best headline for this context)
        best_id      = self._pool.best_headline_for_context(context)
        oracle_action = Action(bid_price=action.bid_price,
                               headline_id=best_id,
                               creative_id=best_id)
        _, oracle_ctr = self._sim.compute_ctr(oracle_action, context, trend,
                                              fatigue=0.0)
        oracle_ctr    = max(oracle_ctr, 1e-6)
        step_score    = float(np.clip(adjusted_ctr / oracle_ctr, 0.0, 1.0))
        self._episode_scores.append(step_score)
        return step_score

    def episode_score(self) -> float:
        if not self._episode_scores:
            return 0.0
        return round(float(np.mean(self._episode_scores)), 4)

    def reset(self) -> None:
        self._episode_scores.clear()


# ---------------------------------------------------------------------------

class MediumPacingGrader:
    """
    Level 2: Budget Pacing
    Score combines:
      • Smoothness penalty: penalises spending too fast in early hours.
      • Peak-hour survival: full bonus if ≥ 20% budget survives to hour 18.
      • Revenue factor: how much revenue was earned relative to target.
    """

    _INITIAL_BUDGET = 50.0
    _PEAK_START     = 18   # hours 18–22 are "peak"
    _BUDGET_RESERVE = 0.20 # must retain ≥ 20% when peak starts

    def __init__(self) -> None:
        self._burn_rates: List[float] = []
        self._survived_peak = False
        self._total_revenue = 0.0

    def record_step(self, step: int, spent_this_step: float,
                    remaining_budget: float, revenue: float) -> None:
        ideal_spend_per_step = self._INITIAL_BUDGET / 24.0
        self._burn_rates.append(spent_this_step)
        self._total_revenue  = revenue
        if step == self._PEAK_START:
            reserve_pct = remaining_budget / self._INITIAL_BUDGET
            self._survived_peak = reserve_pct >= self._BUDGET_RESERVE

    def episode_score(self) -> float:
        # Smoothness: penalise variance in hourly spend
        if not self._burn_rates:
            return 0.0
        ideal   = self._INITIAL_BUDGET / 24.0
        devs    = [abs(b - ideal) / ideal for b in self._burn_rates]
        smooth  = float(np.clip(1.0 - np.mean(devs), 0.0, 1.0))

        # Survival bonus
        survival = 1.0 if self._survived_peak else 0.0

        # Revenue factor (target: $30)
        rev_factor = float(np.clip(self._total_revenue / 30.0, 0.0, 1.0))

        score = 0.3 * smooth + 0.3 * survival + 0.4 * rev_factor
        return round(score, 4)

    def reset(self) -> None:
        self._burn_rates.clear()
        self._survived_peak  = False
        self._total_revenue  = 0.0


# ---------------------------------------------------------------------------

class HardAssemblyGrader:
    """
    Level 3: Multi-modal Viral Assembly
    Score = mean over winning steps of:
        max(0, CosineSim(caption_or_headline, viral_token))

    Uses the same SentenceTransformer for consistency; can be swapped for
    CLIP ViT-B/32 when image embeddings are available.
    """

    def __init__(self, semantic_model: SentenceTransformer) -> None:
        self._model  = semantic_model
        self._scores: List[float] = []

    def score_step(self, caption: str, viral_trend: str) -> float:
        cap_emb   = self._model.encode(caption,      convert_to_tensor=True)
        trend_emb = self._model.encode(viral_trend,  convert_to_tensor=True)
        cos       = float(util.cos_sim(cap_emb, trend_emb).item())
        step_score = float(np.clip(cos, 0.0, 1.0))
        self._scores.append(step_score)
        return step_score

    def episode_score(self) -> float:
        if not self._scores:
            return 0.0
        return round(float(np.mean(self._scores)), 4)

    def reset(self) -> None:
        self._scores.clear()


# ===========================================================================
# Main Environment
# ===========================================================================

class OpenEnvAuctioneer:
    """
    Gym-style RL environment.

    Interface
    ---------
    obs          = env.reset()
    obs, r, done, info = env.step(action)

    Reward signal
    -------------
    If auction won:   r = (adjusted_ctr × CONVERSION_VALUE) − clearing_price
    If auction lost:  r = −0.10  (missed opportunity penalty)
    Pacing penalty (medium_pacing only): −1.0 if spending too fast.
    """

    CONVERSION_VALUE = 15.0   # USD value of one conversion
    FATIGUE_STEP     = 0.015  # fatigue added per won auction
    FATIGUE_DECAY    = 0.05   # fatigue removed per missed auction

    # Per-task initial budgets
    _BUDGETS = {
        "easy_headline":  100.0,
        "medium_pacing":   50.0,
        "hard_assembly":  100.0,
    }

    def __init__(self, task_id: str = "easy_headline") -> None:
        assert task_id in self._BUDGETS, \
            f"Unknown task_id '{task_id}'. Valid: {list(self._BUDGETS)}"
        self.task_id   = task_id
        self.max_steps = 24

        # ── Dataset layers ────────────────────────────────────────────────
        self._market     = MarketCalibrator();  self._market.calibrate()
        self._ctr_cal    = CTRCalibrator();     self._ctr_cal.calibrate()
        self._pool       = CreativePool();      self._pool.load()
        self._personas   = PersonaBank();       self._personas.load()

        # ── Simulation layers ──────────────────────────────────────────────
        self._user_sim   = UserSimulator(self._personas, self._pool, self._ctr_cal)

        # ── Graders ────────────────────────────────────────────────────────
        self._easy_grader   = EasyHeadlineGrader(self._user_sim, self._pool)
        self._medium_grader = MediumPacingGrader()
        self._hard_grader   = HardAssemblyGrader(self._user_sim._smodel)

        # ── Episode state (populated by reset) ────────────────────────────
        self._step            = 0
        self._budget_init     = 0.0
        self._remaining       = 0.0
        self._total_revenue   = 0.0
        self._fatigue         = 0.0
        self._ads_shown       = 0
        self._last_ctr        = 0.0
        self._context         = ""
        self._trend           = ""
        self._prev_remaining  = 0.0

        self._contexts = ["Fitness", "Tech", "Fashion", "Gaming"]
        self._trends   = ["Quiet Luxury", "Eco-Friendly", "Cyberpunk", "Minimalism"]

    # -----------------------------------------------------------------------
    def state(self) -> Observation:
        return self._make_obs()

    # -----------------------------------------------------------------------
    def reset(self) -> Observation:
        self._step           = 0
        self._budget_init    = self._BUDGETS[self.task_id]
        self._remaining      = self._budget_init
        self._prev_remaining = self._budget_init
        self._total_revenue  = 0.0
        self._fatigue        = 0.0
        self._ads_shown      = 0
        self._last_ctr       = 0.0

        self._easy_grader.reset()
        self._medium_grader.reset()
        self._hard_grader.reset()

        self._update_context()
        return self._make_obs()

    # -----------------------------------------------------------------------
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        # Guard: episode already over
        if self._remaining <= 0:
            return self._make_obs(), Reward(value=0.0), True, self._make_info(0.0, 0.0, False, 0.0, 0.0, 0.0)

        clearing_price = self._market.sample_clearing_price(self._step)
        step_reward    = 0.0
        auction_won    = False
        raw_ctr        = 0.0
        adjusted_ctr   = 0.0
        spent          = 0.0

        can_bid = (action.bid_price >= clearing_price and
                   self._remaining  >= clearing_price)

        if can_bid:
            auction_won          = True
            spent                = clearing_price
            self._remaining     -= clearing_price
            self._ads_shown     += 1

            raw_ctr, adjusted_ctr = self._user_sim.compute_ctr(
                action, self._context, self._trend, self._fatigue)
            self._last_ctr = adjusted_ctr

            revenue       = adjusted_ctr * self.CONVERSION_VALUE
            self._total_revenue += revenue
            step_reward   = revenue - clearing_price
            self._fatigue = min(1.0, self._fatigue + self.FATIGUE_STEP)
        else:
            step_reward   = -0.10
            self._fatigue = max(0.0, self._fatigue - self.FATIGUE_DECAY)

        # ── Pacing penalty (medium_pacing only) ─────────────────────────
        if self.task_id == "medium_pacing":
            ideal_burn = self._budget_init / 24.0
            actual_burn = (self._budget_init - self._remaining) / max(1, self._step + 1)
            if actual_burn > ideal_burn * 1.5:
                step_reward -= 1.0

        # ── Update graders ───────────────────────────────────────────────
        if self.task_id == "easy_headline" and auction_won:
            self._easy_grader.score_step(
                action, self._context, self._trend, adjusted_ctr)

        if self.task_id == "medium_pacing":
            self._medium_grader.record_step(
                self._step, spent, self._remaining, self._total_revenue)

        if self.task_id == "hard_assembly" and auction_won:
            caption_text = (action.generated_caption
                            if action.generated_caption
                            else self._pool.headlines.get(action.headline_id, ""))
            clip_score = self._hard_grader.score_step(caption_text, self._trend)
        else:
            clip_score = 0.0

        # ── Advance time ─────────────────────────────────────────────────
        self._prev_remaining  = self._remaining
        self._step           += 1
        done = (self._step >= self.max_steps) or (self._remaining <= 0)
        if not done:
            self._update_context()

        info = self._make_info(
            clearing_price, raw_ctr, auction_won, adjusted_ctr,
            clip_score, spent)
        return self._make_obs(), Reward(value=round(step_reward, 2)), done, info

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _update_context(self) -> None:
        self._context = self._contexts[self._step % len(self._contexts)]
        self._trend   = self._trends[self._step   % len(self._trends)]

    def _make_obs(self) -> Observation:
        return Observation(
            hour_of_day           = min(self._step, 23),
            remaining_budget      = round(self._remaining, 2),
            spent_so_far          = round(self._budget_init - self._remaining, 2),
            current_context       = self._context,
            viral_trend           = self._trend,
            market_pressure       = self._market.market_pressure(
                                        min(self._step, 23)),
            ads_shown_this_session= self._ads_shown,
            fatigue_level         = round(self._fatigue, 3),
            last_ctr              = self._last_ctr,
            cumulative_revenue    = round(self._total_revenue, 2),
        )

    def _make_info(self, clearing_price: float, raw_ctr: float,
                   auction_won: bool, adjusted_ctr: float,
                   clip_score: float, spent: float) -> Info:
        # ── Compute final task score ─────────────────────────────────────
        if self.task_id == "easy_headline":
            task_score         = self._easy_grader.episode_score()
            headline_score     = task_score
            pacing_score       = 0.0
            clip_sim_score     = 0.0

        elif self.task_id == "medium_pacing":
            task_score         = self._medium_grader.episode_score()
            headline_score     = 0.0
            pacing_score       = task_score
            clip_sim_score     = 0.0

        else:  # hard_assembly
            clip_sim_score     = self._hard_grader.episode_score()
            revenue_factor     = float(np.clip(self._total_revenue / 45.0, 0.0, 1.0))
            # Blend: 60% viral alignment + 40% revenue
            task_score         = round(0.6 * clip_sim_score + 0.4 * revenue_factor, 4)
            headline_score     = 0.0
            pacing_score       = 0.0

        return Info(
            task_id                 = self.task_id,
            current_step            = self._step,
            total_revenue           = round(self._total_revenue, 2),
            clearing_price          = clearing_price,
            auction_won             = auction_won,
            raw_ctr                 = raw_ctr,
            adjusted_ctr            = adjusted_ctr,
            task_score              = task_score,
            headline_alignment_score= headline_score,
            pacing_score            = pacing_score,
            clip_similarity_score   = clip_sim_score,
        )