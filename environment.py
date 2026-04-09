"""
environment.py — OpenEnv Creative Auctioneer  v0.3
===================================================

Architecture
------------
┌───────────────────────────────────────────────────────────┐
│  OpenEnvAuctioneer (Gym-style environment)                 │
│                                                             │
│  ┌──────────────────┐   ┌───────────────────────────────┐  │
│  │  Market Engine    │   │   User Simulator               │  │
│  │  (Statistical)    │   │   (Semantic / LLM)             │  │
│  │                   │   │                               │  │
│  │  iPinYou RTB logs │   │  SentenceTransformer          │  │
│  │  → Lognormal per  │   │  all-MiniLM-L6-v2             │  │
│  │    hour bucket    │   │  + optional Llama-3-8B        │  │
│  └──────────────────┘   └───────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  MIND Dataset Layer  (Microsoft News Dataset)          │  │
│  │                                                        │  │
│  │  behaviours.tsv  →  CTRCalibrator                      │  │
│  │    parse impression logs (click=1/skip=0) per          │  │
│  │    news category → baseline CTR lookup table           │  │
│  │                                                        │  │
│  │  news.tsv        →  MINDCreativePool                   │  │
│  │    news_id, category, subcategory, title, abstract     │  │
│  │    → context pool + headlines catalog for agent        │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Grader (task-specific, deterministic 0.0–1.0)         │  │
│  │   Level 1: easy_headline  → headline CTR lookup        │  │
│  │   Level 2: medium_pacing  → pacing + survival          │  │
│  │   Level 3: hard_assembly  → CLIP cosine sim            │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘

Dataset Calibration  (lazy-loaded at first reset)
--------------------------------------------------
• MIND behaviours.tsv  →  per-category baseline CTR lookup table
• MIND news.tsv        →  creative pool (headlines + categories)
• iPinYou RTB logs     →  per-hour Lognormal (μ, σ) for competitor bids
• Vogue Dialogue       →  user persona bank for User Simulator prompts

Remote hosting options for MIND (see MIND_LOADER below):
  Option A — Hugging Face Hub dataset streaming (no local disk needed)
  Option B — Azure Blob Storage (SAS URL in env var MIND_AZURE_SAS_URL)
  Option C — Local mount via DATA_DIR env var  (default: /data)

All datasets fall back gracefully so the env runs without any downloads.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import pathlib
import random
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

from models import Action, Info, Observation, Reward

# ---------------------------------------------------------------------------
# Paths — datasets may be local mounts OR fetched remotely (see MINDLoader)
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent.resolve() / "Datasets"

# iPinYou (market calibration — unchanged)
IPINYOU_PATH = DATA_DIR

# MIND (Microsoft News Dataset) — replaces Criteo + Pitt
MIND_PATH        = DATA_DIR / "MINDlarge_train"

MIND_BEHAVIOURS  = MIND_PATH / "behaviors.tsv"   # impression click log
MIND_NEWS        = MIND_PATH / "news.tsv"          # news article metadata

# Vogue Dialogue (persona bank — unchanged)
VOGUE_PATH = DATA_DIR / "vogue_dialogue"


# ===========================================================================
# MIND Remote Loader
# ===========================================================================
# ┌──────────────────────────────────────────────────────────────────────┐
# │  THREE HOSTING OPTIONS — choose one via env vars                      │
# │                                                                        │
# │  OPTION A: Hugging Face Hub streaming (recommended — zero local disk) │
# │    Set: MIND_SOURCE=huggingface                                        │
# │    pip install datasets                                                │
# │    The 'mind' dataset on HF Hub is streamed record-by-record;          │
# │    we buffer only what we need (≈5 MB) then discard.                  │
# │                                                                        │
# │  OPTION B: Azure Blob Storage (SAS URL)                                │
# │    Set: MIND_SOURCE=azure                                              │
# │         MIND_AZURE_SAS_URL=https://<account>.blob.core.windows.net/   │
# │                             <container>/MINDsmall_train.zip?<SAS>      │
# │    The zip is downloaded once, extracted to DATA_DIR/mind/, cached.   │
# │    Recommended storage: Azure Blob (hot tier) ~100 MB for small split  │
# │                                                                        │
# │  OPTION C: Local mount (default — DATA_DIR=/data)                     │
# │    Set: MIND_SOURCE=local  (or leave unset)                            │
# │    Mount the pre-extracted MIND small/large split as a Docker volume:  │
# │      docker run -v /host/mind:/data/mind ...                           │
# │    Download from: https://msnews.github.io/                            │
# └──────────────────────────────────────────────────────────────────────┘

class MINDLoader:
    """
    Handles fetching/caching of MIND behaviours.tsv and news.tsv
    from whichever source is configured via env vars.

    After load(), self.behaviours_path and self.news_path point to
    valid local files regardless of which source was used.
    """

    _SOURCE       = os.environ.get("MIND_SOURCE", "local").lower()
    _AZURE_SAS    = os.environ.get("MIND_AZURE_SAS_URL", "")
    # Public MIND-small demo files (Hugging Face raw — no auth needed)
    _HF_BEHAVIOURS_URL = (
        "https://huggingface.co/datasets/telord/mind-news-recommendation"
        "/resolve/main/MINDsmall_train/behaviors.tsv"
    )
    _HF_NEWS_URL = (
        "https://huggingface.co/datasets/telord/mind-news-recommendation"
        "/resolve/main/MINDsmall_train/news.tsv"
    )

    def __init__(self) -> None:
        self.behaviours_path: Optional[pathlib.Path] = None
        self.news_path:       Optional[pathlib.Path] = None
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Returns True if files are ready, False if all sources failed."""
        if self._loaded:
            return True

        # ── Option C: local mount (fastest, no network) ────────────────
        if MIND_BEHAVIOURS.exists() and MIND_NEWS.exists():
            self.behaviours_path = MIND_BEHAVIOURS
            self.news_path       = MIND_NEWS
            print("[MINDLoader] Using local MIND files ✓")
            self._loaded = True
            return True

        # ── Option B: Azure Blob SAS URL ───────────────────────────────
        if self._SOURCE == "azure" and self._AZURE_SAS:
            return self._load_azure()

        # ── Option A: Hugging Face direct download ─────────────────────
        if self._SOURCE in ("huggingface", "hf", "auto"):
            return self._load_huggingface()

        # Auto-fallback: try HF if local not found and no Azure configured
        if self._SOURCE == "local":
            print("[MINDLoader] Local files not found and MIND_SOURCE=local. "
                  "Set MIND_SOURCE=huggingface to auto-download.")
            return False

        return False

    # ------------------------------------------------------------------
    def _load_huggingface(self) -> bool:
        """Download behaviours.tsv and news.tsv directly from HF Hub."""
        MIND_PATH.mkdir(parents=True, exist_ok=True)
        b_path = MIND_PATH / "behaviours.tsv"
        n_path = MIND_PATH / "news.tsv"
        try:
            if not b_path.exists():
                print(f"[MINDLoader] Downloading behaviours.tsv from HuggingFace…")
                urllib.request.urlretrieve(self._HF_BEHAVIOURS_URL, b_path)
            if not n_path.exists():
                print(f"[MINDLoader] Downloading news.tsv from HuggingFace…")
                urllib.request.urlretrieve(self._HF_NEWS_URL, n_path)
            self.behaviours_path = b_path
            self.news_path       = n_path
            self._loaded = True
            print("[MINDLoader] MIND files ready from HuggingFace ✓")
            return True
        except Exception as exc:
            print(f"[MINDLoader] HuggingFace download failed: {exc}")
            return False

    # ------------------------------------------------------------------
    def _load_azure(self) -> bool:
        """Download MIND zip from Azure Blob SAS URL, extract to MIND_PATH."""
        MIND_PATH.mkdir(parents=True, exist_ok=True)
        zip_path = MIND_PATH / "mind.zip"
        try:
            if not zip_path.exists():
                print(f"[MINDLoader] Downloading MIND zip from Azure…")
                urllib.request.urlretrieve(self._AZURE_SAS, zip_path)
            print("[MINDLoader] Extracting zip…")
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    fname = pathlib.Path(member).name
                    if fname in ("behaviors.tsv", "behaviours.tsv"):
                        zf.extract(member, MIND_PATH)
                        (MIND_PATH / member).rename(MIND_PATH / "behaviours.tsv")
                    elif fname == "news.tsv":
                        zf.extract(member, MIND_PATH)
                        (MIND_PATH / member).rename(MIND_PATH / "news.tsv")
            self.behaviours_path = MIND_PATH / "behaviours.tsv"
            self.news_path       = MIND_PATH / "news.tsv"
            self._loaded = True
            print("[MINDLoader] MIND files ready from Azure ✓")
            return True
        except Exception as exc:
            print(f"[MINDLoader] Azure download failed: {exc}")
            return False


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


# ===========================================================================
# CTR Calibrator — MIND behaviours.tsv
# ===========================================================================
#
# behaviours.tsv column layout (tab-separated, no header):
#   0  impression_id   (int)
#   1  user_id         (str — NOT used; privacy-native)
#   2  time            (str "MM/DD/YYYY H:MM:SS AM/PM")
#   3  history         (space-separated news_ids previously clicked — ignored)
#   4  impressions     (space-separated "news_id-label" pairs, label 1=click 0=skip)
#
# We join impressions back to news.tsv via news_id to get the category,
# then accumulate clicks and totals per MIND category.
# Result: category → baseline_CTR  (used by UserSimulator)
#
# MIND top-level categories (news.tsv col 1) mapped to our 4 env contexts:
#   sports, health      → "Fitness"
#   technology          → "Tech"
#   lifestyle, autos    → "Fashion"
#   entertainment, tv,
#   music, video,
#   movies, games,
#   travel              → "Gaming"  (engagement proxy)
#   news, finance,
#   weather, etc.       → fallback benchmark

class CTRCalibrator:
    """
    Parses MIND behaviours.tsv + news.tsv to build:
        context_category  → baseline_CTR   (coarse, 4 buckets)
        news_subcategory  → baseline_CTR   (fine-grained, exposed to UserSimulator)

    Falls back to published MIND CTR statistics if files are absent.
    Reference: Wu et al. (2020) "MIND: A Large-scale Dataset for
    News Recommendation", ACL 2020.
    """

    # MIND-derived fallback CTRs (Wu et al. 2020, MIND-small split)
    _BENCHMARK_CTR: Dict[str, float] = {
        "Fitness":  0.062,
        "Tech":     0.048,
        "Fashion":  0.055,
        "Gaming":   0.071,
    }

    # MIND top-level category → env context
    _CATEGORY_MAP: Dict[str, str] = {
        "sports":        "Fitness",
        "health":        "Fitness",
        "foodanddrink":  "Fitness",
        "technology":    "Tech",
        "lifestyle":     "Fashion",
        "autos":         "Fashion",
        "travel":        "Fashion",
        "entertainment": "Gaming",
        "tv":            "Gaming",
        "music":         "Gaming",
        "video":         "Gaming",
        "movies":        "Gaming",
        "games":         "Gaming",
        "kids":          "Gaming",
        # unmapped → None (excluded from coarse table)
        "news":          None,
        "finance":       None,
        "weather":       None,
        "middleeast":    None,
    }

    _CONTEXTS = ["Fitness", "Tech", "Fashion", "Gaming"]

    def __init__(self, loader: "MINDLoader") -> None:
        self._loader     = loader
        self._ctr_table: Dict[str, float] = {}
        self._subcat_ctr: Dict[str, float] = {}  # fine-grained subcategory CTRs

    # ------------------------------------------------------------------
    def calibrate(self) -> None:
        if not self._loader.load():
            print("[CTRCalibrator] MIND not available → benchmark CTR.")
            self._ctr_table = dict(self._BENCHMARK_CTR)
            return

        # ── Step 1: Build news_id → (category, subcategory) map ────────
        # news.tsv: news_id  category  subcategory  title  abstract  url  ...
        news_meta: Dict[str, Tuple[str, str]] = {}
        try:
            with open(self._loader.news_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    news_id  = parts[0].strip()
                    category = parts[1].strip().lower()
                    subcat   = parts[2].strip().lower()
                    news_meta[news_id] = (category, subcat)
        except Exception as exc:
            print(f"[CTRCalibrator] Error reading news.tsv: {exc} → benchmark.")
            self._ctr_table = dict(self._BENCHMARK_CTR)
            return

        # ── Step 2: Parse impressions, accumulate clicks per context ────
        ctx_clicks: Dict[str, int] = {c: 0 for c in self._CONTEXTS}
        ctx_total:  Dict[str, int] = {c: 0 for c in self._CONTEXTS}
        sub_clicks: Dict[str, int] = {}
        sub_total:  Dict[str, int] = {}

        try:
            with open(self._loader.behaviours_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    # col 4 = impressions: "N116694-1 N32154-0 ..."
                    if len(parts) < 5 or not parts[4].strip():
                        continue
                    for item in parts[4].split():
                        if "-" not in item:
                            continue
                        news_id, label_str = item.rsplit("-", 1)
                        try:
                            label = int(label_str)  # 1=click, 0=skip
                        except ValueError:
                            continue

                        meta = news_meta.get(news_id)
                        if meta is None:
                            continue
                        cat, subcat = meta

                        ctx = self._CATEGORY_MAP.get(cat)
                        if ctx is not None:
                            ctx_total[ctx]  += 1
                            ctx_clicks[ctx] += label

                        sub_total[subcat]  = sub_total.get(subcat, 0) + 1
                        sub_clicks[subcat] = sub_clicks.get(subcat, 0) + label

        except Exception as exc:
            print(f"[CTRCalibrator] Error reading behaviours.tsv: {exc} → benchmark.")
            self._ctr_table = dict(self._BENCHMARK_CTR)
            return

        # ── Step 3: Compute CTRs ────────────────────────────────────────
        for ctx in self._CONTEXTS:
            total = ctx_total[ctx]
            self._ctr_table[ctx] = (
                ctx_clicks[ctx] / total if total > 0
                else self._BENCHMARK_CTR[ctx]
            )
        for sub, total in sub_total.items():
            if total >= 20:  # only trust subcategories with enough samples
                self._subcat_ctr[sub] = sub_clicks[sub] / total

        print(f"[CTRCalibrator] MIND calibrated: "
              f"{sum(ctx_total.values()):,} impressions, "
              f"{len(self._subcat_ctr)} subcategories ✓")
        print(f"  Context CTRs: {self._ctr_table}")

    # ------------------------------------------------------------------
    def baseline_ctr(self, context: str) -> float:
        if not self._ctr_table:
            self.calibrate()
        return self._ctr_table.get(context, 0.05)

    def subcategory_ctr(self, subcat: str) -> Optional[float]:
        """Fine-grained CTR for a MIND subcategory; None if unknown."""
        return self._subcat_ctr.get(subcat.lower())


# ===========================================================================
# Creative Pool — MIND news.tsv
# ===========================================================================
#
# news.tsv column layout (tab-separated, no header):
#   0  news_id
#   1  category       (e.g. "sports", "technology")
#   2  subcategory    (e.g. "golf", "gadgets")
#   3  title          ← used directly as ad headline text
#   4  abstract       ← used as creative description
#   5  url            (ignored)
#   6  title_entities (JSON — ignored)
#   7  abstract_entities (JSON — ignored)
#
# We sample 1 article per env context as the catalog slot.
# This means the 6-slot catalog is seeded from real MIND article titles
# and abstracts, giving the agent headlines grounded in real news language.

class MINDCreativePool:
    """
    Builds the 6-slot headline+creative catalog from MIND news.tsv.

    Slot assignment:
      0 → Fitness   (sports / health subcategory, best CTR article)
      1 → Tech      (technology subcategory)
      2 → Fashion   (lifestyle subcategory)
      3 → Gaming    (entertainment subcategory)
      4 → Fitness   (foodanddrink — eco/wellness overlap)
      5 → Fashion   (autos/travel — aspirational lifestyle)

    Falls back to hard-coded catalog if MIND is unavailable.
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
    # slot → target MIND subcategory keyword (we pick the best-CTR article matching this)
    _SLOT_SUBCAT: Dict[int, str] = {
        0: "sports",
        1: "technology",
        2: "lifestyle",
        3: "entertainment",
        4: "foodanddrink",
        5: "travel",
    }
    _SLOT_CONTEXT: Dict[int, str] = {
        0: "Fitness",
        1: "Tech",
        2: "Fashion",
        3: "Gaming",
        4: "Fitness",
        5: "Fashion",
    }

    def __init__(self, loader: "MINDLoader",
                 ctr_cal: CTRCalibrator) -> None:
        self._loader  = loader
        self._ctr_cal = ctr_cal
        self.headlines:        Dict[int, str] = dict(self._FALLBACK_HEADLINES)
        self.creatives:        Dict[int, str] = dict(self._FALLBACK_CREATIVES)
        self.context_affinity: Dict[int, str] = dict(self._SLOT_CONTEXT)
        # Expose subcategory per slot for UserSimulator fine-graining
        self.slot_subcat:      Dict[int, str] = dict(self._SLOT_SUBCAT)

    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self._loader.load() or self._loader.news_path is None:
            print("[MINDCreativePool] MIND not available → fallback catalog.")
            return

        # Collect candidate articles per target subcategory
        candidates: Dict[str, List[Tuple[str, str]]] = {
            s: [] for s in self._SLOT_SUBCAT.values()
        }

        try:
            with open(self._loader.news_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        continue
                    subcat   = parts[2].strip().lower()
                    title    = parts[3].strip()
                    abstract = parts[4].strip()
                    if not title:
                        continue
                    for target in candidates:
                        if target in subcat or subcat in target:
                            candidates[target].append((title, abstract or title))
        except Exception as exc:
            print(f"[MINDCreativePool] Error reading news.tsv: {exc} → fallback.")
            return

        # Pick one article per slot (random sample from candidates)
        for slot_id, subcat_key in self._SLOT_SUBCAT.items():
            pool = candidates.get(subcat_key, [])
            if pool:
                title, abstract = random.choice(pool)
                # Truncate to reasonable ad lengths
                self.headlines[slot_id] = title[:80]
                self.creatives[slot_id] = abstract[:160]
            # else keep fallback

        n_loaded = sum(1 for s, p in candidates.items() if p)
        print(f"[MINDCreativePool] Loaded headlines from MIND news.tsv "
              f"({n_loaded}/6 slots filled from real articles) ✓")

    # ------------------------------------------------------------------
    def best_headline_for_context(self, context: str) -> int:
        """Return the slot_id with the highest CTR affinity for this context."""
        # Among slots matching this context, pick the one whose subcategory
        # has the highest empirical CTR from behaviours.tsv
        best_id    = 0
        best_score = -1.0
        for slot_id, ctx in self.context_affinity.items():
            if ctx != context:
                continue
            sub  = self.slot_subcat.get(slot_id, "")
            ctr  = self._ctr_cal.subcategory_ctr(sub)
            if ctr is None:
                ctr = self._ctr_cal.baseline_ctr(ctx)
            if ctr > best_score:
                best_score = ctr
                best_id    = slot_id
        return best_id

# Alias so the rest of the file uses a consistent name
CreativePool = MINDCreativePool


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

#easy_headline
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

    def score_step(self, caption: str, viral_trend: str, image_desc: str) -> float:
        cap_emb   = self._model.encode(caption,      convert_to_tensor=True)
        trend_emb = self._model.encode(viral_trend,  convert_to_tensor=True)
        img_emb   = self._model.encode(image_desc,   convert_to_tensor=True)
        
        cos_trend = float(np.clip(util.cos_sim(cap_emb, trend_emb).item(), 0.0, 1.0))
        cos_img   = float(np.clip(util.cos_sim(cap_emb, img_emb).item(), 0.0, 1.0))
        
        step_score = (0.35 * cos_trend + 0.20 * cos_img) / 0.55
        self._scores.append(step_score)
        return round(step_score, 4)

    def episode_score(self) -> float:
        if not self._scores:
            return 0.0
        return round(float(np.mean(self._scores)), 4)

    def reset(self) -> None:
        self._scores.clear()


# ---------------------------------------------------------------------------

class HardSequencingGrader:
    """
    Level 4: Cross-Context Campaign Sequencing

    Tracks per-step data (clearing prices, CTRs, contexts, costs) and at
    episode end runs a DP oracle to find the optimal bid/skip sequence.

    Score = min(1.0, (agent_conversions / oracle_conversions) x diversity_mult)
    """

    CARRYOVER_BOOSTS = [0.15, 0.10, 0.05]   # t+1, t+2, t+3
    DIVERSITY_THRESHOLD = 3
    DIVERSITY_BONUS = 1.20

    def __init__(self) -> None:
        self._step_log: List[dict] = []
        self._contexts_won: set = set()

    def record_step(self, step: int, context: str, clearing_price: float,
                    base_ctr: float, auction_won: bool, cost: float,
                    conversion_value: float) -> None:
        self._step_log.append({
            "step": step, "context": context,
            "clearing_price": clearing_price,
            "base_ctr": base_ctr,
            "auction_won": auction_won,
            "cost": cost,
            "conversion_value": conversion_value,
        })
        if auction_won:
            self._contexts_won.add(context)

    # ------------------------------------------------------------------
    def _diversity_multiplier(self) -> float:
        return (self.DIVERSITY_BONUS
                if len(self._contexts_won) >= self.DIVERSITY_THRESHOLD
                else 1.0)

    # ------------------------------------------------------------------
    def _agent_conversions(self) -> float:
        """Total weighted conversions the agent actually achieved."""
        total = 0.0
        for idx, s in enumerate(self._step_log):
            if not s["auction_won"]:
                continue
            boost = 0.0
            for j, bv in enumerate(self.CARRYOVER_BOOSTS):
                prev = idx - 1 - j
                if prev >= 0 and self._step_log[prev]["auction_won"]:
                    boost += bv
            eff_ctr = s["base_ctr"] * (1.0 + boost)
            total += eff_ctr * s["conversion_value"]
        return total

    # ------------------------------------------------------------------
    def _oracle_conversions(self, budget: float) -> float:
        """
        DP over N steps.
        State = (step_index, budget_units, carry_state).
        carry_state is a 3-bit int encoding wins at t-1, t-2, t-3.
        """
        steps = self._step_log
        n = len(steps)
        if n == 0:
            return 0.0

        RES = 0.10
        budget_units = int(budget / RES) + 1

        # cur maps (budget_units, carry_state_3bit) -> best cumulative value
        cur: Dict[tuple, float] = {(budget_units, 0): 0.0}

        for i in range(n):
            s = steps[i]
            cp = s["clearing_price"]
            base = s["base_ctr"]
            cv = s["conversion_value"]
            cost_u = max(1, int(round(cp / RES)))

            nxt: Dict[tuple, float] = {}

            for (bu, cs), val in cur.items():
                w1 = (cs >> 2) & 1
                w2 = (cs >> 1) & 1
                w3 = cs & 1
                boost = 0.15 * w1 + 0.10 * w2 + 0.05 * w3

                # Option A: skip
                ns_skip = ((w1 << 1) | w2) & 0b111
                key_s = (bu, ns_skip)
                if key_s not in nxt or val > nxt[key_s]:
                    nxt[key_s] = val

                # Option B: bid (if affordable)
                if bu >= cost_u:
                    eff = base * (1.0 + boost)
                    nv = val + eff * cv
                    nb = bu - cost_u
                    ns_bid = ((1 << 2) | (w1 << 1) | w2) & 0b111
                    key_b = (nb, ns_bid)
                    if key_b not in nxt or nv > nxt[key_b]:
                        nxt[key_b] = nv

            cur = nxt

        return max(cur.values()) if cur else 0.0

    # ------------------------------------------------------------------
    def episode_score(self, initial_budget: float) -> float:
        oracle = self._oracle_conversions(initial_budget)
        if oracle <= 0:
            return 0.0
        agent = self._agent_conversions()
        div = self._diversity_multiplier()
        return min(1.0, round((agent / oracle) * div, 4))

    def reset(self) -> None:
        self._step_log.clear()
        self._contexts_won.clear()


# ===========================================================================
# Main Environment
# ===========================================================================

import urllib.request
    
class ViralHashtagScraper:
    """
    Pulls live hashtags from Reddit /r/popular and uses fallback static seeds
    per context/trend if the network fails.
    """
    _SEEDS = {
        "Fitness": ["#workout", "#fitlife", "#health", "#gym", "#motivation"],
        "Tech": ["#tech", "#gadgets", "#innovation", "#future", "#software"],
        "Fashion": ["#style", "#ootd", "#fashion", "#beauty", "#trendy"],
        "Gaming": ["#esports", "#gaming", "#gamer", "#streamer", "#play"],
    }
    
    def scrape(self, context: str, trend: str) -> list[str]:
        tags = list(self._SEEDS.get(context, ["#viral"]))
        tags.extend([f"#{trend.replace(' ', '').lower()}"])
        try:
            req = urllib.request.Request("https://www.reddit.com/r/popular/hot.json?limit=10", headers={'User-Agent': 'OpenEnvAuctioneer/1.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                for child in data.get("data", {}).get("children", []):
                    title = child["data"].get("title", "")
                    for w in title.split():
                        if len(w) > 5 and w.isalpha():
                            tags.append(f"#{w.lower()}")
        except Exception:
            pass
        
        tags = list(set(tags))
        random.shuffle(tags)
        return tags[:10]

class AdCaptionDataset:
    """
    Loads MS-COCO Captions from HuggingFace dataset 'shunk031/MSCOCO'.
    Buckets images by context using keyword matching.
    """
    _CONTEXT_KEYWORDS = {
        "Fitness": ["run", "sport", "bike", "tennis", "baseball", "surf", "snowboard", "ski", "gym", "jump"],
        "Tech": ["computer", "laptop", "phone", "tv", "remote", "screen", "keyboard"],
        "Fashion": ["suit", "dress", "shirt", "tie", "bag", "umbrella", "shoe"],
        "Gaming": ["game", "play", "controller", "tv", "video"],
    }
    
    def __init__(self):
        self._buckets = {ctx: [] for ctx in self._CONTEXT_KEYWORDS}
        self._fallback = {
            "Fitness": [("A runner in the park.", "Start your fitness journey today.")],
            "Tech": [("A modern laptop on a desk.", "Upgrade your setup with the latest tech.")],
            "Fashion": [("A model wearing a stylish dress.", "Elevate your wardrobe.")],
            "Gaming": [("A person holding a game controller.", "Level up your experience.")],
        }

    def load(self):
        try:
            import json, os
            from collections import defaultdict
            path = "Datasets/Coco/coco128/coco128_captions.json"
            print(f"[AdCaptionDataset] Loading local COCO file: {path} ...")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}")
                
            with open(path, "r") as f:
                data = json.load(f)
                
            grouped = defaultdict(list)
            for ann in data.get("annotations", []):
                grouped[ann["image_id"]].append(ann["caption"])
                
            for img_id, texts in grouped.items():
                if not texts:
                    continue
                
                if len(texts) == 1:
                    t1, t2 = texts[0], texts[0]
                else:
                    t1, t2 = texts[0], texts[1]
                
                texts_lower = texts[0].lower()
                for ctx, kws in self._CONTEXT_KEYWORDS.items():
                    if any(kw in texts_lower for kw in kws):
                        self._buckets[ctx].append((t1, t2))
                        break
            
            stats = {k: len(v) for k, v in self._buckets.items()}
            print(f"[AdCaptionDataset] Dataset loaded. Buckets: {stats}")
        except Exception as exc:
            print(f"[AdCaptionDataset] Could not load dataset: {exc} -> using fallback.")
            self._buckets = self._fallback

    def sample(self, context: str) -> tuple[str, str]:
        pool = self._buckets.get(context, [])
        if not pool:
            pool = self._fallback.get(context, [("A standard image.", "A standard caption.")])
        return random.choice(pool)

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
        "easy_headline":   100.0,
        "medium_pacing":   50.0,
        "hard_assembly":   100.0,
        "hard_sequencing": 100.0,
    }

    def __init__(self, task_id: str = "easy_headline") -> None:
        assert task_id in self._BUDGETS, \
            f"Unknown task_id '{task_id}'. Valid: {list(self._BUDGETS)}"
        self.task_id   = task_id
        self.max_steps = 24

        # ── Dataset layers ────────────────────────────────────────────────
        self._mind       = MINDLoader()
        self._market     = MarketCalibrator();          self._market.calibrate()
        self._ctr_cal    = CTRCalibrator(self._mind);   self._ctr_cal.calibrate()
        self._pool       = CreativePool(self._mind, self._ctr_cal); self._pool.load()
        self._personas   = PersonaBank();               self._personas.load()
        self._ad_dataset = AdCaptionDataset();          self._ad_dataset.load()
        self._hashtag_scraper = ViralHashtagScraper()

        # ── Simulation layers ──────────────────────────────────────────────
        self._user_sim   = UserSimulator(self._personas, self._pool, self._ctr_cal)

        # ── Graders ────────────────────────────────────────────────────────
        self._easy_grader   = EasyHeadlineGrader(self._user_sim, self._pool)
        self._medium_grader = MediumPacingGrader()
        self._hard_grader   = HardAssemblyGrader(self._user_sim._smodel)
        self._seq_grader    = HardSequencingGrader()

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
        self._carryover_boost = 0.0
        self._carryover_schedule: Dict[int, float] = {}

        self._contexts = ["Fitness", "Tech", "Fashion", "Gaming"]
        self._trends   = ["Quiet Luxury", "Eco-Friendly", "Cyberpunk", "Minimalism"]

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
        self._seq_grader.reset()
        self._carryover_boost = 0.0
        self._carryover_schedule = {}

        self._update_context()
        return self._make_obs()

    # -----------------------------------------------------------------------
    def state(self) -> dict:
        """Return a serialisable snapshot of the current environment state."""
        return {
            "task_id":          self.task_id,
            "step":             self._step,
            "max_steps":        self.max_steps,
            "initial_budget":   self._budget_init,
            "remaining_budget": round(self._remaining, 2),
            "spent_so_far":     round(self._budget_init - self._remaining, 2),
            "total_revenue":    round(self._total_revenue, 2),
            "fatigue_level":    round(self._fatigue, 3),
            "ads_shown":        self._ads_shown,
            "current_context":  self._context,
            "current_trend":    self._trend,
            "last_ctr":         self._last_ctr,
            "carryover_boost":  round(self._carryover_boost, 4),
            "done":             self._step >= self.max_steps or self._remaining <= 0,
        }

    # -----------------------------------------------------------------------
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        # Guard: episode already over
        if self._remaining <= 0:
            return self._make_obs(), Reward(value=0.0), True, self._make_info(0.0, 0.0, False, 0.0, 0.0, 0.0)

        # ── Read carry-over boost for hard_sequencing ─────────────────
        if self.task_id == "hard_sequencing":
            self._carryover_boost = self._carryover_schedule.get(self._step, 0.0)

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

            # Apply carry-over boost for hard_sequencing
            if self.task_id == "hard_sequencing":
                effective_ctr = round(adjusted_ctr * (1.0 + self._carryover_boost), 4)
                self._last_ctr = effective_ctr
                revenue = effective_ctr * self.CONVERSION_VALUE
                # Schedule carry-over boosts for future steps
                for ci, bv in enumerate(HardSequencingGrader.CARRYOVER_BOOSTS):
                    fs = self._step + 1 + ci
                    if fs < 24:
                        self._carryover_schedule[fs] = (
                            self._carryover_schedule.get(fs, 0.0) + bv)
            else:
                self._last_ctr = adjusted_ctr
                revenue = adjusted_ctr * self.CONVERSION_VALUE

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
            clip_score = self._hard_grader.score_step(caption_text, self._trend, getattr(self, "_last_image_desc", ""))
        else:
            clip_score = 0.0

        if self.task_id == "hard_sequencing":
            self._seq_grader.record_step(
                step=self._step, context=self._context,
                clearing_price=clearing_price, base_ctr=adjusted_ctr,
                auction_won=auction_won, cost=spent,
                conversion_value=self.CONVERSION_VALUE)

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
        # Determine the news subcategory for the current catalog slot
        # (the slot that best matches the current context)
        best_slot = self._pool.best_headline_for_context(self._context)
        news_subcat = self._pool.slot_subcat.get(best_slot, "")

        obs_kwargs = {
            "hour_of_day":          min(self._step, 23),
            "remaining_budget":     round(self._remaining, 2),
            "spent_so_far":         round(self._budget_init - self._remaining, 2),
            "current_context":      self._context,
            "news_category":        news_subcat,
            "viral_trend":          self._trend,
            "market_pressure":      self._market.market_pressure(min(self._step, 23)),
            "ads_shown_this_session": self._ads_shown,
            "fatigue_level":        round(self._fatigue, 3),
            "carryover_boost":      round(min(self._carryover_boost, 0.30), 4),
            "last_ctr":             self._last_ctr,
            "cumulative_revenue":   round(self._total_revenue, 2),
        }
        
        if self.task_id == "hard_assembly":
            img_desc, base_cap = self._ad_dataset.sample(self._context)
            obs_kwargs["image_description"] = img_desc
            obs_kwargs["base_caption"] = base_cap
            obs_kwargs["live_hashtags"] = self._hashtag_scraper.scrape(self._context, self._trend)
            self._last_image_desc = img_desc
            
        return Observation(**obs_kwargs)

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

        elif self.task_id == "hard_assembly":
            clip_sim_score     = self._hard_grader.episode_score()
            revenue_factor     = float(np.clip(self._total_revenue / 45.0, 0.0, 1.0))
            # Blend: 60% viral alignment + 40% revenue
            task_score         = round(0.6 * clip_sim_score + 0.4 * revenue_factor, 4)
            headline_score     = 0.0
            pacing_score       = 0.0

        else:  # hard_sequencing
            seq_sc             = self._seq_grader.episode_score(self._budget_init)
            task_score         = seq_sc
            headline_score     = 0.0
            pacing_score       = 0.0
            clip_sim_score     = 0.0

        # Sequencing sub-scores (non-zero only for hard_sequencing)
        if self.task_id == "hard_sequencing":
            _seq_score = task_score
            _ctx_cov   = len(self._seq_grader._contexts_won)
            _div_mult  = self._seq_grader._diversity_multiplier()
        else:
            _seq_score = 0.0
            _ctx_cov   = 0
            _div_mult  = 1.0

        # Clamp task_score to strictly (0, 1) — validator rejects 0.0 and 1.0
        task_score = float(np.clip(task_score, 0.001, 0.999))

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
            sequencing_score        = _seq_score,
            contexts_covered        = _ctx_cov,
            diversity_multiplier    = _div_mult,
        )