"""
app.py — FastAPI server for the OpenEnv Creative Auctioneer environment.

Runs inside the Docker container and exposes HTTP endpoints:
  POST /reset?task_id=easy_headline  → initialize environment, return observation
  POST /step                         → take an action, return (obs, reward, done, info)
  GET  /state                        → current environment state snapshot
  GET  /health                       → liveness check
"""

from typing import List, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from environment import OpenEnvAuctioneer
from models import Action

app = FastAPI(title="OpenEnv Creative Auctioneer", version="0.4.0")

# ---------------------------------------------------------------------------
# Global environment instance (one per container)
# ---------------------------------------------------------------------------
_env: Optional[OpenEnvAuctioneer] = None


@app.on_event("startup")
def preload_env():
    global _env
    print("[Startup] Preloading OpenEnv ML models and datasets...")
    # Initialize the default task so downloading and caching happens immediately
    _env = OpenEnvAuctioneer(task_id="easy_headline")
    _env.reset()
    print("[Startup] Caching complete!")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    bid_price: float
    headline_id: int
    creative_id: int
    generated_caption: Optional[str] = None
    generated_hashtags: Optional[List[str]] = None


class ResetResponse(BaseModel):
    observation: dict
    done: bool


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset_env(task_id: str = Query("easy_headline")):
    global _env
    # Avoid re-initializing if task hasn't changed. This saves time!
    if _env is None or getattr(_env, "task_id", None) != task_id:
        _env = OpenEnvAuctioneer(task_id=task_id)
    obs = _env.reset()
    return ResetResponse(observation=obs.model_dump(), done=False)


@app.post("/step", response_model=StepResponse)
def step_env(action: StepRequest):
    global _env
    if _env is None:
        raise ValueError("Environment not initialised. Call /reset first.")

    act = Action(
        bid_price=action.bid_price,
        headline_id=action.headline_id,
        creative_id=action.creative_id,
        generated_caption=action.generated_caption,
        generated_hashtags=action.generated_hashtags,
    )
    obs, reward, done, info = _env.step(act)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.value,
        done=done,
        info=info.model_dump(),
    )


@app.get("/state")
def get_state():
    global _env
    if _env is None:
        return {"error": "Environment not initialised. Call /reset first."}
    return _env.state()
