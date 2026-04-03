"""
app.py — FastAPI server for the OpenEnv Creative Auctioneer environment.

Runs inside the Docker container and exposes HTTP endpoints:
  POST /reset?task_id=easy_headline  → initialize environment, return observation
  POST /step                         → take an action, return (obs, reward, done, info)
  GET  /state                        → current environment state snapshot
  GET  /health                       → liveness check
"""

from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from environment import OpenEnvAuctioneer
from models import Action

app = FastAPI(title="OpenEnv Creative Auctioneer", version="0.3.0")

# ---------------------------------------------------------------------------
# Global environment instance (one per container)
# ---------------------------------------------------------------------------
_env: Optional[OpenEnvAuctioneer] = None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    bid_price: float
    headline_id: int
    creative_id: int
    generated_caption: Optional[str] = None


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
