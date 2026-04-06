"""
inference.py — Baseline inference script for OpenEnv Creative Auctioneer
=========================================================================

Runs tasks using an LLM agent via the OpenAI-compatible API client.
Communicates with the environment running inside a Docker container.

Usage:
    HF_TOKEN=<key> LOCAL_IMAGE_NAME=openenv-auctioneer python inference.py

Environment Variables (MANDATORY):
    API_BASE_URL       LLM endpoint (default: HuggingFace router)
    MODEL_NAME         Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN           API key for the LLM service
    LOCAL_IMAGE_NAME   Docker image name for the environment container

Optional:
    AUCTIONEER_TASK    easy_headline | medium_pacing | hard_assembly | hard_sequencing | all
"""

import asyncio
import json
import os
import socket
import subprocess
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

from models import Action

# ── Configuration from env vars ──────────────────────────────────────────────
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("AUCTIONEER_TASK", "all")
BENCHMARK = "openenv-auctioneer"
MAX_STEPS = 24
TEMPERATURE = 0.3
MAX_TOKENS = 256

ALL_TASKS = ["easy_headline", "medium_pacing", "hard_assembly", "hard_sequencing"]

SUCCESS_THRESHOLDS: Dict[str, float] = {
    "easy_headline": 0.50,
    "medium_pacing": 0.45,
    "hard_assembly": 0.40,
    "hard_sequencing": 0.35,
}

CATALOG_CTX = (
    "Headlines: 0=Fitness, 1=Tech, 2=Fashion, 3=Gaming, 4=Eco/Fitness, 5=Luxury/Fashion\n"
    "Creatives: 0=Runner, 1=Microchip, 2=Model, 3=Keyboard, 4=Eco-pkg, 5=Gold-watch"
)

# ── Task-specific system prompts ─────────────────────────────────────────────
SYSTEM_PROMPTS: Dict[str, str] = {
    "easy_headline": textwrap.dedent("""\
        You are an AI ad account manager. Select the headline+creative with the
        best semantic match for the context and viral trend each hour.
        Context guide: Fitness→0/4, Tech→1, Fashion→2/5, Gaming→3.
        Keep bids modest (0.30–0.80). Never bid over $2."""),

    "medium_pacing": textwrap.dedent("""\
        You are an AI ad account manager focused on BUDGET PACING.
        Budget $50 for 24 hours (~$2.08/hr). MUST have >=20% ($10) at hour 18.
        Hours 0-17: bid 0.20-0.60. Hours 18-22: bid up to $1.80.
        If budget < $5 before hour 18, bid $0."""),

    "hard_assembly": textwrap.dedent("""\
        You are an AI Account Manager and Creative Director for the hard_assembly task.

        YOUR JOB each step:
        1. You receive a SOURCE AD CREATIVE: an image description + a base caption.
        2. You receive LIVE VIRAL HASHTAGS scraped from Google Trends / Reddit.
        3. You receive the current VIRAL TREND token (cultural keyword).
        4. You must ASSEMBLE a final ad by:
           (a) Selecting 2–4 hashtags from the live list that best match the trend.
           (b) Rewriting the base caption to weave those hashtags into natural, punchy
               ad copy — DO NOT just append hashtags at the end.  Blend them into prose.
           (c) Adding your own creative words (target 30–50% new vocabulary).
           (d) The final caption must stay coherent with the image description.

        GRADER weights (what earns you points):
           35% — Hashtag relevance:  chosen hashtags semantically match viral_trend
           35% — Caption-trend align: your caption text matches viral_trend vocabulary
           20% — Image coherence:    your caption stays faithful to the image
           10% — Novelty:             you added real creative words, not just copy-paste

        REWARD:  auction_base + (composite_score × $8 bonus per winning step)
        BUDGET:  $120 for 24 hours.  Bid $0.60–$1.50 per step."""),

    "hard_sequencing": textwrap.dedent("""\
        You focus on CROSS-CONTEXT CAMPAIGN SEQUENCING.
        Winning boosts CTR for 3 hrs (+15/+10/+5%). Cover >=3 contexts for 20% bonus.
        Skip expensive hours, bid aggressively in cheap ones.
        Use carryover_boost to time high-value bids. Keep >=15% budget for hrs 18-22."""),
}


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict


# ── Environment client (Docker container) ────────────────────────────────────
class AuctioneerEnvClient:
    """Async HTTP client for the containerised OpenEnv Auctioneer server."""

    def __init__(self, base_url: str, container_id: Optional[str] = None,
                 task_id: str = "easy_headline"):
        self.base_url = base_url.rstrip("/")
        self.container_id = container_id
        self.task_id = task_id
        self._client = httpx.AsyncClient(timeout=300.0)

    @classmethod
    async def from_docker_image(cls, image_name: str,
                                task_id: str = "easy_headline"):
        """Start env container and return a connected client."""
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        if os.getenv("NO_DOCKER"):
            proc = subprocess.Popen(["uvicorn", "app:app", "--port", str(port)])
            container_id = "local_uvicorn"
            base_url = f"http://localhost:{port}"
            inst = cls(base_url=base_url, container_id=container_id, task_id=task_id)
            inst.proc = proc
        else:
            container_id = subprocess.check_output([
                "docker", "run", "-d", "--rm",
                "-p", f"{port}:7860",
                image_name,
            ]).decode().strip()
            base_url = f"http://localhost:{port}"
            inst = cls(base_url=base_url, container_id=container_id, task_id=task_id)

        # Wait for the server to become ready
        for _ in range(90):
            try:
                r = await inst._client.get(f"{base_url}/health")
                if r.status_code == 200:
                    return inst
            except Exception:
                pass
            await asyncio.sleep(1.0)
        raise RuntimeError(f"Container {container_id} did not become ready")

    async def reset(self) -> StepResult:
        r = await self._client.post(
            f"{self.base_url}/reset", params={"task_id": self.task_id})
        r.raise_for_status()
        d = r.json()
        return StepResult(observation=d["observation"], reward=0.0,
                          done=d.get("done", False), info={})

    async def step(self, action: Action) -> StepResult:
        r = await self._client.post(
            f"{self.base_url}/step", json=action.model_dump())
        r.raise_for_status()
        d = r.json()
        return StepResult(observation=d["observation"], reward=d["reward"],
                          done=d["done"], info=d.get("info", {}))

    async def close(self):
        await self._client.aclose()
        if self.container_id:
            if getattr(self, "proc", None):
                self.proc.terminate()
            else:
                subprocess.run(["docker", "stop", self.container_id],
                               capture_output=True)


# ── STDOUT logging helpers (MANDATORY format) ────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rstr}", flush=True)


# ── LLM decision-making ─────────────────────────────────────────────────────
def build_user_prompt(task_id: str, obs: dict) -> str:
    lines = [
        f"Hour: {obs['hour_of_day']:02d}:00 | Context: {obs['current_context']} | "
        f"Trend: '{obs['viral_trend']}'",
        f"Budget: ${obs['remaining_budget']:.2f} | Pressure: {obs['market_pressure']:.2f} | "
        f"Fatigue: {obs['fatigue_level']:.2f}",
    ]
    if task_id == "hard_sequencing":
        lines.append(f"Carryover boost: {obs.get('carryover_boost', 0):.2f}")

    if task_id == "hard_assembly":
        # Show source creative and live hashtags
        img_desc   = obs.get("image_description", "")
        base_cap   = obs.get("base_caption", "")
        live_tags  = obs.get("live_hashtags", [])
        hashtag_list = "  ".join(live_tags) if live_tags else "(none scraped)"
        lines.append("")
        lines.append(f"━━━━━ SOURCE CREATIVE ━━━━━")
        lines.append(f"Image description : {img_desc}")
        lines.append(f"Base caption      : {base_cap}")
        lines.append(f"")
        lines.append(f"━━━━━ LIVE VIRAL HASHTAGS (scraped now) ━━━━━")
        lines.append(f"  {hashtag_list}")
        lines.append(f"")
        lines.append(f"━━━━━ TASK ━━━━━")
        lines.append(f"Select 2–4 hashtags from the list above that best match "
                     f"the viral trend '{obs['viral_trend']}'.")
        lines.append(f"Rewrite the base caption to weave them in naturally.")
        lines.append(f"Stay coherent with the image. Add your own creative words.")
        lines.append("")
        schema = ('Respond ONLY with JSON:\n'
                  '{"bid_price": <float>, "headline_id": <int 0-5>, "creative_id": <int 0-5>, '
                  '"generated_caption": "<your caption>", '
                  '"generated_hashtags": ["#Tag1", "#Tag2", ...]}')
    else:
        lines.append(CATALOG_CTX)
        schema = '{"bid_price": <float>, "headline_id": <int 0-5>, "creative_id": <int 0-5>}'
        if task_id != "hard_assembly":
            schema = f"Respond ONLY with JSON: {schema}"

    lines.append(schema)
    return "\n".join(lines)


def call_llm(client: OpenAI, system: str, user: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {"bid_price": 0.5, "headline_id": 0, "creative_id": 0}


# ── Main episode loop ───────────────────────────────────────────────────────
async def run_task(task_id: str, image_name: str) -> float:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await AuctioneerEnvClient.from_docker_image(image_name, task_id=task_id)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_data = call_llm(llm, SYSTEM_PROMPTS[task_id],
                                   build_user_prompt(task_id, obs))

            action = Action(
                bid_price=float(action_data.get("bid_price", 0.5)),
                headline_id=int(action_data.get("headline_id", 0)),
                creative_id=int(action_data.get("creative_id", 0)),
                generated_caption=action_data.get("generated_caption"),
                generated_hashtags=action_data.get("generated_hashtags"),
            )

            result = await env.step(action)
            obs = result.observation
            reward = result.reward

            rewards.append(reward)
            steps_taken = step

            act_str = f"bid({action.bid_price:.2f},h={action.headline_id},c={action.creative_id})"
            if action.generated_caption:
                act_str += f",cap={action.generated_caption[:25]}"
            if action.generated_hashtags:
                act_str += f",tags={len(action.generated_hashtags)}"

            log_step(step=step, action=act_str, reward=reward,
                     done=result.done, error=None)

            if result.done:
                break

        score = result.info.get("task_score", 0.0)
        success = score >= SUCCESS_THRESHOLDS.get(task_id, 0.5)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    if not IMAGE_NAME:
        print("[ERROR] Set LOCAL_IMAGE_NAME env var to the Docker image name.",
              flush=True)
        return
    if not API_KEY:
        print("[ERROR] Set HF_TOKEN or API_KEY env var.", flush=True)
        return

    tasks = ALL_TASKS if TASK_NAME == "all" else (
        [TASK_NAME] if TASK_NAME in ALL_TASKS
        else (print(f"[ERROR] Unknown task: {TASK_NAME}") or []))
    if not tasks:
        return

    scores: Dict[str, float] = {}
    for t in tasks:
        scores[t] = await run_task(t, IMAGE_NAME)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    for t, s in scores.items():
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {t:<20} {bar}  {s:.4f}")
    if len(scores) > 1:
        avg = sum(scores.values()) / len(scores)
        print(f"  {'AVERAGE':<20} {'█' * int(avg * 20) + '░' * (20 - int(avg * 20))}  {avg:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    asyncio.run(main())