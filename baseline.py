
"""
baseline.py — LLM-powered baseline agent for OpenEnv Creative Auctioneer
(Updated to use Gemini API instead of OpenAI)
"""

import json
import os
import sys
import textwrap
import requests
from typing import Optional

from environment import OpenEnvAuctioneer
from models import Action

# ---------------------------------------------------------------------------
# Gemini API Setup
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CATALOG = textwrap.dedent("""
    ╔══════════════════ HEADLINES CATALOG ══════════════════╗
    ║ ID │ Text                                              ║
    ╠════╪═══════════════════════════════════════════════════╣
    ║  0 │ Push your limits every single day.               ║
    ║  1 │ Next-generation processing power.                ║
    ║  2 │ Elevate your everyday style.                     ║
    ║  3 │ Level up your competitive play.                  ║
    ║  4 │ Sustainable choices for a better tomorrow.       ║
    ║  5 │ Uncompromising quality and elegance.             ║
    ╚════╧═══════════════════════════════════════════════════╝

    ╔══════════════════ CREATIVES CATALOG ══════════════════╗
    ║ ID │ Image Description                                 ║
    ╠════╪═══════════════════════════════════════════════════╣
    ║  0 │ A runner silhouetted against a mountain sunrise. ║
    ║  1 │ A glowing silicon microchip on a motherboard.   ║
    ║  2 │ A model wearing a tailored coat on a city street.║
    ║  3 │ An RGB mechanical keyboard in a dark room.       ║
    ║  4 │ Product packaging from recycled kraft paper.     ║
    ║  5 │ A gold watch resting on black velvet.            ║
    ╚════╧═══════════════════════════════════════════════════╝
""")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_EASY = """
You are an AI Account Manager. Your ONLY goal is to maximise CTR.

Context→Best match:
Fitness → 0 or 4, 0
Tech → 1, 1
Fashion → 2 or 5, 2 or 5
Gaming → 3, 3

Keep bids modest (0.30–0.80)
""".strip()

SYSTEM_MEDIUM = """
You are focused on BUDGET PACING.

Rules:
- Start with $50 for 24h
- ≥$10 remaining at hour 18
- Hours 18–22 → aggressive
- Hours 0–17 → conservative
- If < $5 early → bid 0
""".strip()

SYSTEM_HARD = """
Optimise for VIRAL TREND ALIGNMENT.

- Generate caption ≤12 words
- Include trend keywords
- Match context + trend
- Bid $0.60–$1.50
""".strip()

SYSTEM_PROMPTS = {
    "easy_headline": SYSTEM_EASY,
    "medium_pacing": SYSTEM_MEDIUM,
    "hard_assembly": SYSTEM_HARD,
}

# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def build_user_prompt(task_id: str, obs, catalog: str) -> str:
    base = (
        f"Hour: {obs.hour_of_day:02d}:00 | "
        f"Context: {obs.current_context} | "
        f"Viral Trend: '{obs.viral_trend}'\n"
        f"Budget remaining: ${obs.remaining_budget}\n\n"
        f"{catalog}\n\n"
    )

    schema = (
        'Respond ONLY with VALID JSON.\n'
        '{"bid_price": <float>, "headline_id": <int>, "creative_id": <int>'
    )

    if task_id == "hard_assembly":
        schema += ', "generated_caption": "<string>"'

    schema += "}"

    return base + schema

# ---------------------------------------------------------------------------
# Gemini Call
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str) -> dict:
    prompt = system + "\n\n" + user

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2
        }
    }

    response = requests.post(GEMINI_URL, json=payload)
    result = response.json()

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]

        # Extract JSON safely
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]

        return json.loads(json_str)

    except Exception:
        print("Gemini error:", result)
        return {"bid_price": 0.5, "headline_id": 0, "creative_id": 0}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    print(f"\nRunning: {task_id}")

    env = OpenEnvAuctioneer(task_id=task_id)
    obs = env.reset()
    done = False
    system_prompt = SYSTEM_PROMPTS[task_id]

    final_info = None

    while not done:
        user_prompt = build_user_prompt(task_id, obs, CATALOG)
        action_data = call_llm(system_prompt, user_prompt)

        action = Action(
            bid_price=float(action_data.get("bid_price", 0.5)),
            headline_id=int(action_data.get("headline_id", 0)),
            creative_id=int(action_data.get("creative_id", 0)),
            generated_caption=action_data.get("generated_caption", None),
        )

        obs, reward, done, info = env.step(action)
        final_info = info

        print(
            f"[{obs.hour_of_day:02d}] bid={action.bid_price:.2f} "
            f"rev=${info.total_revenue:.2f} score={info.task_score:.3f}"
        )

    print(f"Final Score: {final_info.task_score:.4f}")
    return final_info.task_score

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_arg = os.environ.get("TASK", "all").lower()

    if task_arg == "all":
        tasks = ["easy_headline", "medium_pacing", "hard_assembly"]
    else:
        tasks = [task_arg]

    scores = {}

    for t in tasks:
        scores[t] = run_task(t)

    print("\nSummary:")
    for t, s in scores.items():
        print(f"{t}: {s:.4f}")

