# """
# baseline.py — LLM-powered baseline agent for OpenEnv Creative Auctioneer
# =========================================================================

# Runs all three tasks sequentially and prints a structured score report.
# The agent uses GPT-4o-mini as the decision-maker, with task-specific
# prompts that expose the grader logic so the model can optimise accordingly.

# Usage
# -----
#   OPENAI_API_KEY=<key> python baseline.py

# Optional env vars
# -----------------
#   TASK          = easy_headline | medium_pacing | hard_assembly | all  (default: all)
#   USE_LLM_SIMULATOR = 1   to enable Llama-3 User Simulator (requires GPU + model)
# """

# import json
# import os
# import sys
# import textwrap
# from typing import Optional

# from openai import OpenAI

# from environment import OpenEnvAuctioneer
# from models import Action

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# # ---------------------------------------------------------------------------
# # Catalog  (matches environment defaults; extended here for agent context)
# # ---------------------------------------------------------------------------

# CATALOG = textwrap.dedent("""
#     ╔══════════════════ HEADLINES CATALOG ══════════════════╗
#     ║ ID │ Text                                              ║
#     ╠════╪═══════════════════════════════════════════════════╣
#     ║  0 │ Push your limits every single day.               ║  ← Fitness
#     ║  1 │ Next-generation processing power.                ║  ← Tech
#     ║  2 │ Elevate your everyday style.                     ║  ← Fashion
#     ║  3 │ Level up your competitive play.                  ║  ← Gaming
#     ║  4 │ Sustainable choices for a better tomorrow.       ║  ← Eco/Fitness
#     ║  5 │ Uncompromising quality and elegance.             ║  ← Luxury/Fashion
#     ╚════╧═══════════════════════════════════════════════════╝

#     ╔══════════════════ CREATIVES CATALOG ══════════════════╗
#     ║ ID │ Image Description                                 ║
#     ╠════╪═══════════════════════════════════════════════════╣
#     ║  0 │ A runner silhouetted against a mountain sunrise. ║  ← Fitness
#     ║  1 │ A glowing silicon microchip on a motherboard.   ║  ← Tech
#     ║  2 │ A model wearing a tailored coat on a city street.║  ← Fashion
#     ║  3 │ An RGB mechanical keyboard in a dark room.       ║  ← Gaming
#     ║  4 │ Product packaging from recycled kraft paper.     ║  ← Eco
#     ║  5 │ A gold watch resting on black velvet.            ║  ← Luxury
#     ╚════╧═══════════════════════════════════════════════════╝
# """)

# # ---------------------------------------------------------------------------
# # Task-specific system prompts
# # ---------------------------------------------------------------------------

# SYSTEM_EASY = """
# You are an AI Account Manager. Your ONLY goal is to maximise CTR by
# selecting the headline and creative that semantically best match the
# user's CURRENT CONTEXT and VIRAL TREND each hour.

# Context→Best match guide (context: headline_id, creative_id):
#   Fitness  → 0 or 4,  0
#   Tech     → 1,       1
#   Fashion  → 2 or 5,  2 or 5
#   Gaming   → 3,       3

# Keep bids modest (0.30–0.80) to stay in budget. Never bid over $2.
# """.strip()

# SYSTEM_MEDIUM = """
# You are an AI Account Manager focused on BUDGET PACING.
# Rules:
# 1. You start with $50 for 24 hours → ideal spend = $50/24 ≈ $2.08/hour.
# 2. You MUST have ≥20% of budget ($10) remaining when hour 18 arrives.
# 3. Hours 18–22 are peak — bid aggressively (up to $1.80) during those hours.
# 4. Hours 0–17: bid conservatively (0.20–0.60) to preserve budget.
# 5. If budget < $5 before hour 18, bid $0 (skip the auction deliberately).

# Scoring: smoothness (30%) + peak survival (30%) + total revenue (40%).
# """.strip()

# SYSTEM_HARD = """
# You are an AI Account Manager optimising for VIRAL TREND ALIGNMENT.
# Your score is 60% cosine-similarity between your caption and the viral trend
# + 40% total revenue.

# For each step you MUST generate a short, punchy caption (≤12 words) that:
#   • Contains the exact viral trend keywords if possible.
#   • Semantically aligns with both the context AND the trend.
#   • Sounds like a real social-media ad, not a generic statement.

# Examples:
#   Context=Fitness, Trend=Eco-Friendly  → "Run clean. Live green. Eco-gear for your trail."
#   Context=Tech,    Trend=Cyberpunk     → "Jack in. The future runs on neon silicon."
#   Context=Fashion, Trend=Quiet Luxury  → "Quiet luxury. Loud statement. Yours."
#   Context=Gaming,  Trend=Minimalism    → "Less lag. Less noise. Pure performance."

# Also choose headline_id and creative_id that reinforce the caption theme.
# Bid competitively ($0.60–$1.50) since revenue matters too.
# """.strip()

# SYSTEM_PROMPTS = {
#     "easy_headline":  SYSTEM_EASY,
#     "medium_pacing":  SYSTEM_MEDIUM,
#     "hard_assembly":  SYSTEM_HARD,
# }


# # ---------------------------------------------------------------------------
# # Agent
# # ---------------------------------------------------------------------------

# def build_user_prompt(task_id: str, obs, catalog: str) -> str:
#     base = (
#         f"Hour: {obs.hour_of_day:02d}:00 | "
#         f"Context: {obs.current_context} ({obs.news_category}) | "
#         f"Viral Trend: '{obs.viral_trend}'\n"
#         f"Budget remaining: ${obs.remaining_budget} | "
#         f"Market pressure: {obs.market_pressure:.2f} | "
#         f"Fatigue level: {obs.fatigue_level:.2f}\n\n"
#         f"{catalog}\n\n"
#     )

#     schema = (
#         'Respond ONLY with a JSON object:\n'
#         '{"bid_price": <float>, "headline_id": <int 0-5>, "creative_id": <int 0-5>'
#     )
#     if task_id == "hard_assembly":
#         schema += ', "generated_caption": "<your caption string>"'
#     schema += "}"

#     return base + schema


# def call_llm(system: str, user: str) -> dict:
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user",   "content": user},
#         ],
#         response_format={"type": "json_object"},
#         temperature=0.2,
#     )
#     return json.loads(resp.choices[0].message.content)


# def run_task(task_id: str) -> float:
#     sep = "═" * 52
#     print(f"\n{sep}")
#     print(f"  TASK: {task_id.upper()}")
#     print(f"{sep}")

#     env  = OpenEnvAuctioneer(task_id=task_id)
#     obs  = env.reset()
#     done = False
#     system_prompt = SYSTEM_PROMPTS[task_id]
#     final_info    = None

#     while not done:
#         user_prompt   = build_user_prompt(task_id, obs, CATALOG)
#         action_data   = call_llm(system_prompt, user_prompt)

#         # Safely build Action (generated_caption may be absent for easy/medium)
#         action = Action(
#             bid_price         = float(action_data.get("bid_price", 0.5)),
#             headline_id       = int(action_data.get("headline_id", 0)),
#             creative_id       = int(action_data.get("creative_id", 0)),
#             generated_caption = action_data.get("generated_caption", None),
#         )

#         obs, reward_obj, done, info = env.step(action)
#         reward = reward_obj.value
#         final_info = info

#         won_mark = "✓ WON" if info.auction_won else "✗ LOST"
#         caption_str = ""
#         if action.generated_caption:
#             caption_str = f" | Caption: \"{action.generated_caption[:40]}…\""

#         print(
#             f"  [{obs.hour_of_day:02d}:00] {won_mark} "
#             f"bid=${action.bid_price:.2f} clear=${info.clearing_price:.2f} | "
#             f"CTR={info.adjusted_ctr:.3f} | "
#             f"r={reward:+.2f} | "
#             f"rev=${info.total_revenue:.2f} | "
#             f"score={info.task_score:.3f}"
#             f"{caption_str}"
#         )

#     # ── Final Score Report ───────────────────────────────────────────────
#     print(f"\n{'─'*52}")
#     print(f"  FINAL SCORE  {final_info.task_score:.4f} / 1.0")
#     if task_id == "easy_headline":
#         print(f"  Headline Alignment Score : {final_info.headline_alignment_score:.4f}")
#     elif task_id == "medium_pacing":
#         print(f"  Pacing Score             : {final_info.pacing_score:.4f}")
#     elif task_id == "hard_assembly":
#         print(f"  CLIP Similarity Score    : {final_info.clip_similarity_score:.4f}")
#         print(f"  Revenue Factor           : ${final_info.total_revenue:.2f}")
#     print(f"{'─'*52}\n")

#     return final_info.task_score


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     task_arg = os.environ.get("TASK", "all").lower()

#     if task_arg == "all":
#         tasks = ["easy_headline", "medium_pacing", "hard_assembly"]
#     elif task_arg in ("easy_headline", "medium_pacing", "hard_assembly"):
#         tasks = [task_arg]
#     else:
#         print(f"Unknown TASK='{task_arg}'. Choose: easy_headline | medium_pacing | hard_assembly | all")
#         sys.exit(1)

#     scores = {}
#     for t in tasks:
#         scores[t] = run_task(t)

#     # ── Summary ──────────────────────────────────────────────────────────
#     print("╔══════════════════ SCORE SUMMARY ══════════════════╗")
#     for t, s in scores.items():
#         bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
#         print(f"║  {t:<20} {bar}  {s:.4f} ║")
#     if len(scores) > 1:
#         avg = sum(scores.values()) / len(scores)
#         print(f"╠{'═'*50}╣")
#         print(f"║  {'AVERAGE':<20} {'█'*int(avg*20) + '░'*(20-int(avg*20))}  {avg:.4f} ║")
#     print("╚══════════════════════════════════════════════════╝")

"""
baseline.py — Gemini-powered baseline agent for OpenEnv Creative Auctioneer
============================================================================
"""

import json
import os
import sys
import textwrap
from typing import Optional

import google.generativeai as genai

from environment import OpenEnvAuctioneer
from models import Action

# ---------------------------------------------------------------------------
# Gemini Setup
# ---------------------------------------------------------------------------

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Use Gemini 2.5 Pro (best reasoning)
model = genai.GenerativeModel("gemini-2.5")

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
# System Prompts
# ---------------------------------------------------------------------------

SYSTEM_EASY = """
Maximise CTR by matching context with best headline + creative.

Fitness → (0 or 4, 0)
Tech → (1,1)
Fashion → (2 or 5, 2 or 5)
Gaming → (3,3)

Keep bids between 0.30–0.80. Never exceed $2.
"""

SYSTEM_MEDIUM = """
Budget pacing rules:

- Total budget: $50 for 24h → ~2.08/hour
- Must keep ≥$10 by hour 18
- Hours 18–22: bid aggressively (≤1.8)
- Hours 0–17: bid conservatively (0.2–0.6)
- If budget < $5 before hour 18 → bid 0

Optimise smooth spend + peak survival + revenue
"""

SYSTEM_HARD = """
Optimise for viral trend alignment.

Score = 60% similarity + 40% revenue

Generate caption (≤12 words):
- Include viral keywords
- Match context + trend
- Sound like real ad

Also pick matching headline + creative.

Bid: 0.60–1.50
"""

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
        f"Context: {obs.current_context} ({obs.news_category}) | "
        f"Trend: '{obs.viral_trend}'\n"
        f"Budget: ${obs.remaining_budget} | "
        f"Pressure: {obs.market_pressure:.2f} | "
        f"Fatigue: {obs.fatigue_level:.2f}\n\n"
        f"{catalog}\n\n"
    )

    schema = (
        'Return ONLY JSON:\n'
        '{"bid_price": float, "headline_id": int, "creative_id": int'
    )

    if task_id == "hard_assembly":
        schema += ', "generated_caption": "string"'

    schema += "}"

    return base + schema


# ---------------------------------------------------------------------------
# Gemini Call
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str) -> dict:
    prompt = f"""
{system}

{user}

STRICT RULES:
- Output ONLY valid JSON
- No explanation
- No markdown
"""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2}
    )

    text = response.text.strip()

    # Clean markdown if present
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except:
        print("⚠️ JSON parse failed. Raw output:\n", text)
        return {
            "bid_price": 0.5,
            "headline_id": 0,
            "creative_id": 0
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    print(f"\n=== TASK: {task_id} ===")

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

        obs, reward_obj, done, info = env.step(action)

        print(
            f"[{obs.hour_of_day:02d}] "
            f"bid={action.bid_price:.2f} | "
            f"CTR={info.adjusted_ctr:.3f} | "
            f"rev=${info.total_revenue:.2f} | "
            f"score={info.task_score:.3f}"
        )

        final_info = info

    print(f"\nFINAL SCORE: {final_info.task_score:.4f}\n")
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

    print("\n=== SUMMARY ===")
    for t, s in scores.items():
        print(f"{t}: {s:.4f}")