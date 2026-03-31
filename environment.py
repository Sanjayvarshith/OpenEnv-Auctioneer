import math
from typing import Tuple
from models import Observation, Action, Info

class OpenEnvAuctioneer:
    def __init__(self, task_id: str = "easy_headline"):
        self.task_id = task_id
        self.max_steps = 24 # 24-hour cycle
        self.contexts = ["Fitness", "Tech", "Fashion", "Gaming"]
        self.trends = ["Quiet Luxury", "Eco-Friendly", "Cyberpunk", "Minimalism"]
        self.reset()

    def reset(self) -> Observation:
        self.current_step = 0
        self.total_revenue = 0.0
        self.hour_of_day = 0
        self.remaining_budget = 100.0 if self.task_id != "medium_pacing" else 50.0
        self.fatigue_penalty = 0.0
        
        self._update_context()
        return self.state()

    def _update_context(self):
        # Deterministic context switching based on the hour
        self.current_context = self.contexts[self.hour_of_day % len(self.contexts)]
        self.viral_trend = self.trends[self.hour_of_day % len(self.trends)]

    def state(self) -> Observation:
        return Observation(
            hour_of_day=self.hour_of_day,
            remaining_budget=round(self.remaining_budget, 2),
            current_context=self.current_context,
            viral_trend=self.viral_trend
        )

    def _simulate_market_clearing_price(self) -> float:
        # Deterministic market simulation: Prices peak at hour 12 and 18
        base_price = 0.50
        peak_multiplier = math.sin((self.hour_of_day / 24.0) * math.pi) 
        return round(base_price + (peak_multiplier * 1.50), 2)

    def _simulate_user_ctr(self, action: Action) -> float:
        # Mock User Simulator (Replace with LLM/CLIP later)
        ctr = 0.10 # Base CTR
        
        # Reward matching headline ID to context index (Level 1 logic)
        context_idx = self.contexts.index(self.current_context)
        if action.headline_id == context_idx:
            ctr += 0.30
            
        # Reward matching creative to viral trend (Level 3 logic)
        trend_idx = self.trends.index(self.viral_trend)
        if action.creative_id == trend_idx:
            ctr += 0.40
            
        return min(1.0, ctr - self.fatigue_penalty)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Info]:
        if self.remaining_budget <= 0:
            return self.state(), 0.0, True, self._get_info()

        clearing_price = self._simulate_market_clearing_price()
        step_reward = 0.0
        
        # 1. Market Engine: Did we win the auction?
        if action.bid_price >= clearing_price and self.remaining_budget >= clearing_price:
            self.remaining_budget -= clearing_price
            
            # 2. User Simulator: Did they click/convert?
            ctr = self._simulate_user_ctr(action)
            revenue = ctr * 10.0 # Arbitrary conversion value
            self.total_revenue += revenue
            
            # Calculate Return on Ad Spend (ROAS) for this step
            step_reward = revenue - clearing_price
            self.fatigue_penalty += 0.02 # Ads get less effective if spammed
            
        else:
            # Penalty for bidding too low or having no budget
            step_reward = -0.10
            self.fatigue_penalty = max(0.0, self.fatigue_penalty - 0.05) # Fatigue recovers

        # Pacing Penalty (Level 2 logic: penalize spending too fast)
        if self.task_id == "medium_pacing":
            ideal_burn_rate = 50.0 / 24.0
            actual_burn_rate = (50.0 - self.remaining_budget) / max(1, self.current_step + 1)
            if actual_burn_rate > ideal_burn_rate * 1.5:
                step_reward -= 1.0 # Continuous penalty for blowing budget early

        self.current_step += 1
        self.hour_of_day = self.current_step
        done = self.current_step >= self.max_steps or self.remaining_budget <= 0
        
        if not done:
            self._update_context()

        return self.state(), round(step_reward, 2), done, self._get_info()

    def _get_info(self) -> Info:
        # Deterministic Grader (0.0 to 1.0)
        score = 0.0
        if self.task_id == "easy_headline":
            score = min(1.0, self.total_revenue / 30.0) 
        elif self.task_id == "medium_pacing":
            # High score if budget survived past hour 20 and revenue is decent
            survival_bonus = self.current_step / 24.0
            revenue_factor = min(1.0, self.total_revenue / 40.0)
            score = (survival_bonus * 0.5) + (revenue_factor * 0.5)
        elif self.task_id == "hard_assembly":
            score = min(1.0, self.total_revenue / 60.0)

        return Info(
            task_id=self.task_id,
            current_step=self.current_step,
            total_revenue=round(self.total_revenue, 2),
            task_score=round(score, 2)
        )