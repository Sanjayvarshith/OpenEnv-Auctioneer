import math
import numpy as np
from typing import Tuple
from sentence_transformers import SentenceTransformer, util
from models import Observation, Action, Info

class OpenEnvAuctioneer:
    def __init__(self, task_id: str = "easy_headline"):
        self.task_id = task_id
        self.max_steps = 24
        
        # Load the small, fast semantic model for real NLP evaluation
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.contexts = ["Fitness", "Tech", "Fashion", "Gaming"]
        self.trends = ["Quiet Luxury", "Eco-Friendly", "Cyberpunk", "Minimalism"]
        
        # Textual dictionaries for the LLM to evaluate
        self.headlines = {
            0: "Push your limits every single day.",
            1: "Next-generation processing power.",
            2: "Elevate your everyday style.",
            3: "Level up your competitive play.",
            4: "Sustainable choices for a better tomorrow.",
            5: "Uncompromising quality and elegance."
        }
        self.creatives = {
            0: "[Image: A runner silhouetted against a mountain sunrise.]",
            1: "[Image: A glowing silicon microchip on a motherboard.]",
            2: "[Image: A model wearing a tailored coat on a city street.]",
            3: "[Image: An RGB mechanical keyboard in a dark room.]",
            4: "[Image: Product packaging made from recycled kraft paper.]",
            5: "[Image: A gold watch resting on black velvet.]"
        }
        
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
        # Real statistical market simulation (Lognormal distribution)
        # Prices peak at noon (hour 12) and 6 PM (hour 18)
        base_mu = -0.5 # Lognormal underlying mean
        peak_shift = math.sin((self.hour_of_day / 24.0) * math.pi) * 0.8
        
        mu = base_mu + peak_shift
        sigma = 0.25 # Market volatility
        
        # Sample the clearing price
        price = np.random.lognormal(mean=mu, sigma=sigma)
        return round(float(price), 2)

    def _simulate_user_ctr(self, action: Action) -> float:
        # 1. Construct the User Persona string
        user_persona = f"A user watching {self.current_context} content, currently interested in the {self.viral_trend} trend."
        
        # 2. Construct the Ad Content string
        ad_content = f"{self.headlines.get(action.headline_id, '')} {self.creatives.get(action.creative_id, '')}"
        
        # 3. Use the real Neural Network to calculate semantic alignment
        user_embedding = self.semantic_model.encode(user_persona, convert_to_tensor=True)
        ad_embedding = self.semantic_model.encode(ad_content, convert_to_tensor=True)
        
        cosine_score = util.cos_sim(user_embedding, ad_embedding).item()
        
        # Map cosine similarity (-1 to 1) to a realistic CTR (0.01 to 0.40)
        base_ctr = max(0.01, (cosine_score * 0.40))
        
        return max(0.0, min(1.0, base_ctr - self.fatigue_penalty))

    def step(self, action: Action) -> Tuple[Observation, float, bool, Info]:
        if self.remaining_budget <= 0:
            return self.state(), 0.0, True, self._get_info()

        clearing_price = self._simulate_market_clearing_price()
        step_reward = 0.0
        
        if action.bid_price >= clearing_price and self.remaining_budget >= clearing_price:
            self.remaining_budget -= clearing_price
            
            # Real AI-driven CTR calculation
            ctr = self._simulate_user_ctr(action)
            revenue = ctr * 15.0 # Conversion value
            self.total_revenue += revenue
            
            step_reward = revenue - clearing_price
            self.fatigue_penalty += 0.015 
            
        else:
            step_reward = -0.10
            self.fatigue_penalty = max(0.0, self.fatigue_penalty - 0.05)

        if self.task_id == "medium_pacing":
            ideal_burn_rate = 50.0 / 24.0
            actual_burn_rate = (50.0 - self.remaining_budget) / max(1, self.current_step + 1)
            if actual_burn_rate > ideal_burn_rate * 1.5:
                step_reward -= 1.0 

        self.current_step += 1
        self.hour_of_day = self.current_step
        done = self.current_step >= self.max_steps or self.remaining_budget <= 0
        
        if not done:
            self._update_context()

        return self.state(), round(step_reward, 2), done, self._get_info()

    def _get_info(self) -> Info:
        score = 0.0
        if self.task_id == "easy_headline":
            score = min(1.0, self.total_revenue / 25.0) 
        elif self.task_id == "medium_pacing":
            survival_bonus = self.current_step / 24.0
            revenue_factor = min(1.0, self.total_revenue / 30.0)
            score = (survival_bonus * 0.4) + (revenue_factor * 0.6)
        elif self.task_id == "hard_assembly":
            score = min(1.0, self.total_revenue / 45.0)

        return Info(
            task_id=self.task_id,
            current_step=self.current_step,
            total_revenue=round(self.total_revenue, 2),
            task_score=round(score, 2)
        )