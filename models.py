from pydantic import BaseModel, Field

class Observation(BaseModel):
    hour_of_day: int = Field(..., ge=0, le=23, description="Current hour of the campaign (0-23).")
    remaining_budget: float = Field(..., description="Remaining daily budget in USD.")
    current_context: str = Field(..., description="The user's current content context (e.g., 'Fitness', 'Tech').")
    viral_trend: str = Field(..., description="The current cultural viral token (e.g., 'Quiet Luxury').")

class Action(BaseModel):
    bid_price: float = Field(..., ge=0.0, description="Your auction bid in USD.")
    headline_id: int = Field(..., ge=0, le=5, description="Selected headline ID (0-5).")
    creative_id: int = Field(..., ge=0, le=5, description="Selected image/creative ID (0-5).")

class Info(BaseModel):
    task_id: str
    current_step: int
    total_revenue: float
    task_score: float = Field(..., description="0.0 to 1.0 task completion score.")