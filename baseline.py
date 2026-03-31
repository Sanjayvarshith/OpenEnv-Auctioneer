import os
import json
from openai import OpenAI
from environment import OpenEnvAuctioneer
from models import Action

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_agent(task_id: str):
    print(f"\n{'='*40}\nStarting Task: {task_id}\n{'='*40}")
    env = OpenEnvAuctioneer(task_id=task_id)
    obs = env.reset()
    done = False
    
    # Provide the catalog to the agent
    catalog = f"""
    Headlines Catalog:
    0: "Push your limits every single day."
    1: "Next-generation processing power."
    2: "Elevate your everyday style."
    3: "Level up your competitive play."
    4: "Sustainable choices for a better tomorrow."
    5: "Uncompromising quality and elegance."
    
    Creatives Catalog:
    0: "[Image: A runner silhouetted against a mountain sunrise.]"
    1: "[Image: A glowing silicon microchip on a motherboard.]"
    2: "[Image: A model wearing a tailored coat on a city street.]"
    3: "[Image: An RGB mechanical keyboard in a dark room.]"
    4: "[Image: Product packaging made from recycled kraft paper.]"
    5: "[Image: A gold watch resting on black velvet.]"
    """
    
    while not done:
        print(f"[{obs.hour_of_day}:00] Budget: ${obs.remaining_budget} | Context: {obs.current_context} | Trend: {obs.viral_trend}")
        
        prompt = f"""
        You are an AI Account Manager. 
        Current Context: {obs.current_context}. Viral Trend: {obs.viral_trend}.
        Budget remaining: ${obs.remaining_budget}. Hour: {obs.hour_of_day}.
        Prices peak around hour 12 and 18.
        
        {catalog}
        
        Select the headline_id and creative_id that best align semantically with the user's current context and trend.
        Respond ONLY with a JSON object matching this schema:
        {{"bid_price": <float>, "headline_id": <int 0-5>, "creative_id": <int 0-5>}}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        action_data = json.loads(response.choices[0].message.content)
        action = Action(**action_data)
        
        obs, reward, done, info = env.step(action)
        print(f"  -> Action: Bid ${action.bid_price}, Head: {action.headline_id}, Creative: {action.creative_id}")
        print(f"  -> Reward: {reward} | Revenue so far: ${info.total_revenue}\n")

    print(f"Task Complete! Final Score: {info.task_score}/1.0")

if __name__ == "__main__":
    run_agent("easy_headline")
    run_agent("medium_pacing")
    run_agent("hard_assembly")