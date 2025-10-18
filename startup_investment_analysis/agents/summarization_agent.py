from core.base_agent import BaseAgent
from openai import OpenAI

class SummarizationAgent(BaseAgent):
    def __init__(self, perplexity_api_key: str):
        super().__init__("SummarizationAgent")
        self.client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def process(self, pitch_text: str) -> dict:
        prompt = f"""
You are a senior VC analyst with 15+ years of experience. Create a comprehensive executive summary in one detailed, flowing paragraph that seamlessly integrates all critical investment elements. Your response should be an elaborate, cohesive narrative that thoroughly covers the company's value proposition, market opportunity, business model, team credentials, financial traction, competitive advantages, growth strategy, and investment thesis. Write in a sophisticated, professional tone that demonstrates deep analytical insight while maintaining readability. Ensure the summary flows naturally from one concept to the next, providing specific metrics, concrete examples, and compelling evidence that builds a strong investment case. The paragraph should be substantial enough to serve as a comprehensive standalone summary for senior investment committee review, incorporating nuanced observations about market timing, execution capabilities, and strategic positioning that distinguish this opportunity in the venture capital landscape.

Pitch Content:
{pitch_text}
"""
        response = self.client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        summary = response.choices[0].message.content.strip()
        return {"summary": summary}
