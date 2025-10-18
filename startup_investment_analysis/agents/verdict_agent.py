from core.base_agent import BaseAgent
from openai import OpenAI
import json

class VerdictAgent(BaseAgent):
    def __init__(self, perplexity_api_key: str):
        super().__init__("VerdictAgent")
        self.client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def process(self, agent_outputs: dict, vc_qa: dict) -> dict:
        """
        agent_outputs: dict containing outputs from all analysis agents
        vc_qa: dict containing VC Q&A history
        Returns: dict with verdict, confidence, investment amount, equity %, justification
        """
        prompt = f"""
You are a senior venture capital partner with 15+ years of investment experience and a proven track record of successful portfolio companies. Based on the comprehensive multi-agent analysis provided, deliver your investment decision in one detailed, elaborate paragraph that thoroughly integrates all analytical findings into a cohesive investment thesis. Your verdict should seamlessly weave together market opportunity assessment with specific market size validation and competitive positioning analysis, team evaluation including founder experience and execution capabilities, product-market fit evidence through traction metrics and customer validation, financial viability analysis incorporating unit economics, revenue projections, and capital efficiency, technical feasibility and innovation assessment, comprehensive risk evaluation across all identified categories with mitigation strategies, scalability potential and growth trajectory analysis, competitive advantages and defensibility factors, exit scenario potential and timeline considerations, strategic value alignment with current market trends and investor thesis, confidence level justification based on evidence quality and analytical depth, recommended investment amount rationale tied to funding requirements and growth milestones, equity percentage recommendation based on valuation methodology and risk-reward profile, and detailed investment justification that addresses potential concerns while highlighting compelling investment drivers. Your response should demonstrate sophisticated investment judgment, incorporate specific evidence from the multi-agent analysis, address potential red flags or mitigation requirements, and provide clear rationale for your decision that would be compelling to senior investment committee members and limited partners.

Multi-Agent Analysis Results:
{json.dumps(agent_outputs, indent=2)[:3000]}...

VC Q&A Validation:
{json.dumps(vc_qa, indent=2)[:1500]}...

Return JSON format: {{"verdict": "Invest/Maybe/Do Not Invest", "confidence": 0.85, "recommended_investment_usd": 2000000, "recommended_equity_percent": 15, "justification": "detailed reasoning..."}}
"""

        response = self.client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer_text = response.choices[0].message.content.strip()
        try:
            verdict_json = json.loads(answer_text)
        except json.JSONDecodeError:
            # fallback if GPT returns slightly invalid JSON
            verdict_json = {
                "verdict": "Unknown",
                "confidence": 0.0,
                "recommended_investment_usd": 0,
                "recommended_equity_percent": 0,
                "justification": answer_text
            }

        return verdict_json
