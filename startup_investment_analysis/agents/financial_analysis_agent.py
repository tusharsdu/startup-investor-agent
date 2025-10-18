from core.base_agent import BaseAgent
from openai import OpenAI

class FinancialAnalysisAgent(BaseAgent):
    def __init__(self, perplexity_api_key: str):
        super().__init__("FinancialAnalysisAgent")
        self.client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def process(self, pitch_text):
        prompt = f"""
You are a senior financial analyst with extensive expertise in startup valuations and venture capital investments. Provide a comprehensive financial analysis in one detailed, elaborate paragraph that thoroughly examines the startup's financial foundation, growth trajectory, and investment viability. Your analysis should seamlessly integrate current revenue performance, growth metrics, burn rate efficiency, unit economics including customer acquisition costs and lifetime value ratios, cash runway calculations, funding requirements, valuation methodologies using comparable company analysis and growth multiples, competitive financial positioning, path to profitability with timeline projections, capital efficiency metrics, working capital requirements, revenue diversification strategies, financial risk factors including market sensitivity and operational leverage, scalability of the business model from a financial perspective, and strategic financial recommendations for sustainable growth. Ensure your response flows naturally while incorporating specific financial metrics, benchmarking data, and quantitative insights that demonstrate sophisticated financial modeling capabilities and provide actionable intelligence for investment decision-making at the senior management level.

Pitch Data:
{pitch_text}
"""
        response = self.client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        financial_text = response.choices[0].message.content.strip()

        return {"financial_analysis": financial_text}
