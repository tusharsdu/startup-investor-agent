from core.base_agent import BaseAgent
from openai import OpenAI

class RiskComplianceAgent(BaseAgent):
    def __init__(self, perplexity_api_key: str):
        super().__init__("RiskComplianceAgent")
        self.client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def process(self, pitch_text):
        prompt = f"""
You are a senior risk management consultant with extensive expertise in startup risk assessment and regulatory compliance. Deliver a comprehensive risk analysis in one detailed, elaborate paragraph that thoroughly examines all critical risk dimensions and mitigation strategies. Your analysis should seamlessly integrate legal and regulatory risk assessment including industry-specific compliance requirements, intellectual property vulnerabilities, data privacy regulations such as GDPR and CCPA, employment law considerations, consumer protection obligations, international regulatory complexities, litigation exposure and legal precedent analysis, operational risk evaluation covering key person dependencies, supply chain vulnerabilities, cybersecurity threats, scalability bottlenecks, business continuity planning, third-party vendor risks, financial risk analysis including cash flow volatility, customer concentration risks, market sensitivity to economic cycles, funding and capital raising challenges, currency exposure for international operations, credit and counterparty risks, market and competitive risk assessment covering market timing uncertainties, competitive displacement threats, technology obsolescence risks, customer behavior shifts, strategic and execution risks including product development challenges, go-to-market execution complexities, partnership dependencies, acquisition integration risks, ESG (Environmental, Social, Governance) risk factors including environmental impact, social responsibility obligations, corporate governance structure adequacy, reputation management, and comprehensive risk mitigation strategies with priority ranking and monitoring frameworks. Ensure your response flows naturally while incorporating specific risk metrics, probability assessments, and strategic recommendations that demonstrate sophisticated risk management capabilities for senior investment decision-making.

Risk Assessment Target:
{pitch_text}
"""
        response = self.client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        risk_text = response.choices[0].message.content.strip()

        return {"risk_analysis": risk_text}
