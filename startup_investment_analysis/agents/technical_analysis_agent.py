from core.base_agent import BaseAgent
from openai import OpenAI

class TechnicalAnalysisAgent(BaseAgent):
    def __init__(self, perplexity_api_key: str):
        super().__init__("TechnicalAnalysisAgent")
        self.client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def process(self, pitch_text):
        prompt = f"""
You are a senior technical architect and CTO advisor with extensive experience in technology evaluation and system design. Provide a comprehensive technical analysis in one detailed, elaborate paragraph that thoroughly examines the technology foundation, architectural soundness, and innovation potential of the startup. Your analysis should seamlessly integrate technology stack assessment including frontend frameworks, backend infrastructure, database architecture, cloud platform utilization and scalability strategies, system design principles and architectural patterns with emphasis on microservices versus monolithic approaches, scalability analysis covering horizontal and vertical scaling capabilities, performance optimization strategies, load balancing and traffic management, innovation assessment focusing on proprietary algorithms, artificial intelligence and machine learning implementations, technical differentiation from competitors, development methodology and engineering best practices including code quality standards, testing frameworks, continuous integration and deployment pipelines, security architecture and cybersecurity protocols, data encryption strategies, authentication and authorization systems, compliance with industry standards and regulatory requirements, intellectual property analysis including patent landscape and technical moat strength, technical feasibility assessment of product roadmap and development timeline, resource requirements for engineering team scaling, technology risk evaluation including vendor dependencies, technical debt assessment, and strategic technology recommendations for sustainable growth. Ensure your response flows naturally while incorporating specific technical metrics, architectural insights, and innovation scoring that demonstrate sophisticated technical evaluation capabilities for investment decision-making.

Technical Review Target:
{pitch_text}
"""
        response = self.client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        tech_text = response.choices[0].message.content.strip()

        return {"technical_analysis": tech_text}
