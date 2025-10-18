from core.base_agent import BaseAgent
from openai import OpenAI

class VCQuestionAnswerAgent(BaseAgent):
    def __init__(self, openai_api_key: str, router=None):
        super().__init__("VCQuestionAnswerAgent")
        self.client = OpenAI(api_key=openai_api_key)
        self.router = router

    def generate_question(self, pitch_text: str) -> str:
        prompt = f"""
You are a senior venture capital partner with extensive due diligence experience conducting investment evaluations for high-potential startups. Based on the comprehensive startup pitch provided, generate one sophisticated, strategic question that demonstrates deep analytical thinking and addresses critical investment considerations. Your question should be elaborate and multifaceted, requiring a detailed response that reveals key insights about business model validation, market opportunity assessment, competitive positioning, financial sustainability, scalability potential, execution capabilities, risk mitigation strategies, or growth trajectory planning. The question should be the type that a seasoned investment professional would ask during a thorough due diligence process, focusing on uncovering potential challenges, validating assumptions, or exploring strategic opportunities that could significantly impact investment decision-making. Ensure the question is specific enough to elicit actionable intelligence while being comprehensive enough to demonstrate sophisticated understanding of venture capital evaluation frameworks and startup dynamics.

Startup Pitch:
{pitch_text}

Return only the professional VC question.
"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def process(self, pitch_text: str, num_questions: int = 5) -> dict:
        qa_results = []

        for _ in range(num_questions):
            question = self.generate_question(pitch_text)

            if self.router:
                routing_text = f"Answer this question: {question}\nPitch: {pitch_text}"
                agent_name = self.router.decide_agent(routing_text)
                context = self.router.agents[agent_name].process(pitch_text)
            else:
                context = {"text": pitch_text}

            prompt = f"""
You are a startup founder presenting to a senior VC partner during a due diligence meeting. Provide a comprehensive, elaborate answer to the investor's question in one detailed paragraph that demonstrates deep understanding of your business, strategic thinking, and execution capabilities. Your response should seamlessly integrate relevant data points, strategic insights, market analysis, competitive positioning, financial metrics, operational details, risk mitigation strategies, and growth projections as appropriate to the question. The answer should be sophisticated and thorough, showing that you have considered multiple dimensions of the issue, understand potential concerns, and have well-reasoned strategies to address challenges while capitalizing on opportunities. Ensure your response flows naturally while providing specific examples, concrete evidence, and actionable insights that would satisfy a sophisticated investor's due diligence requirements and demonstrate your readiness for institutional investment.

Question: {question}
Context: {context}

Return JSON: {{"question": "{question}", "answer": "..."}}
"""
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer_text = response.choices[0].message.content.strip()
            qa_results.append(answer_text)

        # Return proper Python dictionary
        return {"qa_history": qa_results}
