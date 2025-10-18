from core.base_agent import BaseAgent
from openai import OpenAI
from agents.summarization_agent import SummarizationAgent
from agents.market_analysis_agent import MarketAnalysisAgent
from agents.risk_compliance_agent import RiskComplianceAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.financial_analysis_agent import FinancialAnalysisAgent
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import SentenceTransformerEmbeddings

class MCPRouterAgent(BaseAgent):
    def __init__(self, perplexity_api_key: str):
        super().__init__("MCPRouterAgent")
        self.client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

        self.agents = {
            "summarization": SummarizationAgent(perplexity_api_key),
            "market": MarketAnalysisAgent(perplexity_api_key),
            "risk": RiskComplianceAgent(perplexity_api_key),
            "technical": TechnicalAnalysisAgent(perplexity_api_key),
            "financial": FinancialAnalysisAgent(perplexity_api_key),
        }

        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_index = None
        self.documents = []

    def process(self, task, data):
        if task in self.agents:
            output = self.agents[task].process(data)
            self.documents.append(str(output))
            self.vector_index = FAISS.from_texts(self.documents, self.embedding_model)
            return output
        else:
            raise ValueError(f"Unknown task: {task}")

    def retrieve_context(self, query, top_k=3):
        if self.vector_index is None:
            return ""
        results = self.vector_index.similarity_search(query, k=top_k)
        return " ".join([r.page_content for r in results])

    def decide_agent(self, text):
        prompt = f"""
You are an intelligent MCP routing agent for startup investment analysis. Analyze the content and determine the most appropriate specialized agent based on keywords, context, and domain expertise required.

AGENT SPECIALIZATIONS:
- summarization: Company overview, executive summary, general pitch content, business description
- market: Market size, TAM/SAM/SOM, competitors, industry analysis, customer segments, market trends
- risk: Legal compliance, regulatory issues, operational risks, security concerns, business risks
- technical: Technology stack, architecture, scalability, AI/ML, engineering, product development
- financial: Revenue models, burn rate, funding, valuation, financial projections, unit economics

ROUTING DECISION:
Analyze the content focus and select the agent with the most relevant expertise.
For multi-domain content, choose the primary domain that requires specialized analysis.

Content to route:
{text}

Return only the agent name (summarization, market, risk, technical, financial).
"""
        response = self.client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        decision = response.choices[0].message.content.strip().lower()
        return decision if decision in self.agents else "summarization"
