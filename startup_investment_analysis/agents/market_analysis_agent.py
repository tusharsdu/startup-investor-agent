from core.base_agent import BaseAgent
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

class MarketAnalysisAgent(BaseAgent):
    def __init__(self, perplexity_api_key: str):
        super().__init__("MarketAnalysisAgent")
        self.client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_index = None

    def process(self, pitch_text):
        prompt = f"""
You are a senior market research expert with deep expertise in startup market analysis and competitive intelligence. Deliver a comprehensive market analysis in one elaborate, detailed paragraph that thoroughly explores the market landscape, opportunity assessment, and competitive dynamics. Your analysis should seamlessly weave together total addressable market calculations with supporting data sources, serviceable addressable market segmentation, competitive landscape mapping including direct and indirect competitors with their market positioning and strategic advantages, market growth trajectories with historical and projected data, customer demographic analysis and buying behavior patterns, emerging market trends and technological disruptions that create opportunities or threats, regulatory environment impact and compliance requirements, market timing assessment and adoption cycles, barriers to entry and competitive moat analysis, market maturity evaluation and saturation risks, geographic expansion opportunities and international market considerations, partnership ecosystem and strategic alliance potential, market validation evidence and customer traction indicators, pricing dynamics and value proposition differentiation, and strategic market positioning recommendations. Ensure your response flows naturally while incorporating specific market data, competitive benchmarking, industry insights, and strategic recommendations that demonstrate sophisticated market intelligence capabilities for senior investment decision-making.

Market Research Target:
{pitch_text}
"""
        response = self.client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        result_text = response.choices[0].message.content.strip()

        # Create vector index for semantic search
        self.vector_index = FAISS.from_texts([result_text], self.embedding_model)

        # Fixed: single curly braces for returning dict
        return {"market_analysis": result_text}
