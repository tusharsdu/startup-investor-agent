from core.base_agent import BaseAgent
import os
import openai

class ReportGeneratorAgent(BaseAgent):
    def __init__(self, report_dir="reports", openai_api_key=None):
        super().__init__("ReportGeneratorAgent")
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key

    def generate_narrative(self, section_title: str, data: dict) -> str:
        """
        Use OpenAI to generate a readable paragraph from the data.
        """
        if not data:
            return f"<h2>{section_title}</h2><p>No information available.</p>"

        prompt = f"""
Create a concise paragraph for "{section_title}" from this data:

{data}

Write in professional English for an investment report.
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            narrative = response.choices[0].message.content.strip()
        except Exception as e:
            narrative = f"<p>Error generating narrative: {e}</p>"

        return f"<h2>{section_title}</h2><p>{narrative}</p>"

    def process(self, analysis: dict, verdict: dict, filename="investment_report.html") -> str:
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Startup Investment Report</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 40px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 20px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 5px;
        }}
        p {{
            line-height: 1.6;
            margin: 10px 0;
        }}
        .verdict {{
            font-size: 1.2em;
            font-weight: bold;
            color: #e74c3c;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #2980b9;
            color: white;
        }}
        details {{
            margin-top: 10px;
        }}
        summary {{
            cursor: pointer;
            font-weight: bold;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Startup Investment Report</h1>

    {self.generate_narrative("Pitch Deck Summary", analysis.get("summarization", {}))}
    {self.generate_narrative("Market Analysis", analysis.get("market", {}))}
    {self.generate_narrative("Technical Analysis", analysis.get("technical", {}))}
    {self.generate_narrative("Risk & Compliance Analysis", analysis.get("risk", {}))}
    {self.generate_narrative("Financial Analysis", analysis.get("financial", {}))}

    <div class="card">
        <h2>VC Q&A History</h2>
        <details>
            <summary>Click to expand Q&A</summary>
            {self.generate_narrative("Q&A", analysis.get("vc_qa", {}))}
        </details>
    </div>

    <div class="card">
        <h2>Investment Verdict</h2>
        <p class="verdict">{verdict.get("verdict", "Unknown")} - Confidence: {verdict.get("confidence", 0):.2f}</p>
        <table>
            <tr>
                <th>Recommended Investment (USD)</th>
                <td>{verdict.get("recommended_investment_usd", 0)}</td>
            </tr>
            <tr>
                <th>Recommended Equity (%)</th>
                <td>{verdict.get("recommended_equity_percent", 0)}</td>
            </tr>
            <tr>
                <th>Justification</th>
                <td>{verdict.get("justification", "")}</td>
            </tr>
        </table>
    </div>

</div>
</body>
</html>
"""
        file_path = os.path.join(self.report_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return file_path
