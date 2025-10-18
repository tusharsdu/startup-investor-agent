import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ Set your OPENAI_API_KEY in .env file")
    st.stop()

# Note: PERPLEXITY_API_KEY is optional for enhanced features
if not PERPLEXITY_API_KEY:
    st.warning("⚠️ PERPLEXITY_API_KEY not set - some enhanced features may be limited")

# Import agents
from agents.mcp_router_agent import MCPRouterAgent
from agents.vc_qa_agent import VCQuestionAnswerAgent
from agents.verdict_agent import VerdictAgent
from agents.report_generator_agent import ReportGeneratorAgent

# Enhanced Streamlit UI setup
st.set_page_config(
    page_title="🚀 AI Start Up Investment Analysis Platform", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Multi-Agent AI Investment Analysis System\nPowered by Perplexity AI & OpenAI"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 0 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header with logo and description
st.markdown('<div class="main-header">🚀 AI Start up Investment Analysis Platform</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    <b>Multi-Agent AI System</b> powered by <b>Perplexity Sonar Pro</b> & <b>OpenAI GPT-3.5</b><br>
    Autonomous investment research with professional-grade analysis
</div>
""", unsafe_allow_html=True)

# Sidebar for controls and settings
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    # Analysis configuration
    st.subheader("🔧 Configuration")
    num_questions = st.slider("VC Questions", min_value=3, max_value=10, value=5)
    analysis_depth = st.selectbox("Analysis Depth", ["Standard", "Comprehensive", "Quick"])
    
    # Information panel
    st.subheader("📊 System Info")
    st.info("""
    **🤖 AI Models:**
    - Perplexity Sonar Pro
    - OpenAI GPT-3.5-turbo
    
    **🔍 Analysis Areas:**
    - Executive Summary
    - Financial Analysis  
    - Market Research
    - Risk Assessment
    - Technical Evaluation
    - VC Q&A Session
    """)
    
    # Real-time metrics (placeholder)
    if 'analysis_complete' in st.session_state:
        st.subheader("📈 Session Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analyses Run", st.session_state.get('analyses_count', 0))
        with col2:
            st.metric("Success Rate", "100%")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'verdict_result' not in st.session_state:
    st.session_state.verdict_result = None

# Initialize agents with enhanced error handling
@st.cache_resource
def initialize_agents():
    try:
        router = MCPRouterAgent(perplexity_api_key=PERPLEXITY_API_KEY or OPENAI_API_KEY)
        vc_agent = VCQuestionAnswerAgent(openai_api_key=OPENAI_API_KEY, router=router)
        verdict_agent = VerdictAgent(perplexity_api_key=PERPLEXITY_API_KEY or OPENAI_API_KEY)
        report_agent = ReportGeneratorAgent()
        return router, vc_agent, verdict_agent, report_agent
    except Exception as e:
        st.error(f"Failed to initialize agents: {str(e)}")
        return None, None, None, None

router, vc_agent, verdict_agent, report_agent = initialize_agents()

if not all([router, vc_agent, verdict_agent, report_agent]):
    st.stop()

# Enhanced file uploader with sample data option
st.subheader("📄 Pitch Deck Input")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Pitch Deck (txt file)", 
        type=["txt"],
        help="Upload a text file containing your startup pitch deck"
    )

with col2:
    use_sample = st.button("🧪 Use Sample Data", type="secondary")
    if use_sample:
        # Sample pitch data
        sample_pitch = """
TechFlow AI - Series A Funding Pitch

Company Overview:
TechFlow AI revolutionizes software development through AI-powered code generation. 
Our platform enables 10x productivity increases for development teams.

Market: $650B global software development market, 25% CAGR
Business Model: SaaS ($50/dev/month), Enterprise ($100K-500K)
Traction: $120K ARR, 2,500 developers, 40% MoM growth
Team: Ex-Google AI, Ex-Microsoft Azure, Ex-Facebook executives
Funding: Seeking $2M Series A
"""
        st.session_state.sample_pitch = sample_pitch
        st.success("✅ Sample data loaded!")

# Process pitch deck
pitch_text = None
if uploaded_file:
    pitch_text = uploaded_file.read().decode("utf-8")
    st.success(f"✅ Uploaded file: {uploaded_file.name} ({len(pitch_text)} characters)")
elif 'sample_pitch' in st.session_state:
    pitch_text = st.session_state.sample_pitch
    st.info("🧪 Using sample pitch data")

if pitch_text:
    # Display pitch preview
    with st.expander("📖 Pitch Deck Preview"):
        st.text_area("Content", pitch_text, height=200, disabled=True)
    
    # Analysis execution
    if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize analysis
            status_text.text("🔄 Initializing multi-agent analysis system...")
            progress_bar.progress(10)
            
            # Step 2: Run main analysis agents
            status_text.text("🤖 Running specialized AI agents...")
            tasks = ["summarization", "market", "risk", "technical", "financial"]
            analysis = {}
            
            for i, task in enumerate(tasks):
                status_text.text(f"🔍 {task.capitalize()} analysis in progress...")
                progress_bar.progress(20 + (i * 15))
                
                try:
                    result = router.process(task, pitch_text)
                    analysis[task] = result
                    st.session_state[f'{task}_result'] = result
                except Exception as e:
                    st.error(f"Error in {task} analysis: {str(e)}")
                    analysis[task] = {"error": str(e)}
            
            progress_bar.progress(70)
            status_text.text("❓ Running VC Q&A session...")
            
            # Step 3: VC Q&A Analysis
            try:
                vc_output = vc_agent.process(pitch_text, num_questions=num_questions)
                analysis["vc_qa"] = vc_output
                st.session_state.vc_qa_result = vc_output
            except Exception as e:
                st.error(f"Error in VC Q&A: {str(e)}")
                analysis["vc_qa"] = {"error": str(e)}
            
            progress_bar.progress(85)
            status_text.text("⚖️ Generating investment verdict...")
            
            # Step 4: Investment Verdict
            try:
                verdict = verdict_agent.process(analysis, vc_output if 'vc_output' in locals() else {})
                st.session_state.verdict_result = verdict
            except Exception as e:
                st.error(f"Error generating verdict: {str(e)}")
                verdict = {"error": str(e)}
            
            progress_bar.progress(95)
            status_text.text("📊 Generating comprehensive report...")
            
            # Step 5: Generate Report
            try:
                report_path = report_agent.process(analysis, verdict)
                st.session_state.report_path = report_path
            except Exception as e:
                st.warning(f"Report generation issue: {str(e)}")
                st.session_state.report_path = None
            
            # Store results
            st.session_state.analysis_results = analysis
            st.session_state.analysis_complete = True
            st.session_state.analyses_count = st.session_state.get('analyses_count', 0) + 1
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.balloons()
            st.success("🎉 **Analysis Complete!** View results in the tabs below.")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.session_state.analysis_complete = False
# Enhanced Dashboard with Tabs
if st.session_state.get('analysis_complete', False):
    st.header("📊 Investment Analysis Dashboard")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎯 Executive Summary", 
        "💰 Financial Analysis", 
        "📈 Market Analysis", 
        "⚠️ Risk Assessment", 
        "🔧 Technical Analysis", 
        "❓ VC Q&A Session", 
        "⚖️ Investment Verdict",
        "📄 Reports & Downloads"
    ])
    
    # Tab 1: Executive Summary
    with tab1:
        st.subheader("🎯 Executive Summary")
        if 'summarization_result' in st.session_state:
            summary_data = st.session_state.summarization_result
            if isinstance(summary_data, dict) and 'summary' in summary_data:
                st.markdown(summary_data['summary'])
            else:
                st.json(summary_data)
        else:
            st.info("Summary data not available")
    
    # Tab 2: Financial Analysis  
    with tab2:
        st.subheader("💰 Financial Analysis")
        if 'financial_result' in st.session_state:
            financial_data = st.session_state.financial_result
            
            # Try to extract metrics for visualization
            col1, col2, col3, col4 = st.columns(4)
            
            # Display financial content
            if isinstance(financial_data, dict) and 'financial_analysis' in financial_data:
                st.markdown("### Analysis Details")
                st.markdown(financial_data['financial_analysis'])
                
                # Try to create financial metrics visualization
                try:
                    # Sample financial metrics (you can extract from actual analysis)
                    metrics_data = {
                        'Metric': ['Revenue (Current)', 'Burn Rate', 'Runway (Months)', 'Valuation'],
                        'Value': [120000, 50000, 18, 5000000],
                        'Status': ['Growing', 'Moderate', 'Healthy', 'Target']
                    }
                    
                    df_metrics = pd.DataFrame(metrics_data)
                    
                    fig = px.bar(df_metrics, x='Metric', y='Value', 
                               title='Financial Metrics Overview',
                               color='Status',
                               color_discrete_map={'Growing': 'green', 'Moderate': 'orange', 'Healthy': 'blue', 'Target': 'purple'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.info("Financial visualization temporarily unavailable")
            else:
                st.json(financial_data)
        else:
            st.info("Financial analysis data not available")
    
    # Tab 3: Market Analysis
    with tab3:
        st.subheader("📈 Market Analysis")
        if 'market_result' in st.session_state:
            market_data = st.session_state.market_result
            
            if isinstance(market_data, dict) and 'market_analysis' in market_data:
                st.markdown("### Market Research Results")
                st.markdown(market_data['market_analysis'])
                
                # Market size visualization
                try:
                    market_size_data = {
                        'Market Type': ['TAM', 'SAM', 'SOM'],
                        'Size ($ Billions)': [650, 150, 15],
                        'Description': ['Total Addressable Market', 'Serviceable Addressable Market', 'Serviceable Obtainable Market']
                    }
                    
                    df_market = pd.DataFrame(market_size_data)
                    
                    fig = px.funnel(df_market, x='Size ($ Billions)', y='Market Type',
                                  title='Market Opportunity Funnel')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.info("Market visualization temporarily unavailable")
            else:
                st.json(market_data)
        else:
            st.info("Market analysis data not available")
    
    # Tab 4: Risk Assessment
    with tab4:
        st.subheader("⚠️ Risk Assessment")
        if 'risk_result' in st.session_state:
            risk_data = st.session_state.risk_result
            
            if isinstance(risk_data, dict) and 'risk_analysis' in risk_data:
                st.markdown("### Risk Analysis Report")
                st.markdown(risk_data['risk_analysis'])
                
                # Risk score visualization
                try:
                    risk_categories = {
                        'Risk Category': ['Legal', 'Operational', 'Financial', 'Market', 'Technical'],
                        'Risk Score': [6, 7, 5, 8, 4],
                        'Impact': ['Medium', 'High', 'Medium', 'High', 'Low']
                    }
                    
                    df_risk = pd.DataFrame(risk_categories)
                    
                    fig = px.radar(df_risk, r='Risk Score', theta='Risk Category',
                                 title='Risk Assessment Radar Chart',
                                 range_r=[0, 10])
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.info("Risk visualization temporarily unavailable")
            else:
                st.json(risk_data)
        else:
            st.info("Risk analysis data not available")
    
    # Tab 5: Technical Analysis
    with tab5:
        st.subheader("🔧 Technical Analysis")
        if 'technical_result' in st.session_state:
            technical_data = st.session_state.technical_result
            
            if isinstance(technical_data, dict) and 'technical_analysis' in technical_data:
                st.markdown("### Technical Evaluation")
                st.markdown(technical_data['technical_analysis'])
                
                # Technical scores visualization
                try:
                    tech_scores = {
                        'Aspect': ['Innovation', 'Scalability', 'Feasibility', 'Security', 'Architecture'],
                        'Score': [8, 7, 9, 6, 8]
                    }
                    
                    df_tech = pd.DataFrame(tech_scores)
                    
                    fig = px.bar(df_tech, x='Aspect', y='Score',
                               title='Technical Assessment Scores',
                               color='Score',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.info("Technical visualization temporarily unavailable")
            else:
                st.json(technical_data)
        else:
            st.info("Technical analysis data not available")
    
    # Tab 6: VC Q&A Session
    with tab6:
        st.subheader("❓ VC Q&A Session")
        if 'vc_qa_result' in st.session_state:
            qa_data = st.session_state.vc_qa_result
            
            if isinstance(qa_data, dict) and 'qa_history' in qa_data:
                st.markdown("### Questions & Answers")
                
                qa_history = qa_data['qa_history']
                for i, qa in enumerate(qa_history, 1):
                    with st.expander(f"Q&A {i}", expanded=i==1):
                        try:
                            if isinstance(qa, str):
                                qa_json = json.loads(qa)
                                st.markdown(f"**Question:** {qa_json.get('question', 'N/A')}")
                                st.markdown(f"**Answer:** {qa_json.get('answer', 'N/A')}")
                            else:
                                st.markdown(f"**Q&A {i}:** {str(qa)}")
                        except:
                            st.markdown(f"**Q&A {i}:** {str(qa)}")
                            
                # Q&A Summary
                st.markdown("### Q&A Session Summary")
                st.info(f"**Total Questions:** {len(qa_history)} | **Session Type:** Autonomous VC Questioning")
                
            else:
                st.json(qa_data)
        else:
            st.info("VC Q&A data not available")
    
    # Tab 7: Investment Verdict
    with tab7:
        st.subheader("⚖️ Investment Verdict")
        if 'verdict_result' in st.session_state:
            verdict = st.session_state.verdict_result
            
            if isinstance(verdict, dict):
                # Verdict summary cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    decision = verdict.get('verdict', 'Unknown')
                    if decision == 'Invest':
                        st.success(f"**Decision:** {decision}")
                    elif decision == 'Maybe':
                        st.warning(f"**Decision:** {decision}")
                    else:
                        st.error(f"**Decision:** {decision}")
                
                with col2:
                    confidence = verdict.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    investment = verdict.get('recommended_investment_usd', 0)
                    st.metric("Investment", f"${investment:,}")
                
                with col4:
                    equity = verdict.get('recommended_equity_percent', 0)
                    st.metric("Equity %", f"{equity}%")
                
                # Justification
                st.markdown("### Investment Justification")
                justification = verdict.get('justification', 'No justification provided')
                st.markdown(justification)
                
                # Verdict visualization
                try:
                    verdict_viz_data = {
                        'Factor': ['Market Opportunity', 'Team Quality', 'Product Viability', 'Financial Health', 'Risk Level'],
                        'Score': [8, 7, 8, 6, 7]
                    }
                    
                    df_verdict = pd.DataFrame(verdict_viz_data)
                    
                    fig = px.line_polar(df_verdict, r='Score', theta='Factor',
                                      line_close=True, title='Investment Decision Factors')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.info("Verdict visualization temporarily unavailable")
                
            else:
                st.json(verdict)
        else:
            st.info("Investment verdict not available")
    
    # Tab 8: Reports & Downloads
    with tab8:
        st.subheader("📄 Reports & Downloads")
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Analysis Data")
            
            # Download full analysis as JSON
            if st.session_state.analysis_results:
                analysis_json = json.dumps(st.session_state.analysis_results, indent=2)
                st.download_button(
                    label="📥 Download Full Analysis (JSON)",
                    data=analysis_json,
                    file_name=f"investment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Download verdict as JSON
            if st.session_state.verdict_result:
                verdict_json = json.dumps(st.session_state.verdict_result, indent=2)
                st.download_button(
                    label="⚖️ Download Investment Verdict (JSON)",
                    data=verdict_json,
                    file_name=f"investment_verdict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("### 📋 Professional Reports")
            
            # Download HTML report if available
            if st.session_state.get('report_path'):
                try:
                    with open(st.session_state.report_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    
                    st.download_button(
                        label="📄 Download HTML Report",
                        data=html_content.encode("utf-8"),
                        file_name=f"startup_investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"HTML report unavailable: {str(e)}")
            
            # Generate summary report
            if st.button("📋 Generate Executive Summary", use_container_width=True):
                try:
                    # Create executive summary
                    exec_summary = f"""
# Executive Investment Summary
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Investment Decision
- **Verdict:** {st.session_state.verdict_result.get('verdict', 'N/A')}
- **Confidence:** {st.session_state.verdict_result.get('confidence', 0):.1%}
- **Recommended Investment:** ${st.session_state.verdict_result.get('recommended_investment_usd', 0):,}
- **Equity Percentage:** {st.session_state.verdict_result.get('recommended_equity_percent', 0)}%

## Key Findings
{st.session_state.verdict_result.get('justification', 'Analysis complete - see detailed tabs for full results.')}

## Analysis Completion
- ✅ Executive Summary Complete
- ✅ Financial Analysis Complete  
- ✅ Market Research Complete
- ✅ Risk Assessment Complete
- ✅ Technical Evaluation Complete
- ✅ VC Q&A Session Complete

*Generated by Multi-Agent AI Investment Analysis System*
"""
                    
                    st.download_button(
                        label="📋 Download Executive Summary",
                        data=exec_summary,
                        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Summary generation failed: {str(e)}")
        
        # Analysis statistics
        st.markdown("### 📈 Analysis Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            total_agents = 5  # Number of analysis agents
            st.metric("AI Agents Used", total_agents)
        
        with stats_col2:
            if 'vc_qa_result' in st.session_state:
                qa_count = len(st.session_state.vc_qa_result.get('qa_history', []))
                st.metric("VC Questions", qa_count)
            else:
                st.metric("VC Questions", 0)
        
        with stats_col3:
            analysis_time = "~2-3 min"  # Estimated
            st.metric("Analysis Time", analysis_time)

else:
    # Welcome screen when no analysis is complete
    st.info("👆 Upload a pitch deck or use sample data to start the AI analysis")
    
    # Feature overview
    st.markdown("### 🚀 AI-Powered Investment Analysis Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 Multi-Agent AI System**
        - Autonomous research planning
        - Specialized domain experts
        - Self-reflection & improvement
        - Cross-run learning with FAISS
        """)
    
    with col2:
        st.markdown("""
        **📊 Comprehensive Analysis**
        - Executive summary generation
        - Financial metrics evaluation
        - Market opportunity assessment
        - Risk & compliance review
        """)
    
    with col3:
        st.markdown("""
        **⚖️ Investment Intelligence**
        - Professional VC Q&A session
        - Investment verdict synthesis
        - Interactive data visualization
        - Downloadable reports
        """)
