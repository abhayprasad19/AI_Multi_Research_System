import os
import streamlit as st
import time

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain
)

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.hero { padding-top: 50px; padding-bottom: 30px; }
.hero-eyebrow { color: #38bdf8; font-size: 14px; letter-spacing: 2px; margin-bottom: 12px; font-weight: 600; }
.hero-title { font-size: 72px; font-weight: 800; line-height: 1.1; margin-bottom: 18px; }
.hero-title span { color: #38bdf8; }
.hero-sub { font-size: 18px; color: #cbd5e1; max-width: 750px; line-height: 1.8; }
.divider { height: 1px; background: rgba(255,255,255,0.08); margin-top: 35px; margin-bottom: 35px; }
.input-card { background: #0f172a; padding: 28px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); }
.step-card { background: #111827; padding: 18px; border-radius: 14px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.06); }
.step-header { display: flex; justify-content: space-between; align-items: center; }
.step-num { color: #38bdf8; font-size: 13px; font-weight: bold; }
.step-title { font-size: 18px; font-weight: 600; }
.step-status { font-size: 12px; padding: 5px 12px; border-radius: 999px; }
.status-running { background: orange; color: black; }
.status-done { background: #22c55e; color: white; }
.report-panel { background: #0f172a; padding: 30px; border-radius: 18px; margin-top: 30px; border: 1px solid rgba(255,255,255,0.06); }
.feedback-panel { background: #111827; padding: 30px; border-radius: 18px; margin-top: 30px; border: 1px solid rgba(255,255,255,0.06); }
.notice { text-align: center; margin-top: 50px; margin-bottom: 20px; color: #94a3b8; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="hero-eyebrow">MULTI AGENT AI RESEARCH SYSTEM</div><div class="hero-title">Research<span>Mind</span></div><div class="hero-sub">Autonomous AI research agents that search the web, read sources, analyze information and generate professional research reports using Groq + Tavily.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

left_col, right_col = st.columns([5, 4])

with left_col:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    topic = st.text_input("Research Topic", placeholder="Enter any topic...")
    run_btn = st.button("🚀 Generate Research Report", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="step-card"><div class="step-header"><div class="step-num">STEP 01</div><div class="step-title">Search Agent</div><div class="step-status status-done">READY</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="step-card"><div class="step-header"><div class="step-num">STEP 02</div><div class="step-title">Reader Agent</div><div class="step-status status-done">READY</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="step-card"><div class="step-header"><div class="step-num">STEP 03</div><div class="step-title">Writer Agent</div><div class="step-status status-done">READY</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="step-card"><div class="step-header"><div class="step-num">STEP 04</div><div class="step-title">Critic Agent</div><div class="step-status status-done">READY</div></div></div>', unsafe_allow_html=True)

if run_btn and topic:

    search_agent = build_search_agent()
    reader_agent = build_reader_agent()

    with st.spinner("🔍 Searching web..."):
        search_result = search_agent.invoke({
            "messages": [("user", f"Find detailed information about {topic}")]
        })
    st.success("Search completed")

    with st.spinner("📄 Reading sources..."):
        reader_result = reader_agent.invoke({
            "messages": [("user", f"Read and summarize information about {topic}")]
        })
    st.success("Reading completed")

    research_data = f"""
SEARCH RESULTS:
{search_result}

READER RESULTS:
{reader_result}
"""

    with st.spinner("✍️ Writing report..."):
        report = writer_chain.invoke({
            "topic": topic,
            "research": research_data
        })
    st.success("Report generated")

    with st.spinner("🧠 Critiquing report..."):
        feedback = critic_chain.invoke({
            "report": report
        })
    st.success("Critique completed")

    st.markdown('<div class="report-panel">', unsafe_allow_html=True)
    st.subheader("📘 Research Report")
    st.markdown(report)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feedback-panel">', unsafe_allow_html=True)
    st.subheader("🧠 AI Critic Feedback")
    st.markdown(feedback)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="notice">ResearchMind · Powered by Groq + Tavily + LangChain</div>', unsafe_allow_html=True)