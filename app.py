
import os
import streamlit as st

# ==============================
# ENVIRONMENT CONFIGURATION
# ==============================

from dotenv import load_dotenv

load_dotenv()

# Load secrets safely from Streamlit or .env
for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "GROQ_API_KEY"):
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None

    if value:
        os.environ[key] = value

# ==============================
# AGENT IMPORTS
# ==============================

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

# ==============================
# CUSTOM STYLING
# ==============================

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.hero {
    padding-top: 50px;
    padding-bottom: 30px;
}
.hero-eyebrow {
    color: #38bdf8;
    font-size: 14px;
    letter-spacing: 2px;
    margin-bottom: 12px;
    font-weight: 600;
}
.hero-title {
    font-size: 72px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 18px;
}
.hero-title span {
    color: #38bdf8;
}
.hero-sub {
    font-size: 18px;
    color: #cbd5e1;
    max-width: 750px;
    line-height: 1.8;
}
.divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin-top: 35px;
    margin-bottom: 35px;
}
.input-card {
    background: #0f172a;
    padding: 28px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}
.step-card {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.06);
}
.step-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.step-num {
    color: #38bdf8;
    font-size: 13px;
    font-weight: bold;
}
.step-title {
    font-size: 18px;
    font-weight: 600;
}
.step-status {
    font-size: 12px;
    padding: 5px 12px;
    border-radius: 999px;
}
.status-running {
    background: orange;
    color: black;
}
.status-done {
    background: #22c55e;
    color: white;
}
.report-panel {
    background: #0f172a;
    padding: 30px;
    border-radius: 18px;
    margin-top: 30px;
    border: 1px solid rgba(255,255,255,0.06);
}
.feedback-panel {
    background: #111827;
    padding: 30px;
    border-radius: 18px;
    margin-top: 30px;
    border: 1px solid rgba(255,255,255,0.06);
}
.notice {
    text-align: center;
    margin-top: 50px;
    margin-bottom: 20px;
    color: #94a3b8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# PAGE HEADER
# ==============================

st.markdown(
    '<div class="hero">'
    '<div class="hero-eyebrow">MULTI AGENT AI RESEARCH SYSTEM</div>'
    '<div class="hero-title">Research<span>Mind</span></div>'
    '<div class="hero-sub">Autonomous AI research agents that search the web, '
    'read sources, analyze information and generate professional research '
    'reports using Groq + Tavily.</div></div>',
    unsafe_allow_html=True
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

left_col, right_col = st.columns([5, 4])

with left_col:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    topic = st.text_input(
        "Research Topic",
        placeholder="Enter any topic..."
    )
    run_btn = st.button(
        "🚀 Generate Research Report",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    for number, name in [
        ("01", "Search Agent"),
        ("02", "Reader Agent"),
        ("03", "Writer Agent"),
        ("04", "Critic Agent")
    ]:
        st.markdown(
            f'<div class="step-card"><div class="step-header">'
            f'<div class="step-num">STEP {number}</div>'
            f'<div class="step-title">{name}</div>'
            f'<div class="step-status status-done">READY</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

# ==============================
# HELPER: EXTRACT READABLE TEXT
# ==============================

def extract_text(result):
    """Extract readable content from a LangChain agent result."""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        parts = []

        for message in messages:
            content = getattr(message, "content", None)

            if isinstance(content, str) and content.strip():
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(item["text"])

        if parts:
            return "\n\n".join(parts)

    return str(result)

# ==============================
# RESEARCH WORKFLOW
# ==============================

if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        try:
            search_agent = build_search_agent()
            reader_agent = build_reader_agent()

            with st.spinner("🔍 Searching web..."):
                search_result = search_agent.invoke({
                    "messages": [
                        (
                            "user",
                            f"Find reliable web sources and key facts about "
                            f"{topic}. Return a concise summary with source URLs."
                        )
                    ]
                })
            st.success("Search completed")

            with st.spinner("📄 Reading sources..."):
                reader_result = reader_agent.invoke({
                    "messages": [
                        (
                            "user",
                            f"Read available sources about {topic}. "
                            "Return the most important facts and source URLs "
                            "in a concise summary."
                        )
                    ]
                })
            st.success("Reading completed")

            # Extract readable message text rather than serializing
            # entire agent objects and their metadata.
            search_text = extract_text(search_result)
            reader_text = extract_text(reader_result)

            # Keep the total context modest for Groq's token limits.
            search_text = search_text[:2500]
            reader_text = reader_text[:3500]

            research_data = (
                f"SEARCH RESULTS:\n{search_text}\n\n"
                f"READER RESULTS:\n{reader_text}"
            )

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

            st.markdown(
                '<div class="report-panel">',
                unsafe_allow_html=True
            )
            st.subheader("📘 Research Report")
            st.markdown(report)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="feedback-panel">',
                unsafe_allow_html=True
            )
            st.subheader("🧠 AI Critic Feedback")
            st.markdown(feedback)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as error:
            st.error(
                "The research workflow could not be completed. "
                "Check the error details below."
            )
            st.code(f"{type(error).__name__}: {error}")

# ==============================
# FOOTER
# ==============================

st.markdown(
    '<div class="notice">ResearchMind · Powered by Groq + Tavily + LangChain</div>',
    unsafe_allow_html=True
)
