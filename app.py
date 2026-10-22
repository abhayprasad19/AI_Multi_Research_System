
import os
import time
import streamlit as st
from dotenv import load_dotenv

# ==================================================
# ENVIRONMENT CONFIGURATION
# ==================================================

load_dotenv()

for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "GROQ_API_KEY"):
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None

    if value:
        os.environ[key] = value

# ==================================================
# AGENT IMPORTS
# ==================================================

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
)

# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background-color: #050816;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    .hero {
        padding-top: 35px;
        padding-bottom: 25px;
    }

    .hero-eyebrow {
        color: #38bdf8;
        font-size: 14px;
        letter-spacing: 2px;
        margin-bottom: 12px;
        font-weight: 600;
    }

    .hero-title {
        font-size: clamp(42px, 7vw, 72px);
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
        margin: 30px 0;
    }

    .step-card {
        background: #111827;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .step-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
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
        background: #22c55e;
        color: white;
        white-space: nowrap;
    }

    .report-panel, .feedback-panel {
        background: #0f172a;
        padding: 24px;
        border-radius: 18px;
        margin-top: 24px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .notice {
        text-align: center;
        margin-top: 45px;
        margin-bottom: 20px;
        color: #94a3b8;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">
            MULTI AGENT AI RESEARCH SYSTEM
        </div>
        <div class="hero-title">
            Research<span>Mind</span>
        </div>
        <div class="hero-sub">
            Autonomous AI research agents that search the web,
            read sources, analyze information and generate
            professional research reports using Groq + Tavily.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================================================
# MAIN INTERFACE
# ==================================================

left_col, right_col = st.columns([5, 4])

with left_col:
    topic = st.text_input(
        "Research Topic",
        placeholder="Enter any topic...",
        max_chars=200,
    )

    run_btn = st.button(
        "🚀 Generate Research Report",
        use_container_width=True,
        type="primary",
    )

with right_col:
    for number, name in [
        ("01", "Search Agent"),
        ("02", "Reader Agent"),
        ("03", "Writer Agent"),
        ("04", "Critic Agent"),
    ]:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-header">
                    <div class="step-num">STEP {number}</div>
                    <div class="step-title">{name}</div>
                    <div class="step-status">READY</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==================================================
# HELPERS
# ==================================================

def extract_text(result):
    """Extract readable text from LangChain agent results."""
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        messages = result.get("messages", [])
        parts = []

        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
            else:
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


def invoke_with_retry(chain, inputs, max_retries=2):
    """Retry temporary rate limits with increasing delays."""
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(inputs)

        except Exception as error:
            message = str(error).lower()

            rate_limited = (
                "429" in message
                or "rate_limit_exceeded" in message
                or "rate limit reached" in message
            )

            if not rate_limited or attempt == max_retries:
                raise

            delay = 7 * (attempt + 1)

            st.warning(
                f"Rate limit reached. Retrying in {delay} seconds..."
            )
            time.sleep(delay)


# ==================================================
# RESEARCH WORKFLOW
# ==================================================

if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")

    else:
        try:
            # STEP 1: SEARCH
            with st.spinner("🔍 Search Agent is searching..."):
                search_agent = build_search_agent()

                search_result = invoke_with_retry(
                    search_agent,
                    {
                        "messages": [
                            (
                                "user",
                                f"Research {topic}. Find a few reliable "
                                "sources and return the main findings "
                                "with source URLs. Keep the response short."
                            )
                        ]
                    },
                )

            st.success("✅ Search completed")

            # Extract the search result before the next step.
            search_text = extract_text(search_result)[:1000]

            # STEP 2: READ
            with st.spinner("📄 Reader Agent is analyzing sources..."):
                reader_agent = build_reader_agent()

                reader_result = invoke_with_retry(
                    reader_agent,
                    {
                        "messages": [
                            (
                                "user",
                                f"Summarize {topic} in no more than "
                                "5 short bullet points. Include the most "
                                "important facts and available source URLs. "
                                "Avoid lengthy explanations."
                            )
                        ]
                    },
                )

            st.success("✅ Reading completed")

            reader_text = extract_text(reader_result)[:1200]

            # STEP 3: WRITE
            research_data = (
                f"SEARCH FINDINGS:\n{search_text}\n\n"
                f"READER FINDINGS:\n{reader_text}"
            )

            with st.spinner("✍️ Writer Agent is generating the report..."):
                report = invoke_with_retry(
                    writer_chain,
                    {
                        "topic": topic[:200],
                        "research": research_data,
                    },
                )

            report = extract_text(report)
            st.success("✅ Report generated")

            # STEP 4: CRITIQUE
            with st.spinner("🧠 Critic Agent is reviewing the report..."):
                feedback = invoke_with_retry(
                    critic_chain,
                    {
                        "report": report[:2500],
                    },
                )

            feedback = extract_text(feedback)
            st.success("✅ Critique completed")

            # DISPLAY REPORT
            st.markdown(
                '<div class="report-panel">',
                unsafe_allow_html=True,
            )
            st.subheader("📘 Research Report")
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)

            # DISPLAY CRITIC FEEDBACK
            st.markdown(
                '<div class="feedback-panel">',
                unsafe_allow_html=True,
            )
            st.subheader("🧠 AI Critic Feedback")
            st.markdown(feedback)
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Research Report",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        except Exception as error:
            error_message = str(error).lower()

            st.error(
                "The research workflow could not be completed."
            )

            if (
                "429" in error_message
                or "rate_limit_exceeded" in error_message
                or "rate limit reached" in error_message
            ):
                st.warning(
                    "Groq's token rate limit is still active. "
                    "Wait before retrying. If this continues, the "
                    "model configuration or agent prompts in "
                    "agents.py must be optimized."
                )

            st.code(f"{type(error).__name__}: {error}")

# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="notice">
        ResearchMind · Powered by Groq + Tavily + LangChain
    </div>
    """,
    unsafe_allow_html=True,
)
