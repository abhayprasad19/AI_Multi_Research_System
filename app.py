
import os
import time
import streamlit as st
from dotenv import load_dotenv

# ==================================================
# ENVIRONMENT CONFIGURATION
# ==================================================

load_dotenv()

for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY"):
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
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
)

# ==================================================
# STYLING
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
        padding: 35px 0 25px;
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
            AI research agents that search the web, analyze findings,
            and generate concise professional research reports.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================================================
# INPUT AND OPTIONS
# ==================================================

left_col, right_col = st.columns([5, 4])

with left_col:
    topic = st.text_input(
        "Research Topic",
        placeholder="Enter any topic...",
        max_chars=200,
    )

    use_reader = st.checkbox(
        "Use Reader Agent for additional analysis (extra API usage)",
        value=False,
    )

    use_critic = st.checkbox(
        "Run AI Critic review (extra API usage)",
        value=False,
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
    """Extract readable text from LangChain results."""
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        parts = []

        for message in result.get("messages", []):
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


def invoke_with_retry(chain, inputs, max_retries=1):
    """Retry rate-limit failures once, with a short delay."""
    warning = st.empty()

    for attempt in range(max_retries + 1):
        try:
            result = chain.invoke(inputs)
            warning.empty()
            return result

        except Exception as error:
            message = str(error).lower()

            rate_limited = (
                "429" in message
                or "rate_limit_exceeded" in message
                or "rate limit reached" in message
            )

            if not rate_limited or attempt == max_retries:
                warning.empty()
                raise

            delay = 10 * (attempt + 1)

            warning.warning(
                f"Temporary rate limit. Retrying in {delay} seconds..."
            )
            time.sleep(delay)

    warning.empty()
    raise RuntimeError("The request could not be completed.")


# ==================================================
# RESEARCH WORKFLOW
# ==================================================

if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")

    else:
        try:
            # ------------------------------------------
            # STEP 1: SEARCH
            # ------------------------------------------

            with st.spinner("🔍 Search Agent is searching..."):
                search_agent = build_search_agent()

                search_result = invoke_with_retry(
                    search_agent,
                    {
                        "messages": [
                            (
                                "user",
                                f"Find up to 3 reliable sources about "
                                f"{topic}. Return key findings and source "
                                "URLs. Keep the answer concise."
                            )
                        ]
                    },
                )

            search_text = extract_text(search_result)[:1200]
            st.success("✅ Search completed")

            # ------------------------------------------
            # STEP 2: OPTIONAL READER
            # ------------------------------------------

            reader_text = ""

            if use_reader:
                with st.spinner("📄 Reader Agent is analyzing sources..."):
                    reader_agent = build_reader_agent()

                    reader_result = invoke_with_retry(
                        reader_agent,
                        {
                            "messages": [
                                (
                                    "user",
                                    f"Summarize {topic} in at most "
                                    "4 concise facts with available URLs. "
                                    "Avoid long explanations."
                                )
                            ]
                        },
                    )

                reader_text = extract_text(reader_result)[:900]
                st.success("✅ Reading completed")
            else:
                st.info("Reader Agent skipped to conserve API quota.")

            # ------------------------------------------
            # STEP 3: WRITER
            # ------------------------------------------

            research_data = f"SEARCH FINDINGS:\n{search_text}"

            if reader_text:
                research_data += f"\n\nREADER FINDINGS:\n{reader_text}"

            with st.spinner("✍️ Writer Agent is preparing the report..."):
                report = invoke_with_retry(
                    writer_chain,
                    {
                        "topic": topic[:200],
                        "research": research_data[:2200],
                    },
                )

            report = extract_text(report)
            st.success("✅ Report generated")

            # ------------------------------------------
            # DISPLAY REPORT
            # ------------------------------------------

            st.markdown(
                '<div class="report-panel">',
                unsafe_allow_html=True,
            )
            st.subheader("📘 Research Report")
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Research Report",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

            # ------------------------------------------
            # STEP 4: OPTIONAL CRITIC
            # ------------------------------------------

            if use_critic:
                with st.spinner("🧠 Critic Agent is reviewing the report..."):
                    feedback = invoke_with_retry(
                        critic_chain,
                        {"report": report[:1800]},
                    )

                feedback = extract_text(feedback)
                st.success("✅ Critique completed")

                st.markdown(
                    '<div class="feedback-panel">',
                    unsafe_allow_html=True,
                )
                st.subheader("🧠 AI Critic Feedback")
                st.markdown(feedback)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("AI Critic skipped to conserve API quota.")

        except Exception as error:
            message = str(error).lower()

            st.error("The research workflow could not be completed.")

            if (
                "429" in message
                or "rate_limit_exceeded" in message
                or "rate limit reached" in message
            ):
                st.warning(
                    "Groq's rate limit is active. Wait for the quota "
                    "to recover before trying again. Avoid repeated "
                    "submissions while the limit is active."
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
