
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

load_dotenv()

# ==================================================
# GROQ LLM CONFIGURATION
# ==================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your .env file "
        "or Streamlit Cloud Secrets."
    )

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0,
    max_retries=0,
)

# ==================================================
# SEARCH AGENT
# ==================================================

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a web research assistant. "
            "Search for reliable sources about the user's topic. "
            "Return at most 5 key findings and their source URLs. "
            "Be concise and avoid repeating information."
        ),
    )


# ==================================================
# READER AGENT
# ==================================================

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a research reader. Read available source pages "
            "and extract only relevant facts. Return at most 5 concise "
            "bullet points with source URLs. Do not write a full report. "
            "Avoid unnecessary explanations."
        ),
    )


# ==================================================
# WRITER CHAIN
# ==================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a concise, factual research writer. "
        "Use only the supplied research. Do not invent facts or sources. "
        "Write a useful report without unnecessary repetition."
    ),
    (
        "human",
        """Create a concise research report.

Topic:
{topic}

Research:
{research}

Use these sections:
1. Introduction
2. Key Findings (3 concise bullet points)
3. Conclusion
4. Sources (include URLs supplied in the research)

Keep the report focused and professional.
If the research lacks evidence, state that clearly."""
    ),
])

writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)


# ==================================================
# CRITIC CHAIN
# ==================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a concise and constructive research critic. "
        "Evaluate evidence, clarity, accuracy, and completeness. "
        "Do not rewrite the entire report."
    ),
    (
        "human",
        """Review this research report:

{report}

Return this format:

Score: X/10

Strengths:
- One strength
- Another strength

Areas to Improve:
- One improvement

Verdict:
One concise sentence."""
    ),
])

critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)
