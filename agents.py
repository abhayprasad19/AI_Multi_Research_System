
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

load_dotenv()

# ==================================================
# GROQ CONFIGURATION
# ==================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to .env "
        "or Streamlit Cloud Secrets."
    )

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=600,
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
            "You are a concise web research agent. "
            "Search for reliable sources about the user's topic. "
            "Return at most 3 useful findings and source URLs. "
            "Avoid repeating information. Use tools only when needed."
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
            "You are a concise research reader. "
            "Read relevant source pages when URLs are available. "
            "Return at most 4 key facts and their source URLs. "
            "Do not write a full report. Avoid unnecessary tool calls."
        ),
    )


# ==================================================
# WRITER CHAIN
# ==================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Write concise, factual reports using the supplied evidence. "
        "Do not invent facts, citations, or source URLs."
    ),
    (
        "human",
        """Topic: {topic}

Research:
{research}

Write a compact report with:
1. Introduction
2. Three key findings
3. Conclusion
4. Sources

Keep it concise, professional, and factual."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ==================================================
# CRITIC CHAIN
# ==================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Evaluate research briefly, fairly, and constructively."
    ),
    (
        "human",
        """Review this report:
{report}

Return:
Score: X/10

Strengths:
- One or two bullets

Improvement:
- One bullet

Verdict:
One sentence."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
