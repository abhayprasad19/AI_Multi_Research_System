
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=350,
    max_retries=0,
)

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a concise web research agent. "
            "Call web_search at most once with exactly "
            "one argument: query, containing plain search terms. "
            "Never use cursor, id, or pagination arguments. "
            "Return key findings and source URLs briefly."
        ),
    )

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a concise research reader. "
            "Call scrape_url with exactly one argument: url, "
            "containing a complete HTTP or HTTPS URL. "
            "Return at most four key facts. "
            "Never use cursor, id, or pagination arguments."
        ),
    )

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Write concise, factual reports. Never invent facts or URLs."
    ),
    (
        "human",
        """Topic: {topic}

Research:
{research}

Write a report with:
1. Introduction
2. Three key findings
3. Conclusion
4. Source URLs

Maximum 250 words."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Evaluate reports briefly and constructively."
    ),
    (
        "human",
        """Report:
{report}

Return:
Score: X/10
Strengths: two brief bullets
Improvements: two brief bullets
Verdict: one sentence."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
