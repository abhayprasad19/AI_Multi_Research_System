
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
    raise ValueError("GROQ_API_KEY is missing.")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=600,
    max_retries=0,
)

# Search agent: explicitly enforce the tool argument schema.
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
You are a web research agent.

When calling web_search, you MUST provide exactly one argument:
query: a plain string containing the search terms.

Correct call:
web_search(query="latest AI developments")

Never use cursor, id, page, offset, or any other arguments.
Do not paginate. Make at most one web_search call.
After receiving results, summarize the best findings and URLs.
""",
    )

# Reader agent: use only valid URL arguments.
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are a concise research reader.

The scrape_url tool requires exactly one argument:
url: a complete HTTP or HTTPS URL.

Never use cursor, id, or pagination arguments.
Read only relevant URLs and return up to four key facts.
""",
    )

# Writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Write concise, factual reports. Do not invent facts or URLs."
    ),
    (
        "human",
        """Topic: {topic}

Research:
{research}

Write a report containing:
1. Introduction
2. Three key findings
3. Conclusion
4. Sources with URLs

Keep it concise and professional."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Review reports constructively and concisely."
    ),
    (
        "human",
        """Review this report:
{report}

Return:
Score: X/10
Strengths: two bullets
Improvements: one or two bullets
Verdict: one sentence."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
