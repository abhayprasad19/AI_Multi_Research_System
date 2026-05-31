from langchain.tools import tool

import requests
from bs4 import BeautifulSoup

from tavily import TavilyClient

import os
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# TAVILY CLIENT
# =========================================================

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# =========================================================
# WEB SEARCH TOOL
# =========================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.

    Returns:
    - Titles
    - URLs
    - Snippets
    """

    try:

        results = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        out = []

        for idx, r in enumerate(results["results"], start=1):

            title = r.get("title", "No Title")
            url = r.get("url", "No URL")
            content = r.get("content", "No Content")

            out.append(
                f"""
RESULT {idx}

Title:
{title}

URL:
{url}

Snippet:
{content[:500]}
"""
            )

        return "\n" + ("\n" + "=" * 60 + "\n").join(out)

    except Exception as e:

        return f"""
WEB SEARCH ERROR

{str(e)}
"""


# =========================================================
# SCRAPE URL TOOL
# =========================================================

@tool
def scrape_url(url: str) -> str:
    """
    Scrape and clean text content from a webpage.
    """

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            timeout=10,
            headers=headers
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unwanted elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "noscript"
        ]):
            tag.decompose()

        # Extract paragraphs
        paragraphs = soup.find_all("p")

        text = "\n".join(
            [
                p.get_text(strip=True)
                for p in paragraphs
                if p.get_text(strip=True)
            ]
        )

        # Fallback if paragraphs empty
        if not text:
            text = soup.get_text(
                separator=" ",
                strip=True
            )

        cleaned_text = " ".join(text.split())

        return cleaned_text[:5000]

    except requests.exceptions.Timeout:

        return "SCRAPING ERROR: Request timed out."

    except requests.exceptions.RequestException as e:

        return f"SCRAPING REQUEST ERROR:\n{str(e)}"

    except Exception as e:

        return f"SCRAPING ERROR:\n{str(e)}"