
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool

load_dotenv()

# ==================================================
# TAVILY CLIENT
# ==================================================

def get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is missing. Add it to .env "
            "or Streamlit Cloud Secrets."
        )

    return TavilyClient(api_key=api_key)


# ==================================================
# WEB SEARCH
# ==================================================

@tool
def web_search(query: str) -> str:
    """Search the web for reliable sources.

    Args:
        query: A plain-text search query.
    """
    try:
        query = query.strip()

        if not query:
            return "Search error: query cannot be empty."

        client = get_tavily_client()

        results = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=False,
        )

        output = []

        for index, result in enumerate(
            results.get("results", []), start=1
        ):
            title = result.get("title") or "Untitled"
            url = result.get("url") or ""
            content = result.get("content") or ""

            output.append(
                f"Result {index}\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {content[:250]}"
            )

        if not output:
            return "No search results found."

        return "\n\n".join(output)

    except Exception as error:
        return (
            f"WEB SEARCH ERROR: "
            f"{type(error).__name__}: {error}"
        )


# ==================================================
# WEBPAGE SCRAPER
# ==================================================

@tool
def scrape_url(url: str) -> str:
    """Extract readable text from a webpage.

    Args:
        url: The complete HTTP or HTTPS webpage URL.
    """
    try:
        url = url.strip()

        if not url.startswith(("https://", "http://")):
            return "Scraping error: provide a valid HTTP or HTTPS URL."

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ResearchMind/1.0)"
                )
            },
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "aside", "noscript", "form"
        ]):
            tag.decompose()

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
        ]

        text = " ".join(p for p in paragraphs if p)

        if not text:
            text = soup.get_text(separator=" ", strip=True)

        text = " ".join(text.split())

        return text[:1800] if text else "No readable page text found."

    except requests.exceptions.Timeout:
        return "Scraping error: request timed out."

    except requests.exceptions.RequestException as error:
        return f"Scraping request error: {error}"

    except Exception as error:
        return f"Scraping error: {type(error).__name__}: {error}"
