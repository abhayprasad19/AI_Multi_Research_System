
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool

load_dotenv()

# =========================================================
# TAVILY CLIENT
# =========================================================

def get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is missing. Add it to .env "
            "or Streamlit Cloud Secrets."
        )

    return TavilyClient(api_key=api_key)


# =========================================================
# WEB SEARCH TOOL
# =========================================================

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information.

    Args:
        query: A plain-text search query describing the research topic.

    Returns:
        A concise list of titles, URLs, and source snippets.
    """
    try:
        query = query.strip()

        if not query:
            return "Search error: query cannot be empty."

        tavily = get_tavily_client()

        results = tavily.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=False,
        )

        output = []

        for idx, result in enumerate(
            results.get("results", []), start=1
        ):
            title = result.get("title") or "Untitled"
            url = result.get("url") or ""
            content = result.get("content") or ""

            output.append(
                f"Result {idx}\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {content[:300]}"
            )

        if not output:
            return f"No web results found for: {query}"

        return "\n\n".join(output)

    except Exception as error:
        return f"WEB SEARCH ERROR: {type(error).__name__}: {error}"


# =========================================================
# WEBPAGE SCRAPING TOOL
# =========================================================

@tool
def scrape_url(url: str) -> str:
    """Extract readable text from a webpage.

    Args:
        url: The complete HTTP or HTTPS URL of the webpage.

    Returns:
        Cleaned webpage text, limited to 2500 characters.
    """
    try:
        url = url.strip()

        if not url.startswith(("https://", "http://")):
            return "SCRAPING ERROR: Provide a valid HTTP or HTTPS URL."

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchMind/1.0)"
        }

        response = requests.get(
            url,
            timeout=10,
            headers=headers,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(
            ["script", "style", "nav", "footer", "header",
             "aside", "noscript", "form"]
        ):
            tag.decompose()

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
        ]

        text = " ".join(p for p in paragraphs if p)

        if not text:
            text = soup.get_text(separator=" ", strip=True)

        cleaned_text = " ".join(text.split())

        if not cleaned_text:
            return "No readable page text was found."

        return cleaned_text[:2500]

    except requests.exceptions.Timeout:
        return "SCRAPING ERROR: The webpage request timed out."

    except requests.exceptions.RequestException as error:
        return f"SCRAPING REQUEST ERROR: {error}"

    except Exception as error:
        return f"SCRAPING ERROR: {type(error).__name__}: {error}"
