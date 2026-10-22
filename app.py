
import re
import streamlit as st

from agents import writer_chain, critic_chain
from tools import web_search, scrape_url


st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 ResearchMind")
st.caption("AI-powered multi-agent research system")

st.markdown(
    "Enter a topic to search the web, read relevant sources, "
    "generate a research report, and evaluate its quality."
)

# Keep the completed report available across Streamlit reruns.
if "research_report" not in st.session_state:
    st.session_state["research_report"] = ""

if "research_topic" not in st.session_state:
    st.session_state["research_topic"] = ""

topic = st.text_input(
    "Research topic",
    placeholder="e.g., Latest developments in artificial intelligence",
)

run_research = st.button(
    "🔍 Start Research",
    type="primary",
    use_container_width=True,
)

# Four agent cards are always displayed together.
st.subheader("Research Workflow")

columns = st.columns(4)
card_titles = [
    "Step 1/4 — Search Agent",
    "Step 2/4 — Reader Agent",
    "Step 3/4 — Writer Agent",
    "Step 4/4 — Critic Agent",
]

card_bodies = []

for column, title in zip(columns, card_titles):
    with column:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            body = st.empty()
            body.info("Waiting for research to start.")
            card_bodies.append(body)


def is_quota_error(error):
    message = str(error).lower()
    indicators = [
        "quota",
        "rate_limit",
        "rate limit",
        "tokens per day",
        "tokens per minute",
        "requests per day",
        "resource_exhausted",
        "429",
    ]
    return any(item in message for item in indicators)


def extract_urls(text):
    """Extract unique HTTP/HTTPS URLs from search results."""
    found = re.findall(r"https?://[^\s<>\"']+", text or "")
    urls = []

    for url in found:
        url = url.rstrip(".,;:)]}")
        if url not in urls:
            urls.append(url)

    return urls


def show_failure(card, error, stage):
    if is_quota_error(error):
        card.info(
            f"{stage} is temporarily unavailable because "
            "the AI provider's quota has been reached."
        )
    else:
        card.error(f"{stage} could not complete. Check the terminal for details.")


if run_research:
    if not topic.strip():
        st.error("Please enter a research topic first.")
    else:
        # Clear the previous result only when a new run starts.
        st.session_state["research_topic"] = topic.strip()
        st.session_state["research_report"] = ""

        search_results = ""
        research_text = ""
        report = ""

        # Step 1: Search
        try:
            card_bodies[0].info("Searching the web...")

            search_results = web_search.invoke(
                {"query": topic.strip()}
            )

            if not isinstance(search_results, str):
                search_results = str(search_results)

            if (
                not search_results.strip()
                or "WEB SEARCH ERROR" in search_results
            ):
                raise RuntimeError(
                    search_results or "The search returned no results."
                )

            card_bodies[0].success("Search completed.")
            with st.expander("View search results"):
                st.text(search_results[:12000])

        except Exception as error:
            card_bodies[0].error("Search failed.")
            st.error(f"Search error: {error}")

        # Step 2: Read the sources
        if search_results:
            try:
                card_bodies[1].info("Reading relevant sources...")

                urls = extract_urls(search_results)[:2]
                pages = []

                for url in urls:
                    try:
                        page_text = scrape_url.invoke({"url": url})

                        if page_text and not page_text.startswith(
                            ("SCRAPING ERROR", "SCRAPING REQUEST ERROR")
                        ):
                            pages.append(
                                f"Source URL: {url}\n"
                                f"Page content:\n{page_text[:2500]}"
                            )
                        else:
                            pages.append(
                                f"Source URL: {url}\n"
                                f"Could not extract page text: "
                                f"{str(page_text)[:300]}"
                            )
                    except Exception as scrape_error:
                        pages.append(
                            f"Source URL: {url}\n"
                            f"Reading failed: {str(scrape_error)[:300]}"
                        )

                research_text = (
                    "\n\n--- SOURCE ---\n\n".join(pages)
                    if pages
                    else search_results[:3500]
                )

                # Keep the input to the Writer small.
                research_text = research_text[:4500]

                card_bodies[1].success(
                    f"Reading finished. Sources processed: {len(urls)}."
                )

                with st.expander("View extracted research"):
                    st.text(research_text[:8000])

            except Exception as error:
                card_bodies[1].error("Reading failed.")
                research_text = search_results[:3500]

                st.info(
                    "Some sources could not be read. "
                    "The Writer can still use the available search results."
                )

        # Step 3: Write the report
        if search_results:
            try:
                card_bodies[2].info("Generating the report...")

                report = writer_chain.invoke({
                    "topic": topic.strip(),
                    "research": research_text or search_results[:3500],
                })

                report = str(report).strip()

                if not report:
                    raise RuntimeError("The Writer returned an empty report.")

                st.session_state["research_report"] = report
                card_bodies[2].success("Report generated.")

            except Exception as error:
                card_bodies[2].error("Report generation failed.")
                show_failure(card_bodies[2], error, "Writer Agent")

                if is_quota_error(error):
                    st.info(
                        "The Writer could not run because the AI quota "
                        "is exhausted. Search results are still available."
                    )
                else:
                    st.error(f"Writer error: {error}")

        # Step 4: Critic — never discard a completed report on failure.
        if report:
            try:
                card_bodies[3].info("Reviewing the report...")

                critique = critic_chain.invoke({
                    "report": report[:4500]
                })

                critique = str(critique).strip()
                card_bodies[3].success("Review completed.")

                st.subheader("📝 Critic Review")
                st.markdown(critique or "The Critic returned no review.")

            except Exception as error:
                card_bodies[3].info(
                    "Critic review unavailable. Your research report "
                    "has been preserved."
                )

                if is_quota_error(error):
                    st.info(
                        "The Critic Agent could not run because the AI "
                        "provider's quota is exhausted. Your report is "
                        "available below; you can review it later."
                    )
                else:
                    st.info(
                        "The Critic Agent could not complete its review. "
                        "Your report is still available."
                    )

                # Do not raise the exception or clear the report.


# Display the report independently of the Critic's outcome.
saved_report = st.session_state.get("research_report", "")

if saved_report:
    st.divider()
    st.subheader("📄 Research Report")
    st.markdown(saved_report)

    safe_topic = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        st.session_state.get("research_topic", "research"),
    ).strip("_")[:60] or "research"

    st.download_button(
        label="⬇️ Download Research Report",
        data=saved_report,
        file_name=f"{safe_topic}_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.divider()
st.caption("ResearchMind · Powered by Groq + Tavily + LangChain")
