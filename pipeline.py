from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain
)

import time


# =========================================================
# RESEARCH PIPELINE
# =========================================================

def run_research_pipeline(topic: str) -> dict:

    state = {}

    print("\n" + "=" * 70)
    print("STEP 1 → SEARCH AGENT")
    print("=" * 70)

    try:

        search_agent = build_search_agent()

        search_result = search_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
Find recent, reliable and detailed information about:

{topic}

Focus on:
- latest developments
- trends
- statistics
- expert insights
- trustworthy sources
"""
                )
            ]
        })

        state["search_results"] = (
            search_result["messages"][-1].content
        )

        print("\nSEARCH COMPLETED")
        print("\nSEARCH RESULTS:\n")
        print(state["search_results"][:1500])

    except Exception as e:

        print("\nSEARCH AGENT ERROR:")
        print(str(e))

        state["search_results"] = "No search results found."

    time.sleep(1)

    # =====================================================
    # STEP 2 → READER AGENT
    # =====================================================

    print("\n" + "=" * 70)
    print("STEP 2 → READER AGENT")
    print("=" * 70)

    try:

        reader_agent = build_reader_agent()

        reader_result = reader_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
Based on the following search results about:

{topic}

Pick the BEST and MOST RELEVANT source URL.

Then scrape and summarize the content.

SEARCH RESULTS:
{state["search_results"][:2000]}
"""
                )
            ]
        })

        state["scraped_content"] = (
            reader_result["messages"][-1].content
        )

        print("\nSCRAPING COMPLETED")
        print("\nSCRAPED CONTENT:\n")
        print(state["scraped_content"][:2000])

    except Exception as e:

        print("\nREADER AGENT ERROR:")
        print(str(e))

        state["scraped_content"] = "No scraped content available."

    time.sleep(1)

    # =====================================================
    # STEP 3 → WRITER AGENT
    # =====================================================

    print("\n" + "=" * 70)
    print("STEP 3 → WRITER AGENT")
    print("=" * 70)

    try:

        research_combined = f"""

SEARCH RESULTS:
{state["search_results"]}


DETAILED SCRAPED CONTENT:
{state["scraped_content"]}

"""

        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined
        })

        print("\nREPORT GENERATED SUCCESSFULLY")
        print("\nFINAL REPORT:\n")
        print(state["report"][:3000])

    except Exception as e:

        print("\nWRITER AGENT ERROR:")
        print(str(e))

        state["report"] = "Failed to generate report."

    time.sleep(1)

    # =====================================================
    # STEP 4 → CRITIC AGENT
    # =====================================================

    print("\n" + "=" * 70)
    print("STEP 4 → CRITIC AGENT")
    print("=" * 70)

    try:

        state["feedback"] = critic_chain.invoke({
            "report": state["report"]
        })

        print("\nCRITIC FEEDBACK:\n")
        print(state["feedback"])

    except Exception as e:

        print("\nCRITIC ERROR:")
        print(str(e))

        state["feedback"] = "Critic review failed."

    # =====================================================
    # PIPELINE FINISHED
    # =====================================================

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    return state


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RESEARCHMIND → MULTI AGENT AI RESEARCH SYSTEM")
    print("=" * 70)

    topic = input("\nEnter a research topic: ")

    final_state = run_research_pipeline(topic)

    print("\n")
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(final_state["report"])

    print("\n")
    print("=" * 70)
    print("AI CRITIC FEEDBACK")
    print("=" * 70)

    print(final_state["feedback"])