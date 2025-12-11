from typing import Optional, List
import re

from pydantic import BaseModel
from pydantic_ai import Tool, RunContext

from baikpacking.db.db_connection import get_pg_connection
from duckduckgo_search import DDGS

import logging

logger = logging.getLogger(__name__)




def get_article_title(article_id: int) -> Optional[str]:
    """
    Look up the title from the `articles` table given an article_id.

    Assumes schema:
        articles(id SERIAL PRIMARY KEY, title TEXT NOT NULL, ...)
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM articles WHERE id = %s", (article_id,))
            row = cur.fetchone()

    if not row:
        return None
    return row[0]

class EventSearchResult(BaseModel):
    """Single web search result for an event."""

    title: str
    url: str
    snippet: Optional[str] = None


class EventWebSearchOutput(BaseModel):
    """
    Structured web search output for an event based on its article title.
    """

    event_title: str
    search_query: str

    official_url: Optional[str] = None
    dotwatcher_url: Optional[str] = None

    results: List[EventSearchResult]


_TITLE_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _strip_year(title: str) -> str:
    """Remove any year (e.g. '2025') from the title."""
    return _TITLE_YEAR_RE.sub("", title).strip()


def _slugify_for_domain(s: str) -> str:
    """
    'The Land Between' -> 'thelandbetween'
    Used to detect domains like 'thelandbetween.co.uk'.
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _search_on_web(query: str, max_results: int = 8) -> List[EventSearchResult]:
    """DuckDuckGo text search wrapper."""
    results: List[EventSearchResult] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url") or ""
            if not url:
                continue
            results.append(
                EventSearchResult(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("body") or r.get("snippet"),
                )
            )
    return results


def _guess_official_and_dotwatcher(
    event_title: str,
    results: List[EventSearchResult],
) -> tuple[Optional[str], Optional[str]]:
    """
    Guess the official website and DotWatcher URL from search results.
    """
    base_name = _strip_year(event_title)
    slug = _slugify_for_domain(base_name)

    official_url: Optional[str] = None
    dotwatcher_url: Optional[str] = None

    for r in results:
        url = r.url
        domain = re.sub(r"^https?://", "", url).split("/")[0].lower()

        if "dotwatcher.cc" in domain and dotwatcher_url is None:
            dotwatcher_url = url

        # skip socials for "official"
        if any(bad in domain for bad in ("facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com")):
            continue

        # if our slug appears in the domain (minus dots), it's likely official
        if slug and slug in domain.replace(".", ""):
            if official_url is None:
                official_url = url

    # Fallback: first non-social result
    if official_url is None:
        for r in results:
            domain = re.sub(r"^https?://", "", r.url).split("/")[0].lower()
            if any(bad in domain for bad in ("facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com")):
                continue
            official_url = r.url
            break

    return official_url, dotwatcher_url

@Tool
def event_web_search(
    ctx: RunContext,
    article_id: int,
    max_results: int = 8,
) -> EventWebSearchOutput:
    """
    Tool: given an article_id, look up the event title from Postgres and search
    it on the web.
    """
    logger.info("event_web_search called", extra={"article_id": article_id, "max_results": max_results})

    title = get_article_title(article_id)
    if not title:
        logger.warning("No article found in DB for id=%s", article_id)
        raise ValueError(f"No article found with id={article_id}")

    logger.info("event_web_search resolved title=%r", title)

    search_query = f"{title} ultra cycling official site"

    try:
        results = _search_on_web(search_query, max_results=max_results)
        logger.info("event_web_search got %d search results", len(results))
    except Exception as exc:
        logger.exception("Error during web search for %r: %s", search_query, exc)
        return EventWebSearchOutput(
            event_title=title,
            search_query=search_query,
            official_url=None,
            dotwatcher_url=None,
            results=[],
        )

    official_url, dotwatcher_url = _guess_official_and_dotwatcher(
        event_title=title,
        results=results,
    )

    logger.info(
        "event_web_search resolved official_url=%r dotwatcher_url=%r",
        official_url,
        dotwatcher_url,
    )

    return EventWebSearchOutput(
        event_title=title,
        search_query=search_query,
        official_url=official_url,
        dotwatcher_url=dotwatcher_url,
        results=results,
    )
