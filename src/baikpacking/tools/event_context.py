from __future__ import annotations

import logging
import re
from typing import Optional, List, Tuple

import anyio
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Tool, RunContext, Agent
from pydantic_ai.models.openai import OpenAIChatModel

from baikpacking.db.db_connection import get_pg_connection

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)

# ---------------- Models ----------------


class EventSearchResult(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None


class Evidence(BaseModel):
    """
    Evidence from a fetched page (NOT a search engine snippet).
    """
    source_url: str
    snippet: Optional[str] = None


class EventContextSummary(BaseModel):
    """
    Structured, evidence-backed context for an ultra-distance cycling event.

    Rules:
    - Numeric fields must only be filled if explicitly supported by page text.
    - If a numeric field is filled, the corresponding evidence must be provided.
    """

    # Quantitative (only fill if explicitly supported by evidence)
    distance_km: Optional[float] = None
    distance_evidence: Optional[Evidence] = None

    total_climbing_m: Optional[int] = None
    climbing_evidence: Optional[Evidence] = None

    # Qualitative (summaries should still come from page text)
    surface: Optional[str] = None
    route_character: Optional[str] = None
    climate_notes: Optional[str] = None
    resupply_notes: Optional[str] = None

    # Hard requirements from official sources (mandatory kit, rules, constraints)
    constraints: List[str] = Field(default_factory=list)

    # Free-text synthesis (must not introduce new facts)
    summary: Optional[str] = None

    @model_validator(mode="after")
    def _require_evidence_for_numbers(self) -> "EventContextSummary":
        if self.distance_km is not None and not self.distance_evidence:
            raise ValueError("distance_km provided without distance_evidence")
        if self.total_climbing_m is not None and not self.climbing_evidence:
            raise ValueError("total_climbing_m provided without climbing_evidence")
        return self


class EventWebContext(BaseModel):
    event_title: str
    search_query: str
    official_url: Optional[str] = None
    dotwatcher_url: Optional[str] = None
    context: Optional[EventContextSummary] = None
    results: List[EventSearchResult] = Field(default_factory=list)


# ---------------- Helpers ----------------

_TITLE_YEAR_RE = re.compile(r"(19|20)\d{2}")
_AGGREGATOR_DOMAINS = (
    "granfondoguide.com",
    "bikepacking.com",
    "amateurultracycling.cc",
    "battistrada.com",
    "cycling-calendar",
)

_COMMON_RULE_PATHS = (
    "/rules",
    "/rule",
    "/regulation",
    "/regulations",
    "/reglament",
    "/reglamento",
    "/regulations/",
    "/route",
    "/routes",
    "/faq",
    "/info",
    "/about",
)


def get_article_title(article_id: int) -> Optional[str]:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM articles WHERE id = %s", (article_id,))
            row = cur.fetchone()
    return row[0] if row else None


def _strip_year(title: str) -> str:
    return _TITLE_YEAR_RE.sub("", title).strip()


def _domain(url: str) -> str:
    return re.sub(r"^https?://", "", url).split("/")[0].lower()


def _is_social(url: str) -> bool:
    d = _domain(url)
    return any(bad in d for bad in ("facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com", "tiktok.com"))


def _looks_like_dotwatcher(url: str) -> bool:
    return "dotwatcher.cc" in _domain(url)


def _is_aggregator(url: str) -> bool:
    d = _domain(url)
    return any(a in d for a in _AGGREGATOR_DOMAINS)


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _base_site(url: str) -> str:
    # https://a.b/c/d -> https://a.b
    m = re.match(r"^(https?://[^/]+)", url)
    return m.group(1) if m else url


def _search_on_web_sync(query: str, max_results: int = 8) -> List[EventSearchResult]:
    if DDGS is None:
        logger.warning("ddgs is not installed; returning empty search results.")
        return []

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


async def _search_on_web(query: str, max_results: int = 8) -> List[EventSearchResult]:
    return await anyio.to_thread.run_sync(_search_on_web_sync, query, max_results)


async def _fetch_page_text(url: str, timeout: int = 10, max_chars: int = 12000) -> Optional[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; baikpacking-agent/1.0)",
        "Accept-Language": "en,en-US;q=0.9,es;q=0.8",
    }
    try:
        async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.find("main") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    return text[:max_chars] if text else None


# ---------------- Official-site selection ----------------


async def _pick_official_url_with_llm(
    event_title: str,
    candidates: List[Tuple[str, str]],  # (url, page_text)
    model_name: str,
) -> Optional[str]:
    """
    Ask an LLM to choose which candidate is most likely the official event website,
    based on the page text.
    """
    model = OpenAIChatModel(model_name)

    class PickResult(BaseModel):
        official_url: Optional[str] = None
        rationale: Optional[str] = None

    agent = Agent(
        model,
        output_type=PickResult,
        system_prompt=(
            "Select the official event website URL.\n"
            "Use only evidence in the provided page texts.\n"
            "Prefer pages containing: registration, rules, route, GPX, FAQ, checkpoints, mandatory kit.\n"
            "Avoid aggregators and calendar listings unless no official site exists.\n"
            "Return JSON."
        ),
    )

    blocks: List[str] = []
    for i, (url, text) in enumerate(candidates, start=1):
        blocks.append(f"[{i}] URL: {url}\nTEXT:\n{text[:2500]}")
    user_prompt = f"Event: {event_title}\n\n" + "\n\n".join(blocks)

    out = await agent.run(user_prompt)
    return out.output.official_url


async def _guess_urls_and_context(
    event_title: str,
    results: List[EventSearchResult],
    context_model: str,
) -> Tuple[Optional[str], Optional[str], Optional[EventContextSummary]]:
    """
    Returns (official_url, dotwatcher_url, context_summary).

    Strategy:
    - Find dotwatcher url if present.
    - Prefer non-social, non-dotwatcher, non-aggregator candidates for official selection.
    - If none, allow aggregators as a fallback.
    - Summarize from an official rules-like page if possible; else from official homepage; else dotwatcher.
    """
    dotwatcher_url = next((r.url for r in results if _looks_like_dotwatcher(r.url)), None)

    primary_candidates = [
        r for r in results
        if not _is_social(r.url)
        and not _looks_like_dotwatcher(r.url)
        and not _is_aggregator(r.url)
    ]

    fallback_candidates = [
        r for r in results
        if not _is_social(r.url)
        and not _looks_like_dotwatcher(r.url)
    ]

    candidates = primary_candidates or fallback_candidates
    candidate_urls = [r.url for r in candidates[:3]]

    fetched: List[Tuple[str, str]] = []
    for url in candidate_urls:
        text = await _fetch_page_text(url)
        if text:
            fetched.append((url, text))

    official_url: Optional[str] = None
    if fetched:
        official_url = await _pick_official_url_with_llm(event_title, fetched, model_name=context_model)

    # Choose best page to summarize
    context_url = official_url or dotwatcher_url
    context_summary: Optional[EventContextSummary] = None

    if context_url:
        # If we have an official site, try "rules-ish" pages too (often contain the real details)
        candidate_pages: List[str] = [context_url]

        if official_url:
            base = _base_site(official_url)
            candidate_pages.extend(_join_url(base, p) for p in _COMMON_RULE_PATHS)

        # Fetch first page that gives useful text
        page_text: Optional[str] = None
        chosen_url: Optional[str] = None
        for u in candidate_pages:
            t = await _fetch_page_text(u)
            if t:
                page_text = t
                chosen_url = u
                break

        if page_text and chosen_url:
            context_summary = await _summarise_event_context_from_text(
                event_title=event_title,
                page_text=page_text,
                source_url=chosen_url,
                model_name=context_model,
            )

    return official_url, dotwatcher_url, context_summary


# ---------------- Summarisation ----------------


async def _summarise_event_context_from_text(
    event_title: str,
    page_text: str,
    source_url: str,
    model_name: str = "gpt-4o-mini",
) -> EventContextSummary:
    model = OpenAIChatModel(model_name)

    system_prompt = (
        "Extract structured context for an ultra-distance cycling event.\n"
        "Only use facts supported by the provided page text; otherwise leave null/empty.\n"
        "Do NOT invent exact numbers.\n"
        "If you fill distance_km or total_climbing_m, you MUST also fill the corresponding evidence field\n"
        f"with source_url='{source_url}' and a short verbatim snippet from the page text.\n"
        "Return JSON matching EventContextSummary."
    )

    agent = Agent(model, output_type=EventContextSummary, system_prompt=system_prompt)

    user_prompt = (
        f"Event title: {event_title}\n"
        f"Source URL: {source_url}\n\n"
        f"--- PAGE TEXT START ---\n{page_text}\n--- PAGE TEXT END ---"
    )
    result = await agent.run(user_prompt)
    return result.output


# ---------------- Tool ----------------


@Tool
async def event_web_search(
    ctx: RunContext,
    article_id: Optional[int] = None,
    event_title: Optional[str] = None,
    event_url: Optional[str] = None,
    max_results: int = 8,
    context_model: str = "gpt-4o-mini",
) -> EventWebContext:
    """
    Fetch event context with robust inputs:
    - prefer article_id (DB-backed)
    - else event_title
    - if event_url is provided, try fetching it directly first (more accurate than web search)
    """
    # Normalize inputs
    title: Optional[str] = None
    if article_id is not None:
        title = get_article_title(article_id) or event_title
    else:
        title = event_title

    title = (title or "").strip() or None
    event_url = (event_url or "").strip() or None

    # If we have an explicit URL, try to summarize it directly first
    if event_url:
        page_text = await _fetch_page_text(event_url)
        if page_text:
            ctx_summary = await _summarise_event_context_from_text(
                event_title=title or event_url,
                page_text=page_text,
                source_url=event_url,
                model_name=context_model,
            )
            return EventWebContext(
                event_title=title or event_url,
                search_query="",
                official_url=None if _looks_like_dotwatcher(event_url) else event_url,
                dotwatcher_url=event_url if _looks_like_dotwatcher(event_url) else None,
                context=ctx_summary,
                results=[],
            )
        # If fetch failed, fall back to search using title (if available)

    if not title:
        logger.warning("event_web_search called without usable identifier (no title, no fetchable URL).")
        return EventWebContext(event_title="[unknown event]", search_query="")

    title_for_search = _strip_year(title)
    base_query = f"{title_for_search} official site rules route registration"
    results = await _search_on_web(base_query, max_results=max_results)

    official_url, dotwatcher_url, context_summary = await _guess_urls_and_context(
        event_title=title,
        results=results,
        context_model=context_model,
    )

    return EventWebContext(
        event_title=title,
        search_query=base_query,
        official_url=official_url,
        dotwatcher_url=dotwatcher_url,
        context=context_summary,
        results=results,
    )
