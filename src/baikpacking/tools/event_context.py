import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anyio
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, RunContext, Tool
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
    """Evidence from a fetched page (not a search-engine snippet)."""

    source_url: str
    snippet: Optional[str] = None


class EventContextSummary(BaseModel):
    """
    Structured, evidence-backed context for an ultra-distance cycling event.

    Rules:
    - Numeric fields must only be filled if explicitly supported by page text.
    - If a numeric field is filled, the corresponding evidence must be provided.
    """

    distance_km: Optional[float] = None
    distance_evidence: Optional[Evidence] = None

    total_climbing_m: Optional[int] = None
    climbing_evidence: Optional[Evidence] = None

    surface: Optional[str] = None
    route_character: Optional[str] = None
    climate_notes: Optional[str] = None
    resupply_notes: Optional[str] = None

    constraints: List[str] = Field(default_factory=list)

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


# ---------------- Cache helpers ----------------


def _has_useful_event_context(value: EventWebContext) -> bool:
    """
    Strong-enough result to keep in cache and reuse.

    We accept either:
    - non-empty structured context, or
    - search results / URLs that can still provide fallback signal
    """
    if value.context is not None:
        ctx = value.context
        if any(
            [
                ctx.summary,
                ctx.surface,
                ctx.route_character,
                ctx.climate_notes,
                ctx.resupply_notes,
                bool(ctx.constraints),
                ctx.distance_km is not None,
                ctx.total_climbing_m is not None,
            ]
        ):
            return True

    return bool(value.official_url or value.dotwatcher_url or value.results)


def _event_context_to_text(event_context_obj: Any) -> str:
    """
    Convert EventWebContext to retrieval-friendly text.

    This is useful when structured context is sparse and we need fallback
    signal from search results.
    """
    if not event_context_obj:
        return ""

    parts: List[str] = []

    ctx = getattr(event_context_obj, "context", None)
    if ctx is not None:
        parts.extend(
            [
                ctx.summary or "",
                ctx.surface or "",
                ctx.route_character or "",
                ctx.climate_notes or "",
                ctx.resupply_notes or "",
                " ".join(ctx.constraints or []),
            ]
        )

    for r in getattr(event_context_obj, "results", [])[:5]:
        title = getattr(r, "title", None) or ""
        snippet = getattr(r, "snippet", None) or ""
        if title:
            parts.append(title)
        if snippet:
            parts.append(snippet)

    official_url = getattr(event_context_obj, "official_url", None)
    dotwatcher_url = getattr(event_context_obj, "dotwatcher_url", None)

    if official_url:
        parts.append(str(official_url))
    if dotwatcher_url:
        parts.append(str(dotwatcher_url))

    return "\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


# ---------------- Cache ----------------


EVENT_CONTEXT_CACHE_PATH = Path(
    os.getenv("EVENT_CONTEXT_CACHE_PATH", "data/eval/event_context_cache.jsonl")
)
EVENT_CONTEXT_CACHE_TTL_S = int(
    os.getenv("EVENT_CONTEXT_CACHE_TTL_S", str(7 * 24 * 3600))
)

_EVENT_CONTEXT_CACHE: Dict[str, Tuple[float, EventWebContext]] = {}
_CACHE_LOADED = False


def _event_cache_key(
    *,
    title: str,
    event_url: Optional[str],
    context_model: str,
    max_results: int,
) -> str:
    base = "|".join(
        [
            (title or "").strip().lower(),
            (event_url or "").strip().lower(),
            str(context_model),
            str(max_results),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _load_cache_once() -> None:
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True

    if not EVENT_CONTEXT_CACHE_PATH.exists():
        return

    try:
        with EVENT_CONTEXT_CACHE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                d = json.loads(line)
                key = str(d.get("key", "")).strip()
                created_at = float(d.get("created_at", 0) or 0)
                value = d.get("value")

                if not key or not isinstance(value, dict):
                    continue

                if created_at and (time.time() - created_at) > EVENT_CONTEXT_CACHE_TTL_S:
                    continue

                ctx = EventWebContext.model_validate(value)

                # Skip weak cached objects so they don't poison future runs.
                if not _has_useful_event_context(ctx):
                    continue

                _EVENT_CONTEXT_CACHE[key] = (created_at or time.time(), ctx)

    except Exception as exc:
        logger.warning(
            "Failed to load event context cache (%s): %s",
            EVENT_CONTEXT_CACHE_PATH,
            exc,
        )


def _cache_get(key: str) -> Optional[EventWebContext]:
    _load_cache_once()
    item = _EVENT_CONTEXT_CACHE.get(key)
    if not item:
        return None

    created_at, ctx = item
    if created_at and (time.time() - created_at) > EVENT_CONTEXT_CACHE_TTL_S:
        _EVENT_CONTEXT_CACHE.pop(key, None)
        return None

    if not _has_useful_event_context(ctx):
        _EVENT_CONTEXT_CACHE.pop(key, None)
        return None

    return ctx


def _cache_set(key: str, value: EventWebContext) -> None:
    # Do not persist weak / empty contexts.
    if not _has_useful_event_context(value):
        return

    created_at = time.time()
    _EVENT_CONTEXT_CACHE[key] = (created_at, value)

    try:
        EVENT_CONTEXT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_CONTEXT_CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "key": key,
                        "created_at": created_at,
                        "value": value.model_dump(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception as exc:
        logger.warning("Failed to persist event context cache: %s", exc)


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
    return any(
        bad in d
        for bad in (
            "facebook.com",
            "instagram.com",
            "twitter.com",
            "x.com",
            "youtube.com",
            "tiktok.com",
        )
    )


def _looks_like_dotwatcher(url: str) -> bool:
    return "dotwatcher.cc" in _domain(url)


def _is_aggregator(url: str) -> bool:
    d = _domain(url)
    return any(a in d for a in _AGGREGATOR_DOMAINS)


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _base_site(url: str) -> str:
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
    candidates: List[Tuple[str, str]],
    model_name: str,
) -> Optional[str]:
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
            "Prefer pages containing registration, rules, route, GPX, FAQ, checkpoints, or mandatory kit.\n"
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
    dotwatcher_url = next((r.url for r in results if _looks_like_dotwatcher(r.url)), None)

    primary_candidates = [
        r
        for r in results
        if not _is_social(r.url)
        and not _looks_like_dotwatcher(r.url)
        and not _is_aggregator(r.url)
    ]

    fallback_candidates = [
        r
        for r in results
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
        official_url = await _pick_official_url_with_llm(
            event_title,
            fetched,
            model_name=context_model,
        )

    context_url = official_url or dotwatcher_url
    context_summary: Optional[EventContextSummary] = None

    if context_url:
        candidate_pages: List[str] = [context_url]

        if official_url:
            base = _base_site(official_url)
            candidate_pages.extend(_join_url(base, p) for p in _COMMON_RULE_PATHS)

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
        "Only use facts supported by the provided page text; otherwise leave null or empty.\n"
        "Do not invent exact numbers.\n"
        "If you fill distance_km or total_climbing_m, you must also fill the corresponding evidence field "
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


# ---------------- Plain implementation ----------------


async def run_event_web_search(
    *,
    article_id: Optional[int] = None,
    event_title: Optional[str] = None,
    event_url: Optional[str] = None,
    max_results: int = 8,
    context_model: str = "gpt-4o-mini",
    deps: Any = None,
) -> EventWebContext:
    """
    Plain async implementation for deterministic orchestration.
    """
    del deps  # reserved for future use

    title: Optional[str]
    if article_id is not None:
        title = get_article_title(article_id) or event_title
    else:
        title = event_title

    title = (title or "").strip() or None
    event_url = (event_url or "").strip() or None

    cache_key = _event_cache_key(
        title=title or "",
        event_url=event_url,
        context_model=context_model,
        max_results=max_results,
    )

    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if event_url:
        page_text = await _fetch_page_text(event_url)
        if page_text:
            ctx_summary = await _summarise_event_context_from_text(
                event_title=title or event_url,
                page_text=page_text,
                source_url=event_url,
                model_name=context_model,
            )
            out = EventWebContext(
                event_title=title or event_url,
                search_query="",
                official_url=None if _looks_like_dotwatcher(event_url) else event_url,
                dotwatcher_url=event_url if _looks_like_dotwatcher(event_url) else None,
                context=ctx_summary,
                results=[],
            )
            _cache_set(cache_key, out)
            return out

    if not title:
        logger.warning("run_event_web_search called without usable identifier.")
        return EventWebContext(event_title="[unknown event]", search_query="")

    title_for_search = _strip_year(title)

    queries = [
        f"{title_for_search} official site rules route registration",
        f"{title_for_search} bikepacking race route terrain",
    ]

    merged_results: List[EventSearchResult] = []
    seen_urls = set()

    for q in queries:
        rows = await _search_on_web(q, max_results=max_results)
        for r in rows:
            if r.url in seen_urls:
                continue
            seen_urls.add(r.url)
            merged_results.append(r)

    official_url, dotwatcher_url, context_summary = await _guess_urls_and_context(
        event_title=title,
        results=merged_results,
        context_model=context_model,
    )

    out = EventWebContext(
        event_title=title,
        search_query=queries[0],
        official_url=official_url,
        dotwatcher_url=dotwatcher_url,
        context=context_summary,
        results=merged_results,
    )
    _cache_set(cache_key, out)
    return out


def run_event_web_search_sync(
    *,
    article_id: Optional[int] = None,
    event_title: Optional[str] = None,
    event_url: Optional[str] = None,
    max_results: int = 8,
    context_model: str = "gpt-4o-mini",
    deps: Any = None,
) -> EventWebContext:
    """
    Sync wrapper for deterministic orchestration code.
    """
    return anyio.run(
        lambda: run_event_web_search(
            article_id=article_id,
            event_title=event_title,
            event_url=event_url,
            max_results=max_results,
            context_model=context_model,
            deps=deps,
        )
    )


# ---------------- Tool wrapper ----------------


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
    Agent-exposed wrapper around the plain implementation.
    """
    return await run_event_web_search(
        article_id=article_id,
        event_title=event_title,
        event_url=event_url,
        max_results=max_results,
        context_model=context_model,
        deps=getattr(ctx, "deps", None),
    )