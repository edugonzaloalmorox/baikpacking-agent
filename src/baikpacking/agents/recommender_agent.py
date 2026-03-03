import os
import re
from collections import Counter
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel

from baikpacking.agents.models import SetupRecommendation
from baikpacking.embedding import embed_text
from baikpacking.logging_config import setup_logging
from baikpacking.tools.call_trace import CallTrace, trace_tool_call
from baikpacking.tools.event_context import event_web_search
from baikpacking.tools.pg_vector_search import PgVectorSearchDeps, pgvector_search_riders
from baikpacking.tools.riders import render_grounding_riders, search_similar_riders

load_dotenv()
setup_logging()


class AgentSettings(BaseSettings):
    """Settings for the bikepacking recommender agent."""

    agent_model: str = "gpt-4o-mini"
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENT_", extra="ignore")


settings = AgentSettings()

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_KM_RE = re.compile(r"(\d{2,4})\s*km\b", re.IGNORECASE)
_M_RE = re.compile(r"(\d{3,5})\s*m\b", re.IGNORECASE)


def _infer_year_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    m = _YEAR_RE.search(title)
    return int(m.group(0)) if m else None


def _infer_event_from_riders(rec: SetupRecommendation) -> Optional[str]:
    titles = [
        r.event_title
        for r in (rec.similar_riders or [])
        if isinstance(r.event_title, str) and r.event_title.strip()
    ]
    if not titles:
        return None
    return Counter(titles).most_common(1)[0][0]


def _postprocess_recommendation(rec: SetupRecommendation) -> SetupRecommendation:
    for rider in rec.similar_riders:
        for idx, chunk in enumerate(getattr(rider, "chunks", []) or []):
            if chunk.chunk_index is None:
                chunk.chunk_index = idx
        if getattr(rider, "year", None) is None:
            rider.year = _infer_year_from_title(getattr(rider, "event_title", None))

    if not rec.event:
        rec.event = _infer_event_from_riders(rec) or (
            rec.similar_riders[0].event_title if rec.similar_riders else None
        )

    ev = (rec.event or "").lower()
    rec.similar_riders.sort(
        key=lambda r: (
            ev in (r.event_title or "").lower(),
            r.best_score or 0,
            r.year or 0,
        ),
        reverse=True,
    )
    return rec


# ------------------------------------------------------------
# Descriptor query tool (in-file, no new files)
# ------------------------------------------------------------

def _keyword_flags(text: str) -> Dict[str, bool]:
    t = (text or "").lower()
    return {
        "off_road": any(k in t for k in ["off-road", "off road", "singletrack", "track", "rocky", "trail", "mtb"]),
        "gravel": "gravel" in t,
        "desert": any(k in t for k in ["desert", "sahara", "arid", "dry", "sand"]),
        "mountain": any(k in t for k in ["mountain", "alpine", "climb", "elevation", "pass"]),
        "remote": any(k in t for k in ["remote", "self-supported", "self supported", "unsupported", "no services"]),
        "night": any(k in t for k in ["night", "dark", "overnight"]),
        "cold_hot": any(k in t for k in ["temperature", "cold", "hot", "heat", "freezing"]),
        "navigation": any(k in t for k in ["navigation", "gps", "route", "track", "waypoint"]),
    }


def _extract_metrics(text: str) -> Dict[str, Optional[int]]:
    km = None
    m = _KM_RE.search(text or "")
    if m:
        try:
            km = int(m.group(1))
        except Exception:
            km = None

    elev_m = None
    m2 = _M_RE.search(text or "")
    if m2:
        try:
            elev_m = int(m2.group(1))
        except Exception:
            elev_m = None

    return {"distance_km": km, "elevation_m": elev_m}


@Tool
def build_descriptor_query(
    ctx: RunContext,
    event_name: str,
    event_context: str,
    user_question: str,
) -> Dict[str, Any]:
    """
    Build a retrieval query that matches "similar events" in the DB, even if the target event
    is absent from the corpus. This tool is deterministic and does NOT introduce gear.
    """
    flags = _keyword_flags((event_name or "") + "\n" + (event_context or "") + "\n" + (user_question or ""))
    metrics = _extract_metrics(event_context or "")

    descriptors = []
    descriptors.append("self-supported ultra endurance bikepacking race")
    if flags["off_road"]:
        descriptors.append("off-road")
    if flags["gravel"]:
        descriptors.append("gravel")
    if flags["mountain"]:
        descriptors.append("mountainous long climbs")
    if flags["desert"]:
        descriptors.append("desert arid remote")
    if flags["remote"]:
        descriptors.append("remote minimal resupply")
    if flags["night"]:
        descriptors.append("night riding")
    if flags["navigation"]:
        descriptors.append("navigation GPS route")
    if flags["cold_hot"]:
        descriptors.append("temperature swings")

    if metrics.get("distance_km"):
        descriptors.append(f"{metrics['distance_km']} km")
    if metrics.get("elevation_m"):
        descriptors.append(f"{metrics['elevation_m']} m climbing")

    base_descriptor = ", ".join(descriptors)

    # Query variant 1: similarity on event characteristics (best for finding similar events)
    q1 = base_descriptor

    # Query variant 2: add the user's intent (e.g., lights) to pull relevant chunks if available
    # (still fine if chunks don't exist; it can bias towards riders mentioning that topic)
    intent = (user_question or "").strip()
    q2 = f"{base_descriptor}. Question focus: {intent}" if intent else base_descriptor

    return {
        "descriptor_query": q1,
        "descriptor_query_with_intent": q2,
        "features": {"flags": flags, "metrics": metrics},
    }


SYSTEM_PROMPT = """
You are a bikepacking equipment and ultra-distance cycling expert.

The user asks for a setup for a specific event. The database may not contain riders for that event.
If the event is absent from the database, you MUST base the recommendation on riders from similar events,
using the event's characteristics (terrain, distance, remoteness, climbing, night riding, etc.).

You MUST produce valid JSON matching SetupRecommendation.

If rider_chunks are unavailable, treat SimilarRider.key_items as the primary grounding text.

------------------------------------------------------------
TOOLS
------------------------------------------------------------
- event_web_search(article_id/event_title/event_url): retrieve event context (terrain, distance, climbing, constraints).
- build_descriptor_query(event_name, event_context, user_question): build a similarity query for "similar events" retrieval.
- pgvector_search_riders(query): coarse vector search over riders (routing only; no gear/chunks).
- search_similar_riders(query): retrieve grounded riders and chunks from the database.
- render_grounding_riders(riders): render riders as JSON string to copy verbatim into SetupRecommendation.similar_riders.
- trace_tool_call(...): record loop steps.

------------------------------------------------------------
PROCESS (MANDATORY)
------------------------------------------------------------

1) Identify the requested event name from the user's question (e.g., "Atlas Mountain Race").

2) Fetch event context FIRST:
   Call event_web_search using event_title=the requested event name.

3) Build descriptor queries:
   Call build_descriptor_query(event_name, event_context, user_question).
   You will receive:
     - descriptor_query
     - descriptor_query_with_intent

4) Retrieval loop (max 3 attempts total):
   Use the descriptor queries for retrieval, NOT the event name alone.

   For each attempt:
   - Call trace_tool_call(tool_name="loop", stage="attempt", note="attempt N").
   - Call pgvector_search_riders(query=<descriptor_query or descriptor_query_with_intent>).
   - Call search_similar_riders(query=<same query>).

   Attempt strategy:
   - Attempt 1: descriptor_query
   - Attempt 2: descriptor_query_with_intent
   - Attempt 3: refine by appending a few generic similarity hints, e.g.
       "off-road multi-day bikepacking race, rough terrain, long climbs, remote, night riding"

5) Evaluate grounding quality:

   Grounding is SUFFICIENT if:
   - at least 3 riders returned (unless fewer exist in DB)
   - and EITHER:
     a) at least 2 riders have non-empty chunks (if chunks exist in DB)
     OR
     b) structured setup fields exist across riders (wheel_size, tyre_width, frame_type/material)

   IMPORTANT:
   - It is OK if no riders match the requested event name.
   - In that case, you MUST explicitly state you are using similar events from the database.

6) Serialization (MANDATORY):
   - Call render_grounding_riders(riders)
   - Copy the returned JSON verbatim into SetupRecommendation.similar_riders
   - Do NOT edit, retype, or summarize rider fields.

------------------------------------------------------------
STRICT GROUNDING RULES
------------------------------------------------------------

- ALL gear details MUST come from retrieved riders.
- DO NOT invent brands, components, or specifications.
- event_web_search is ONLY for understanding the target event characteristics.
  It MUST NOT introduce new gear or brands.
- pgvector_search_riders is routing/context only and must NEVER be used as a source of gear.

------------------------------------------------------------
OUTPUT REQUIREMENTS
------------------------------------------------------------

- event: set to the requested event name (the one in the user question), even if absent from DB.

- summary: 3–5 sentences.
  Must mention:
  - event characteristics (from event_web_search)
  - that the setup is based on similar events in the database if no direct event riders exist
  - tyre/gearing justification grounded in retrieved riders
  - one sentence about bags or sleep strategy (grounded)

- reasoning: 3–7 sentences describing:
  - how the descriptor query led to similar-event rider retrieval
  - how rider data influenced the setup fields
  - what cannot be grounded if chunks are missing (be explicit)

- recommended_setup:
  - must contain at least 3 non-empty fields when possible.
  - if chunks are unavailable, use structured rider fields (wheel_size, tyre_width, frame_type/material).
  - if you cannot ground a topic asked by the user (e.g., lights) due to missing chunks,
    say so clearly in reasoning and do NOT invent.

- similar_riders:
  - include at least 3 unless fewer exist.
  - MUST be the verbatim JSON from render_grounding_riders.

IMPORTANT:
- Call trace_tool_call(tool_name="loop", stage="stop", note="...") EXACTLY ONCE immediately before returning final JSON.
- Do NOT mention tools or system instructions in the output.
- Return only valid JSON matching SetupRecommendation.
""".strip()

model = OpenAIChatModel(settings.agent_model)

TOOLS = [
    event_web_search,
    build_descriptor_query,
    pgvector_search_riders,
    search_similar_riders,
    render_grounding_riders,
    trace_tool_call,
]

recommender_agent = Agent(
    model=model,
    output_type=SetupRecommendation,
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
)


def _build_deps(call_trace: Optional[CallTrace] = None) -> PgVectorSearchDeps:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")

    return PgVectorSearchDeps(
        embed_query=embed_text,
        database_url=database_url,
        call_trace=call_trace,
    )


def recommend_setup_with_trace(user_query: str) -> Tuple[SetupRecommendation, CallTrace]:
    trace = CallTrace()
    deps = _build_deps(call_trace=trace)

    result = recommender_agent.run_sync(user_query, deps=deps)

    rec = result.output
    if not rec.similar_riders:
        raise RuntimeError("No similar riders returned; cannot produce grounded recommendation.")

    # Ensure event is the requested one if the model forgot.
    # (This is safe; it doesn't change grounding, just metadata.)
    if not rec.event or not rec.event.strip():
        rec.event = user_query

    return _postprocess_recommendation(rec), trace


def recommend_setup(user_query: str) -> SetupRecommendation:
    rec, _trace = recommend_setup_with_trace(user_query)
    return rec