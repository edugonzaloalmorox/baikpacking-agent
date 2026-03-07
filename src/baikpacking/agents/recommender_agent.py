import os
import re
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import logfire

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai import Agent

from pydantic_ai.models.openai import OpenAIChatModel

from baikpacking.agents.models import SetupRecommendation
from baikpacking.embedding import embed_text
from baikpacking.logging_config import setup_logging
from baikpacking.tools.call_trace import CallTrace, record_trace_call, time_and_record
from baikpacking.tools.pg_vector_search import PgVectorSearchDeps
from baikpacking.tools.riders import run_search_similar_riders
from baikpacking.tools.event_context import run_event_web_search_sync

load_dotenv()
setup_logging()


class AgentSettings(BaseSettings):
    """Settings for the bikepacking recommender agent."""

    planner_model: str = "gpt-4o-mini"
    writer_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENT_",
        extra="ignore",
    )


settings = AgentSettings()

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_EVENT_RE = re.compile(
    r"\b(?:for|use|in|at)\s+([A-Z][A-Za-z0-9' -]+(?:Race|Divide|Challenge|Atlas Mountain Race|Tour|Trail))\b"
)
_KM_RE = re.compile(r"(\d{2,4})\s*km\b", re.IGNORECASE)
_M_RE = re.compile(r"(\d{3,5})\s*m\b", re.IGNORECASE)


# -------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Deterministic descriptor logic
# -------------------------------------------------------------------

def infer_event_archetype(flags: Dict[str, bool]) -> Dict[str, Any]:
    """
    Infer a coarse event archetype from descriptor flags.

    Returns a structured dict so retrieval can use both:
    - a primary archetype label
    - interpretable retrieval facets

    Example output:
    {
        "archetype": "gravel_ultra",
        "surface_family": "gravel",
        "terrain": ["mountainous"],
        "environment": ["remote"],
        "format": ["self_supported", "ultra"]
    }
    """
    terrain = []
    environment = []
    format_ = ["ultra"]

    if flags.get("mountain"):
        terrain.append("mountainous")

    if flags.get("desert"):
        environment.append("desert")
    if flags.get("remote"):
        environment.append("remote")
    if flags.get("cold_hot"):
        environment.append("temperature_swings")

    if flags.get("remote"):
        format_.append("self_supported")
    if flags.get("night"):
        format_.append("night_riding")
    if flags.get("navigation"):
        format_.append("navigation_heavy")

    # Surface family inference
    # Priority matters: MTB > gravel > road > mixed/off-road > unknown
    if flags.get("mtb"):
        surface_family = "mtb"
    elif flags.get("gravel"):
        surface_family = "gravel"
    elif flags.get("road"):
        surface_family = "road"
    elif flags.get("off_road"):
        surface_family = "mixed_offroad"
    else:
        surface_family = "unknown"

    # Archetype label
    if surface_family == "mtb":
        if flags.get("desert"):
            archetype = "desert_mtb_ultra"
        elif flags.get("mountain"):
            archetype = "mountain_mtb_ultra"
        else:
            archetype = "mtb_ultra"

    elif surface_family == "gravel":
        if flags.get("mountain"):
            archetype = "mountain_gravel_ultra"
        else:
            archetype = "gravel_ultra"

    elif surface_family == "road":
        if flags.get("mountain"):
            archetype = "mountain_road_ultra"
        else:
            archetype = "road_ultra"

    elif surface_family == "mixed_offroad":
        if flags.get("desert"):
            archetype = "desert_offroad_ultra"
        elif flags.get("mountain"):
            archetype = "mountain_offroad_ultra"
        else:
            archetype = "offroad_bikepacking_ultra"

    else:
        archetype = "general_bikepacking_ultra"

    return {
        "archetype": archetype,
        "surface_family": surface_family,
        "terrain": terrain,
        "environment": environment,
        "format": format_,
    }



def _keyword_flags(text: str) -> Dict[str, bool]:
    t = (text or "").lower()

    return {
        "road": any(k in t for k in [
            "road race", "paved", "tarmac", "asphalt", "road ultra", "road cycling"
        ]),
        "gravel": any(k in t for k in [
            "gravel", "gravel race", "dirt road", "fire road", "unbound"
        ]),
        "mtb": any(k in t for k in [
            "mtb", "mountain bike", "singletrack", "technical", "rocky", "hardtail",
            "full suspension", "29er", "trail bike"
        ]),
        "off_road": any(k in t for k in [
            "off-road", "off road", "singletrack", "track", "rocky", "trail",
            "technical", "jeep track", "doubletrack", "rough terrain", "dirt road"
        ]),
        "desert": any(k in t for k in [
            "desert", "sahara", "arid", "dry", "sand"
        ]),
        "mountain": any(k in t for k in [
            "mountain", "alpine", "climb", "elevation", "pass", "high mountains"
        ]),
        "remote": any(k in t for k in [
            "remote", "self-supported", "self supported", "unsupported", "no services",
            "minimal resupply"
        ]),
        "night": any(k in t for k in [
            "night", "dark", "overnight"
        ]),
        "cold_hot": any(k in t for k in [
            "temperature", "cold", "hot", "heat", "freezing", "temperature swings"
        ]),
        "navigation": any(k in t for k in [
            "navigation", "gps", "route", "track", "waypoint", "gpx"
        ]),
    }



def _extract_event_name(user_query: str) -> str:
    m = _EVENT_RE.search(user_query or "")
    if m:
        return m.group(1).strip()

    # Fallback: common case in your script
    if "atlas mountain race" in (user_query or "").lower():
        return "Atlas Mountain Race"

    return user_query.strip()



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


def _build_descriptor_query(
    event_name: str,
    event_context: str,
    user_question: str,
) -> Dict[str, Any]:
    """
    Build an archetype-aware retrieval query.

    Goal:
    - if exact event is absent from the KB, retrieve riders from similar event families
    - avoid over-biasing to MTB unless the context actually suggests MTB/off-road terrain
    """
    full_text = "\n".join(
        [
            event_name or "",
            event_context or "",
            user_question or "",
        ]
    )

    flags = _keyword_flags(full_text)
    metrics = _extract_metrics(event_context or "")
    archetype_info = infer_event_archetype(flags)

    archetype = archetype_info["archetype"]
    surface_family = archetype_info["surface_family"]

    descriptors = ["self-supported ultra endurance bikepacking race"]

    # Surface / bike family
    if surface_family == "road":
        descriptors.append("road ultra race")
        descriptors.append("paved")
    elif surface_family == "gravel":
        descriptors.append("gravel ultra race")
        descriptors.append("mixed dirt roads")
    elif surface_family == "mtb":
        descriptors.append("MTB mountain bike ultra")
        descriptors.append("rough off-road terrain")
    elif surface_family == "mixed_offroad":
        descriptors.append("off-road bikepacking ultra")
        descriptors.append("mixed rough terrain")

    # Terrain / environment
    if flags["mountain"]:
        descriptors.append("mountainous long climbs")
    if flags["desert"]:
        descriptors.append("desert arid")
    if flags["remote"]:
        descriptors.append("remote minimal resupply")
    if flags["night"]:
        descriptors.append("night riding")
    if flags["navigation"]:
        descriptors.append("navigation GPS route")
    if flags["cold_hot"]:
        descriptors.append("temperature swings")

    # Metrics
    if metrics.get("distance_km"):
        descriptors.append(f"{metrics['distance_km']} km")
    if metrics.get("elevation_m"):
        descriptors.append(f"{metrics['elevation_m']} m climbing")

    base_descriptor = ", ".join(dict.fromkeys(descriptors))

    # Intent query: only use user intent as a secondary retrieval variant
    intent = (user_question or "").strip()
    descriptor_with_intent = (
        f"{base_descriptor}. Question focus: {intent}"
        if intent
        else base_descriptor
    )

    # Optional adjacent archetypes for future fallback/reranking
    adjacent_archetypes = []
    if archetype == "mountain_gravel_ultra":
        adjacent_archetypes = ["gravel_ultra", "mountain_offroad_ultra"]
    elif archetype == "gravel_ultra":
        adjacent_archetypes = ["mountain_gravel_ultra", "offroad_bikepacking_ultra"]
    elif archetype == "mountain_road_ultra":
        adjacent_archetypes = ["road_ultra"]
    elif archetype == "road_ultra":
        adjacent_archetypes = ["mountain_road_ultra"]
    elif archetype == "desert_mtb_ultra":
        adjacent_archetypes = ["mountain_mtb_ultra", "desert_offroad_ultra"]
    elif archetype == "mountain_mtb_ultra":
        adjacent_archetypes = ["mtb_ultra", "mountain_offroad_ultra"]
    elif archetype == "mtb_ultra":
        adjacent_archetypes = ["mountain_mtb_ultra", "offroad_bikepacking_ultra"]
    elif archetype == "desert_offroad_ultra":
        adjacent_archetypes = ["mountain_offroad_ultra", "desert_mtb_ultra"]
    elif archetype == "mountain_offroad_ultra":
        adjacent_archetypes = ["offroad_bikepacking_ultra", "mountain_mtb_ultra"]

    return {
        "archetype": archetype,
        "surface_family": surface_family,
        "adjacent_archetypes": adjacent_archetypes,
        "descriptor_query": base_descriptor,
        "descriptor_query_with_intent": descriptor_with_intent,
        "features": {
            "flags": flags,
            "metrics": metrics,
            "archetype_info": archetype_info,
        },
    }

# -------------------------------------------------------------------
# Planner / writer schemas
# -------------------------------------------------------------------

class RetrievalPlan(BaseModel):
    event_name: str
    retrieval_query: str
    include_intent_query: bool = False
    top_k_riders: int = 5
    max_chunks_per_rider: int = 2
    top_k_chunks: int = 80


class CompactChunk(BaseModel):
    chunk_index: Optional[int] = None
    text: str


class CompactRider(BaseModel):
    name: Optional[str] = None
    event_title: Optional[str] = None
    year: Optional[int] = None
    best_score: Optional[float] = None
    bike_type: Optional[str] = None
    wheels: Optional[str] = None
    tyres: Optional[str] = None
    drivetrain: Optional[str] = None
    bags: Optional[str] = None
    sleep_system: Optional[str] = None
    key_items: List[str] = Field(default_factory=list)
    chunks: List[CompactChunk] = Field(default_factory=list)


class WriterInput(BaseModel):
    user_query: str
    event_name: str
    event_context: str
    descriptor_query: str
    similar_riders: List[CompactRider]


PLANNER_PROMPT = """
You build a retrieval plan for a bikepacking recommender.

Return only a RetrievalPlan.

Rules:
- Keep the retrieval broad enough to find similar events.
- Prefer event characteristics over exact event-name matching.
- For gear-specific questions like lights, include intent only if it helps retrieval.
- Keep retrieval small and fast.
- Defaults should be:
  - top_k_riders = 5
  - max_chunks_per_rider = 2
  - top_k_chunks = 80
""".strip()


WRITER_PROMPT = """
You are a bikepacking equipment and ultra-distance cycling expert.

Return only valid JSON matching SetupRecommendation.

Grounding rules:
- ALL gear details must come from similar_riders.
- event_context is only for understanding the target event characteristics.
- Do not invent gear, brands, or specs.
- If a field cannot be grounded, leave it empty/null and explain that in reasoning.
- Use similar_riders exactly as provided in the input for the output similar_riders content.
- Prefer concise output:
  - summary: 3-4 sentences
  - reasoning: 3-5 sentences

Output rules:
- event must be the requested event_name
- if the exact event is absent from rider data, clearly say the setup is based on similar events
- recommended_setup should contain as many grounded fields as possible without guessing
""".strip()


planner_model = OpenAIChatModel(settings.planner_model)
writer_model = OpenAIChatModel(settings.writer_model)

planner_agent = Agent(
    model=planner_model,
    output_type=RetrievalPlan,
    system_prompt=PLANNER_PROMPT,
)

writer_agent = Agent(
    model=writer_model,
    output_type=SetupRecommendation,
    system_prompt=WRITER_PROMPT,
)


# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------

def _build_deps(call_trace: Optional[CallTrace] = None) -> PgVectorSearchDeps:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")

    return PgVectorSearchDeps(
        embed_query=embed_text,
        database_url=database_url,
        call_trace=call_trace,
    )


# -------------------------------------------------------------------
# Grounding compaction
# -------------------------------------------------------------------

def _compact_riders(riders: List[Any]) -> List[CompactRider]:
    out: List[CompactRider] = []

    for r in riders or []:
        compact_chunks: List[CompactChunk] = []
        for idx, c in enumerate(getattr(r, "chunks", []) or []):
            text = getattr(c, "text", None) or getattr(c, "content", None) or ""
            if text:
                compact_chunks.append(
                    CompactChunk(
                        chunk_index=getattr(c, "chunk_index", idx),
                        text=text[:300],
                    )
                )

        key_items = []
        raw_key_items = getattr(r, "key_items", None) or []
        for item in raw_key_items[:5]:
            if isinstance(item, str) and item.strip():
                key_items.append(item.strip())

        out.append(
            CompactRider(
                name=getattr(r, "name", None),
                event_title=getattr(r, "event_title", None),
                year=getattr(r, "year", None) or _infer_year_from_title(getattr(r, "event_title", None)),
                best_score=getattr(r, "best_score", None),
                bike_type=getattr(r, "bike_type", None),
                wheels=getattr(r, "wheels", None),
                tyres=getattr(r, "tyres", None),
                drivetrain=getattr(r, "drivetrain", None),
                bags=getattr(r, "bags", None),
                sleep_system=getattr(r, "sleep_system", None),
                key_items=key_items,
                chunks=compact_chunks[:2],
            )
        )

    return out


def _event_context_to_text(event_context_obj: Any) -> str:
    if not event_context_obj or not getattr(event_context_obj, "context", None):
        return ""

    ctx = event_context_obj.context
    parts = [
        ctx.summary or "",
        ctx.surface or "",
        ctx.route_character or "",
        ctx.climate_notes or "",
        ctx.resupply_notes or "",
        " ".join(ctx.constraints or []),
    ]
    return "\n".join(p for p in parts if p)


# -------------------------------------------------------------------
# Main orchestration
# -------------------------------------------------------------------

def recommend_setup_with_trace(user_query: str) -> Tuple[SetupRecommendation, CallTrace]:
    with logfire.span("recommender.run", user_query=user_query):
    
        trace = CallTrace()
        deps = _build_deps(call_trace=trace)

        event_name = _extract_event_name(user_query)

        # 1) Deterministic event context fetch
        event_context_obj = time_and_record(
            deps=deps,
            tool_name="event_web_search",
            args={"event_title": event_name},
            fn=lambda: run_event_web_search_sync(
                event_title=event_name,
                deps=deps,
            ),
        )

        event_context_text = _event_context_to_text(event_context_obj)

        # 2) Deterministic descriptor query build
        descriptor = _build_descriptor_query(
            event_name=event_name,
            event_context=event_context_text,
            user_question=user_query,
        )

        # 3) Small planner call
        planner_input = {
        "user_query": user_query,
        "event_name": event_name,
        "event_context": event_context_text[:2000],
        "archetype": descriptor["archetype"],
        "surface_family": descriptor["surface_family"],
        "descriptor_query": descriptor["descriptor_query"],
        "descriptor_query_with_intent": descriptor["descriptor_query_with_intent"],
    }
        plan = planner_agent.run_sync(str(planner_input)).output

        
        print("\n--- EVENT CONTEXT TEXT ---")
        print(event_context_text)

        print("\n--- DESCRIPTOR FEATURES ---")
        print(json.dumps(descriptor["features"], indent=2, ensure_ascii=False))

        print("\n--- DESCRIPTOR QUERY ---")
        print(descriptor["descriptor_query"])
        

        
        # 4) Retrieval tool call
        retrieval_query = descriptor["descriptor_query"]

        record_trace_call(
            deps=deps,
            tool_name="search_similar_riders_attempt",
            args={
                "attempt": 1,
                "query": retrieval_query,
                "top_k_riders": plan.top_k_riders,
                "max_chunks_per_rider": plan.max_chunks_per_rider,
                "top_k_chunks": plan.top_k_chunks,
            },
            result={"ok": True},
            elapsed_ms=0.0,
        )

        riders = time_and_record(
        deps=deps,
        tool_name="search_similar_riders",
        args={
            "query": retrieval_query,
            "top_k_riders": plan.top_k_riders,
            "max_chunks_per_rider": plan.max_chunks_per_rider,
            "top_k_chunks": plan.top_k_chunks,
        },
        fn=lambda: run_search_similar_riders(
            query=retrieval_query,
            top_k_riders=plan.top_k_riders,
            max_chunks_per_rider=plan.max_chunks_per_rider,
            top_k_chunks=plan.top_k_chunks,
            deps=deps,
        ),
    )

        if (not riders or len(riders) < 3) and plan.include_intent_query:
            fallback_query = descriptor["descriptor_query_with_intent"]

            record_trace_call(
                deps=deps,
                tool_name="search_similar_riders_attempt",
                args={
                    "attempt": 2,
                    "query": fallback_query,
                    "top_k_riders": plan.top_k_riders,
                    "max_chunks_per_rider": plan.max_chunks_per_rider,
                    "top_k_chunks": plan.top_k_chunks,
                },
                result={"ok": True},
                elapsed_ms=0.0,
            )

            riders = time_and_record(
                deps=deps,
                tool_name="search_similar_riders",
                args={
                    "query": fallback_query,
                    "top_k_riders": plan.top_k_riders,
                    "max_chunks_per_rider": plan.max_chunks_per_rider,
                    "top_k_chunks": plan.top_k_chunks,
                },
                fn=lambda: run_search_similar_riders(
                    query=fallback_query,
                    top_k_riders=plan.top_k_riders,
                    max_chunks_per_rider=plan.max_chunks_per_rider,
                    top_k_chunks=plan.top_k_chunks,
                    deps=deps,
                ),
            )
            retrieval_query = fallback_query

        if not riders:
            raise RuntimeError("No similar riders returned; cannot produce grounded recommendation.")

        compact_riders = _compact_riders(riders)

        # 5) Final writer call
        writer_input = WriterInput(
            user_query=user_query,
            event_name=event_name,
            event_context=event_context_text[:2500],
            descriptor_query=retrieval_query,
            similar_riders=compact_riders,
        )

        rec = writer_agent.run_sync(writer_input.model_dump_json(indent=2)).output

        # Keep exact retrieved rider content as final grounding source
        rec.similar_riders = riders

        record_trace_call(
            deps=deps,
            tool_name="loop",
            args={"stage": "stop", "note": "deterministic orchestration complete"},
            result={
                "event_name": event_name,
                "retrieval_query": retrieval_query,
                "riders": len(riders),
            },
            elapsed_ms=0.0,
        )

        if not rec.event or not rec.event.strip():
            rec.event = event_name

        return _postprocess_recommendation(rec), trace


def recommend_setup(user_query: str) -> SetupRecommendation:
    rec, _trace = recommend_setup_with_trace(user_query)
    return rec