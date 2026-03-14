import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from baikpacking.agents.models import SetupRecommendation, QueryIntent, RetrievalIntentBundle
from baikpacking.embedding import embed_text
from baikpacking.logging_config import setup_logging
from baikpacking.tools.call_trace import CallTrace, record_trace_call, time_and_record
from baikpacking.tools.event_context import run_event_web_search_sync
from baikpacking.tools.pg_vector_search import PgVectorSearchDeps
from baikpacking.tools.riders import run_search_similar_riders

load_dotenv()
setup_logging()


class AgentSettings(BaseSettings):
    """Settings for the bikepacking recommender agent."""

    writer_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENT_",
        extra="ignore",
    )


settings = AgentSettings()

DEFAULT_TOP_K_RIDERS = 5
DEFAULT_MAX_CHUNKS_PER_RIDER = 2
DEFAULT_TOP_K_CHUNKS = 80

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_KM_RE = re.compile(r"(\d{2,4})\s*km\b", re.IGNORECASE)
_M_RE = re.compile(r"(\d{3,5})\s*m\b", re.IGNORECASE)

_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9'&\-]+|[?.,:;!()]")
_TITLEISH_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9'&\-]*$")
_ALLCAPS_SHORT_RE = re.compile(r"^[A-Z0-9]{2,6}$")
_EVENT_FRAGMENT_RE = re.compile(
    r"\b(?:official|website|site|rules|route|registration|terrain|weather|setup|gear)\b.*$",
    re.IGNORECASE,
)

KNOWN_EVENTS: Dict[str, str] = {
    "atlas mountain race": "Atlas Mountain Race",
    "amr": "Atlas Mountain Race",
    "transiberica": "Transiberica",
    "gb duro": "GB Duro",
    "silk road mountain race": "Silk Road Mountain Race",
    "srmr": "Silk Road Mountain Race",
    "tour divide": "Tour Divide",
    "badlands": "Badlands",
    "highland trail 550": "Highland Trail 550",
    "ht550": "Highland Trail 550",
    "arizona trail race": "Arizona Trail Race",
    "aztr": "Arizona Trail Race",
    "transcontinental race": "Transcontinental Race",
    "transcontinental": "Transcontinental Race",
    "tcr": "Transcontinental Race",
}

_EVENT_PREFIXES = (
    "for",
    "use",
    "bring",
    "take",
    "ride",
    "riding",
    "setup for",
    "set up for",
    "at",
    "in",
    "doing",
    "race",
    "event",
)

_EVENT_CONTEXT_PATTERNS = [
    re.compile(
        rf"\b(?:{'|'.join(p.replace(' ', r'\s+') for p in _EVENT_PREFIXES)})\s+the\s+([A-Z][A-Za-z0-9'&\- ]{{2,80}})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:{'|'.join(p.replace(' ', r'\s+') for p in _EVENT_PREFIXES)})\s+([A-Z][A-Za-z0-9'&\- ]{{2,80}})",
        re.IGNORECASE,
    ),
]

_EVENT_SUFFIXES = {
    "race",
    "divide",
    "challenge",
    "tour",
    "trail",
    "duro",
    "dash",
    "bikingman",
    "brevet",
    "ultra",
    "rally",
    "odyssey",
    "trailscotland",
}

_EVENT_CONNECTORS = {
    "and", "the", "of", "del", "de", "du", "la", "le", "y", "x", "&", "no",
}

_EVENT_STOPWORDS = {
    "what", "which", "should", "could", "would", "recommend", "use", "bring",
    "setup", "set", "up", "do", "i", "you", "for", "at", "in", "to", "my",
    "best", "good", "bike", "bags", "lights", "tyres", "tires", "wheels",
    "drivetrain", "sleep", "sleeping", "system",
}

_EVENT_HINTS: Dict[str, List[str]] = {
    "transiberica": [
        "road ultra race",
        "endurance road bikepacking",
        "long distance across Spain",
        "lightweight setup",
        "heat",
    ],
    "atlas mountain race": [
        "mountainous",
        "off-road",
        "remote",
        "night riding",
        "long climbs",
    ],
    "gb duro": [
        "off-road bikepacking ultra",
        "mountainous",
        "remote",
        "rough terrain",
    ],
    "silk road mountain race": [
        "mountainous",
        "mtb ultra",
        "remote",
        "high altitude",
        "rough terrain",
    ],
    "tour divide": [
        "off-road bikepacking ultra",
        "long distance",
        "remote",
        "mixed dirt roads",
    ],
    "badlands": [
        "gravel ultra race",
        "arid",
        "heat",
        "remote",
    ],
}

_ARCHETYPE_ADJACENCY: Dict[str, List[str]] = {
    "mountain_gravel_ultra": ["gravel_ultra", "mountain_offroad_ultra"],
    "gravel_ultra": ["mountain_gravel_ultra", "offroad_bikepacking_ultra"],
    "mountain_road_ultra": ["road_ultra"],
    "road_ultra": ["mountain_road_ultra"],
    "desert_mtb_ultra": ["mountain_mtb_ultra", "desert_offroad_ultra"],
    "mountain_mtb_ultra": ["mtb_ultra", "mountain_offroad_ultra"],
    "mtb_ultra": ["mountain_mtb_ultra", "offroad_bikepacking_ultra"],
    "desert_offroad_ultra": ["mountain_offroad_ultra", "desert_mtb_ultra"],
    "mountain_offroad_ultra": ["offroad_bikepacking_ultra", "mountain_mtb_ultra"],
}

_COMPONENT_PATTERNS: Dict[str, List[str]] = {
    "lights": [
        "light", "lights", "lighting", "dynamo", "dynamo hub", "son",
        "supernova", "k-lite", "klite", "exposure", "rear light", "front light",
    ],
    "tyres": [
        "tyre", "tyres", "tire", "tires", "tubeless", "casing", "width", "2.2", "2.35",
        "continental", "gp5000", "maxxis", "ardent", "nobby nic", "schwalbe", "g-one",
    ],
    "bags": [
        "bag", "bags", "frame bag", "seat pack", "handlebar roll", "cargo",
        "apidura", "tailfin", "ortlieb", "restrap", "geosmina",
    ],
    "sleep_system": [
        "sleep", "sleep system", "bivy", "bivvy", "quilt", "sleeping bag", "mat", "pad",
    ],
    "drivetrain": [
        "drivetrain", "groupset", "group set", "group", "cassette", "chainring",
        "gearing", "gear ratio", "grx", "sram", "shimano",
    ],
    "wheels": [
        "wheel", "wheels", "rim", "rims", "hub", "hubs", "wheelset",
    ],
    "bike_type": [
        "bike", "bike type", "frame", "hardtail", "gravel bike", "mtb", "mountain bike", "road", "custom build",
    ],
}

_COMPONENT_QUERY_PHRASES: Dict[str, str] = {
    "lights": "lighting setup, dynamo, front light, rear light, charging",
    "tyres": "tyres, tire width, tubeless, casing",
    "bags": "bikepacking bags, frame bag, seat pack, handlebar roll",
    "sleep_system": "sleep setup, bivy, quilt, sleeping kit",
    "drivetrain": "drivetrain, cassette, chainring, gearing, groupset",
    "wheels": "wheels, wheelset, rims, hubs",
    "bike_type": "bike type, frame, platform, gravel bike, mtb, hardtail",
}


def _append_unique(items: List[str], value: Optional[str]) -> None:
    if not value:
        return
    value = value.strip()
    if value and value not in items:
        items.append(value)


def _has_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _count_titleish_words(words: List[str]) -> int:
    return sum(
        1
        for word in words
        if word.lower() not in _EVENT_CONNECTORS and _TITLEISH_TOKEN_RE.match(word)
    )

def _clean_event_candidate(text: str) -> str:
    candidate = re.sub(r"\s+", " ", (text or "").strip(" \t\r\n?.,:;!()[]{}\"'"))
    candidate = _EVENT_FRAGMENT_RE.sub("", candidate).strip(" \t\r\n?.,:;!()[]{}\"'")
    return candidate


def _looks_like_event_name(candidate: str) -> bool:
    if not candidate:
        return False

    words = candidate.split()
    if not (1 <= len(words) <= 8):
        return False

    lowered_words = [w.lower() for w in words]
    if all(w in _EVENT_STOPWORDS for w in lowered_words):
        return False

    has_digit = any(ch.isdigit() for ch in candidate)
    has_suffix = any(w.lower() in _EVENT_SUFFIXES for w in words)
    titleish_count = _count_titleish_words(words)

    if has_digit or has_suffix or titleish_count >= 2:
        return True

    return len(words) == 1 and bool(re.match(r"^[A-Z][A-Za-z0-9'&\-]{3,}$", words[0]))


def _score_event_candidate(candidate: str) -> int:
    words = candidate.split()
    score = 0

    if any(ch.isdigit() for ch in candidate):
        score += 3
    if any(w.lower() in _EVENT_SUFFIXES for w in words):
        score += 4

    score += min(_count_titleish_words(words), 4)

    if 1 <= len(words) <= 5:
        score += 2
    if candidate.lower() in KNOWN_EVENTS:
        score += 10

    return score


def _extract_capitalized_spans(text: str) -> List[str]:
    if not text:
        return []

    tokens = _WORD_TOKEN_RE.findall(text)
    spans: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            spans.append(" ".join(current))
            current = []

    for token in tokens:
        if re.match(r"^[?.,:;!()]$", token):
            flush()
            continue

        lower = token.lower()
        is_connector = lower in _EVENT_CONNECTORS
        is_titleish = bool(_TITLEISH_TOKEN_RE.match(token))
        is_short_alias = bool(_ALLCAPS_SHORT_RE.match(token))

        if is_titleish or is_short_alias or (current and is_connector):
            current.append(token)
        else:
            flush()

    flush()

    return [
        cleaned
        for span in spans
        if (cleaned := _clean_event_candidate(span)) and _looks_like_event_name(cleaned)
    ]


def _extract_event_name(user_query: str) -> str:
    text = (user_query or "").strip()
    if not text:
        return "Unknown event"

    lowered = text.lower()

    for alias in sorted(KNOWN_EVENTS, key=len, reverse=True):
        if alias in lowered:
            return KNOWN_EVENTS[alias]

    candidates: List[str] = []

    for pattern in _EVENT_CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            candidate = _clean_event_candidate(match.group(1))
            if _looks_like_event_name(candidate):
                candidates.append(candidate)

    candidates.extend(_extract_capitalized_spans(text))

    if not candidates:
        return "Unknown event"

    normalized = [KNOWN_EVENTS.get(candidate.lower(), candidate) for candidate in candidates]
    return max(normalized, key=_score_event_candidate)

def _query_surface_hint(user_query: str) -> Optional[str]:
    q = f" {(user_query or '').lower()} "
    if " road " in q or " all-road " in q or " all road " in q:
        return "road"
    if " gravel " in q:
        return "gravel"
    if " trail " in q or " mtb " in q or " mountain bike " in q:
        return "trail"
    return None


def _is_valid_event_name(event_name: Optional[str]) -> bool:
    if not event_name:
        return False

    event_name = event_name.strip()
    if not event_name or event_name.lower() == "unknown event":
        return False
    if len(event_name.split()) > 8:
        return False

    lowered = event_name.lower()
    bad_prefixes = (
        "what ", "which ", "how ", "recommend ", "setup ", "set up ",
        "should ", "could ", "would ",
    )
    return not any(lowered.startswith(prefix) for prefix in bad_prefixes)


def _event_hint_descriptors(event_name: Optional[str]) -> List[str]:
    if not _is_valid_event_name(event_name):
        return []
    return _EVENT_HINTS.get(event_name.strip().lower(), [])


def _infer_year_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    match = _YEAR_RE.search(title)
    return int(match.group(0)) if match else None


def _infer_event_from_riders(rec: SetupRecommendation) -> Optional[str]:
    titles = [
        rider.event_title
        for rider in (rec.similar_riders or [])
        if isinstance(rider.event_title, str) and rider.event_title.strip()
    ]
    return Counter(titles).most_common(1)[0][0] if titles else None


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

    event_lower = (rec.event or "").lower()
    rec.similar_riders.sort(
        key=lambda rider: (
            event_lower in (rider.event_title or "").lower(),
            rider.best_score or 0,
            rider.year or 0,
        ),
        reverse=True,
    )
    return rec


def infer_event_archetype(flags: Dict[str, bool]) -> Dict[str, Any]:
    terrain: List[str] = []
    environment: List[str] = []
    format_: List[str] = ["ultra"]

    if flags.get("mountain"):
        terrain.append("mountainous")
    if flags.get("desert"):
        environment.append("desert")
    if flags.get("remote"):
        environment.append("remote")
        format_.append("self_supported")
    if flags.get("cold_hot"):
        environment.append("temperature_swings")
    if flags.get("night"):
        format_.append("night_riding")
    if flags.get("navigation"):
        format_.append("navigation_heavy")

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

    if surface_family == "mtb":
        archetype = "desert_mtb_ultra" if flags.get("desert") else (
            "mountain_mtb_ultra" if flags.get("mountain") else "mtb_ultra"
        )
    elif surface_family == "gravel":
        archetype = "mountain_gravel_ultra" if flags.get("mountain") else "gravel_ultra"
    elif surface_family == "road":
        archetype = "mountain_road_ultra" if flags.get("mountain") else "road_ultra"
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
    text = (text or "").lower()

    keyword_map = {
        "road": ["road race", "paved", "tarmac", "asphalt", "road ultra", "road cycling"],
        "gravel": ["gravel", "gravel race", "dirt road", "fire road", "unbound"],
        "mtb": [
            "mtb", "mountain bike", "singletrack", "technical", "rocky", "hardtail",
            "full suspension", "29er", "trail bike",
        ],
        "off_road": [
            "off-road", "off road", "singletrack", "track", "rocky", "trail",
            "technical", "jeep track", "doubletrack", "rough terrain", "dirt road",
        ],
        "desert": ["desert", "sahara", "arid", "dry", "sand"],
        "mountain": ["mountain", "alpine", "climb", "elevation", "pass", "high mountains"],
        "remote": [
            "remote", "self-supported", "self supported", "unsupported",
            "no services", "minimal resupply",
        ],
        "night": ["night", "dark", "overnight"],
        "cold_hot": ["temperature", "cold", "hot", "heat", "freezing", "temperature swings"],
        "navigation": ["navigation", "gps", "route", "track", "waypoint", "gpx"],
    }
    return {name: _has_any(text, terms) for name, terms in keyword_map.items()}


def _extract_metrics(text: str) -> Dict[str, Optional[int]]:
    km_match = _KM_RE.search(text or "")
    elevation_match = _M_RE.search(text or "")

    return {
        "distance_km": int(km_match.group(1)) if km_match else None,
        "elevation_m": int(elevation_match.group(1)) if elevation_match else None,
    }


def _surface_descriptors(surface_family: str) -> List[str]:
    return {
        "road": ["road ultra race", "paved"],
        "gravel": ["gravel ultra race", "mixed dirt roads"],
        "mtb": ["MTB mountain bike ultra", "rough off-road terrain"],
        "mixed_offroad": ["off-road bikepacking ultra", "mixed rough terrain"],
    }.get(surface_family, [])


def _flag_descriptors(flags: Dict[str, bool]) -> List[str]:
    mapping = {
        "mountain": "mountainous long climbs",
        "desert": "desert arid",
        "remote": "remote minimal resupply",
        "night": "night riding",
        "navigation": "navigation GPS route",
        "cold_hot": "temperature swings",
    }
    return [label for key, label in mapping.items() if flags.get(key)]


def _metric_descriptors(metrics: Dict[str, Optional[int]]) -> List[str]:
    descriptors: List[str] = []
    if metrics.get("distance_km"):
        descriptors.append(f"{metrics['distance_km']} km")
    if metrics.get("elevation_m"):
        descriptors.append(f"{metrics['elevation_m']} m climbing")
    return descriptors


def _build_descriptor_query(
    event_name: str,
    event_context: str,
    user_question: str,
) -> Dict[str, Any]:
    full_text = "\n".join([event_name or "", event_context or "", user_question or ""]).strip()

    flags = _keyword_flags(full_text)
    metrics = _extract_metrics(event_context or "")
    archetype_info = infer_event_archetype(flags)

    archetype = archetype_info["archetype"]
    surface_family = archetype_info["surface_family"]

    descriptors: List[str] = ["self-supported ultra endurance bikepacking race"]

    if _is_valid_event_name(event_name):
        _append_unique(descriptors, event_name.strip())

    surface_hint = _query_surface_hint(user_question)
    if surface_hint == "road":
        _append_unique(descriptors, "road event")
        _append_unique(descriptors, "road-oriented setup")
        _append_unique(descriptors, "faster rolling tyres")
        _append_unique(descriptors, "avoid MTB-style tyre widths")
    elif surface_hint == "gravel":
        _append_unique(descriptors, "gravel event")
        _append_unique(descriptors, "gravel-oriented setup")
    elif surface_hint == "trail":
        _append_unique(descriptors, "trail event")
        _append_unique(descriptors, "trail-oriented setup")

    for hint in _event_hint_descriptors(event_name):
        _append_unique(descriptors, hint)

    for descriptor in _surface_descriptors(surface_family):
        _append_unique(descriptors, descriptor)

    for descriptor in _flag_descriptors(flags):
        _append_unique(descriptors, descriptor)

    for descriptor in _metric_descriptors(metrics):
        _append_unique(descriptors, descriptor)

    base_descriptor = ", ".join(descriptors)
    question_focus = (user_question or "").strip()
    descriptor_with_intent = (
        f"{base_descriptor}. Question focus: {question_focus}"
        if question_focus else base_descriptor
    )

    return {
        "archetype": archetype,
        "surface_family": surface_family,
        "adjacent_archetypes": _ARCHETYPE_ADJACENCY.get(archetype, []),
        "descriptor_query": base_descriptor,
        "descriptor_query_with_intent": descriptor_with_intent,
        "features": {
            "flags": flags,
            "metrics": metrics,
            "archetype_info": archetype_info,
            "event_name_used": _is_valid_event_name(event_name),
            "event_hints_used": _event_hint_descriptors(event_name),
            "surface_hint": surface_hint,
        },
    }


def _classify_query_intent(user_query: str) -> QueryIntent:
    text = (user_query or "").strip().lower()
    if not text:
        return QueryIntent(component="full_setup", confidence=0.0)

    scores = {
        component: sum(1 for pattern in patterns if pattern in text)
        for component, patterns in _COMPONENT_PATTERNS.items()
    }
    scores = {component: score for component, score in scores.items() if score > 0}

    if not scores:
        return QueryIntent(
            component="full_setup",
            confidence=0.25,
            component_terms=[],
            asks_for_recommendation=True,
        )

    best_component = max(scores.items(), key=lambda item: item[1])[0]
    confidence = min(1.0, 0.35 + 0.15 * scores[best_component])

    return QueryIntent(
        component=best_component,
        confidence=confidence,
        component_terms=_COMPONENT_PATTERNS[best_component],
        asks_for_recommendation=True,
    )


def _build_retrieval_intent_bundle(
    descriptor: Dict[str, Any],
    intent: QueryIntent,
) -> RetrievalIntentBundle:
    broad_query = descriptor["descriptor_query"]

    if intent.component == "full_setup":
        return RetrievalIntentBundle(
            intent=intent,
            broad_query=broad_query,
            component_query=descriptor["descriptor_query_with_intent"],
            include_component_query=False,
        )

    component_query = (
        f"{broad_query}. Focus: {_COMPONENT_QUERY_PHRASES.get(intent.component, intent.component)}"
    )

    return RetrievalIntentBundle(
        intent=intent,
        broad_query=broad_query,
        component_query=component_query,
        include_component_query=True,
    )


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
    query_component: str = "full_setup"
    component_hit_count: int = 0
    similar_riders: List[CompactRider]


WRITER_PROMPT = """
You are a bikepacking equipment and ultra-distance cycling expert.

Return only valid JSON matching SetupRecommendation.

Grounding rules:
- ALL gear details must come from similar_riders.
- event_context is only for understanding the target event characteristics.
- query_component tells you what part of the setup the user is asking about.
- component_hit_count tells you how many retrieved riders explicitly mention the requested component.
- Do not invent gear, brands, or specs.
- If a field cannot be grounded, leave it empty/null and explain that in reasoning.
- If component_hit_count is 0 for a component-specific question, say evidence is sparse and avoid specific grounded claims.
- Use similar_riders exactly as provided in the input for the output similar_riders content.
- Prefer concise output:
  - summary: 3-4 sentences
  - reasoning: 3-5 sentences

Output rules:
- event must be the requested event_name
- if the exact event is absent from rider data, clearly say the setup is based on similar events
- recommended_setup should contain as many grounded fields as possible without guessing
""".strip()

writer_model = OpenAIChatModel(settings.writer_model)

writer_agent = Agent(
    model=writer_model,
    output_type=SetupRecommendation,
    system_prompt=WRITER_PROMPT,
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


def _rider_component_hit_count(riders: List[Any], component_terms: List[str]) -> int:
    if not riders or not component_terms:
        return 0

    terms = [t.lower() for t in component_terms if t.strip()]
    hits = 0

    for r in riders:
        parts = []

        for value in [
            getattr(r, "bike_type", None),
            getattr(r, "wheels", None),
            getattr(r, "tyres", None),
            getattr(r, "drivetrain", None),
            getattr(r, "bags", None),
            getattr(r, "sleep_system", None),
        ]:
            if isinstance(value, str) and value.strip():
                parts.append(value)

        for item in getattr(r, "key_items", None) or []:
            if isinstance(item, str) and item.strip():
                parts.append(item)

        for c in getattr(r, "chunks", None) or []:
            text = getattr(c, "text", None) or getattr(c, "content", None) or ""
            if text:
                parts.append(text)

        searchable = " ".join(parts).lower()

        if any(term in searchable for term in terms):
            hits += 1

    return hits


def recommend_setup_with_trace(user_query: str) -> Tuple[SetupRecommendation, CallTrace]:
    with logfire.span("recommender.run", user_query=user_query):
        trace = CallTrace()
        deps = _build_deps(call_trace=trace)

        event_name = _extract_event_name(user_query)
        intent = _classify_query_intent(user_query)

        record_trace_call(
            deps=deps,
            tool_name="intent_classification",
            args={"user_query": user_query},
            result=intent.model_dump(),
            elapsed_ms=0.0,
        )

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

        descriptor = _build_descriptor_query(
            event_name=event_name,
            event_context=event_context_text,
            user_question=user_query,
        )

        retrieval_bundle = _build_retrieval_intent_bundle(
            descriptor=descriptor,
            intent=intent,
        )

        if intent.component != "full_setup" and retrieval_bundle.component_query:
            first_query = retrieval_bundle.component_query
            second_query = retrieval_bundle.broad_query
        else:
            first_query = retrieval_bundle.broad_query
            second_query = retrieval_bundle.component_query

        top_k_riders = DEFAULT_TOP_K_RIDERS
        max_chunks_per_rider = DEFAULT_MAX_CHUNKS_PER_RIDER
        top_k_chunks = DEFAULT_TOP_K_CHUNKS

        retrieval_query = first_query

        record_trace_call(
            deps=deps,
            tool_name="search_similar_riders_attempt",
            args={
                "attempt": 1,
                "query": retrieval_query,
                "query_component": intent.component,
                "top_k_riders": top_k_riders,
                "max_chunks_per_rider": max_chunks_per_rider,
                "top_k_chunks": top_k_chunks,
            },
            result={"ok": True},
            elapsed_ms=0.0,
        )

        riders = time_and_record(
            deps=deps,
            tool_name="search_similar_riders",
            args={
                "query": retrieval_query,
                "query_component": intent.component,
                "component_terms": intent.component_terms,
                "top_k_riders": top_k_riders,
                "max_chunks_per_rider": max_chunks_per_rider,
                "top_k_chunks": top_k_chunks,
            },
            fn=lambda: run_search_similar_riders(
                query=retrieval_query,
                query_component=intent.component,
                component_terms=intent.component_terms,
                top_k_riders=top_k_riders,
                max_chunks_per_rider=max_chunks_per_rider,
                top_k_chunks=top_k_chunks,
                deps=deps,
            ),
        )

        component_hit_count = _rider_component_hit_count(riders, intent.component_terms)

        record_trace_call(
            deps=deps,
            tool_name="component_evidence_check",
            args={
                "query_component": intent.component,
                "component_terms": intent.component_terms,
            },
            result={
                "component_hit_count": component_hit_count,
                "rider_count": len(riders or []),
            },
            elapsed_ms=0.0,
        )

        should_try_fallback = (
            bool(second_query)
            and (
                not riders
                or len(riders) < 3
                or (intent.component != "full_setup" and component_hit_count == 0)
            )
        )

        if should_try_fallback:
            fallback_query = second_query

            record_trace_call(
                deps=deps,
                tool_name="search_similar_riders_attempt",
                args={
                    "attempt": 2,
                    "query": fallback_query,
                    "query_component": intent.component,
                    "top_k_riders": top_k_riders,
                    "max_chunks_per_rider": max_chunks_per_rider,
                    "top_k_chunks": top_k_chunks,
                },
                result={"ok": True},
                elapsed_ms=0.0,
            )

            fallback_riders = time_and_record(
                deps=deps,
                tool_name="search_similar_riders",
                args={
                    "query": fallback_query,
                    "query_component": intent.component,
                    "component_terms": intent.component_terms,
                    "top_k_riders": top_k_riders,
                    "max_chunks_per_rider": max_chunks_per_rider,
                    "top_k_chunks": top_k_chunks,
                },
                fn=lambda: run_search_similar_riders(
                    query=fallback_query,
                    query_component=intent.component,
                    component_terms=intent.component_terms,
                    top_k_riders=top_k_riders,
                    max_chunks_per_rider=max_chunks_per_rider,
                    top_k_chunks=top_k_chunks,
                    deps=deps,
                ),
            )

            fallback_component_hit_count = _rider_component_hit_count(
                fallback_riders,
                intent.component_terms,
            )

            if (
                not riders
                or len(riders) < 3
                or fallback_component_hit_count > component_hit_count
            ):
                riders = fallback_riders
                component_hit_count = fallback_component_hit_count
                retrieval_query = fallback_query

        if not riders:
            raise RuntimeError("No similar riders returned; cannot produce grounded recommendation.")

        compact_riders = _compact_riders(riders)

        writer_input = WriterInput(
            user_query=user_query,
            event_name=event_name,
            event_context=event_context_text[:2500],
            descriptor_query=retrieval_query,
            query_component=intent.component,
            component_hit_count=component_hit_count,
            similar_riders=compact_riders,
        )

        rec = writer_agent.run_sync(writer_input.model_dump_json(indent=2)).output
        rec.similar_riders = riders

        if not rec.event or not rec.event.strip():
            rec.event = event_name

        return _postprocess_recommendation(rec), trace


def recommend_setup(user_query: str) -> SetupRecommendation:
    rec, _trace = recommend_setup_with_trace(user_query)
    return rec