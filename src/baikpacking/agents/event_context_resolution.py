import re
from typing import Any, Dict, List, Optional, Tuple

from baikpacking.agents.event_resolution import _event_hint_descriptors, _is_valid_event_name
from baikpacking.agents.orchestration_models import EventContextSummary, EventResolutionResult
from baikpacking.agents.writer_input import _event_context_to_text
from baikpacking.tools.event_context import run_event_web_search_sync


_KM_RE = re.compile(r"(\d{2,4})\s*km\b", re.IGNORECASE)
_M_RE = re.compile(r"(\d{3,5})\s*m\b", re.IGNORECASE)

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


def _append_unique(items: List[str], value: Optional[str]) -> None:
    if not value:
        return
    value = value.strip()
    if value and value not in items:
        items.append(value)


def _has_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


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


def _query_surface_hint(user_query: str) -> Optional[str]:
    q = f" {(user_query or '').lower()} "
    if " road " in q or " all-road " in q or " all road " in q:
        return "road"
    if " gravel " in q:
        return "gravel"
    if " trail " in q or " mtb " in q or " mountain bike " in q:
        return "trail"
    return None


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


def _infer_event_family(web_context_text: str, descriptor: Dict[str, Any], trusted_exact: bool) -> Tuple[Optional[str], float]:
    surface_family = descriptor.get("surface_family")
    archetype = descriptor.get("archetype")
    text = (web_context_text or "").lower()

    if trusted_exact:
        return archetype, 0.8
    if not text.strip():
        return None, 0.1
    if "europe" in text and ("crossing" in text or "across" in text):
        return "trans-europe ultra", 0.35
    if surface_family and surface_family != "unknown":
        return f"{surface_family} ultra", 0.25
    return "general bikepacking ultra", 0.2


def _infer_similar_events(event_resolution: EventResolutionResult, descriptor: Dict[str, Any]) -> List[str]:
    if event_resolution.is_trusted_exact:
        return []

    surface_family = descriptor.get("surface_family")
    if surface_family == "road":
        return ["Transcontinental Race", "Transiberica"]
    if surface_family == "gravel":
        return ["Badlands", "Transiberica"]
    if surface_family in {"mtb", "mixed_offroad"}:
        return ["Tour Divide", "Atlas Mountain Race"]
    return []


def fetch_event_context_summary(
    event_resolution: EventResolutionResult,
    deps: Any,
) -> EventContextSummary:
    event_context_obj = run_event_web_search_sync(
        event_title=event_resolution.display_name,
        deps=deps,
    )
    web_context_text = _event_context_to_text(event_context_obj)

    descriptor = _build_descriptor_query(
        event_name=event_resolution.display_name,
        event_context=web_context_text,
        user_question=event_resolution.display_name,
    )
    event_family, family_confidence = _infer_event_family(
        web_context_text=web_context_text,
        descriptor=descriptor,
        trusted_exact=event_resolution.is_trusted_exact,
    )

    return EventContextSummary(
        requested_event_name=event_resolution.display_name,
        web_context_text=web_context_text,
        similar_events=_infer_similar_events(event_resolution, descriptor),
        event_family=event_family,
        family_confidence=family_confidence,
        archetype=descriptor.get("archetype"),
        surface_family=descriptor.get("surface_family"),
        features=descriptor.get("features", {}),
    )
