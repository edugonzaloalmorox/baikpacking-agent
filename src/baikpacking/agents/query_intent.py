from typing import Any, Dict, List

from baikpacking.agents.models import QueryIntent, RetrievalIntentBundle


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
