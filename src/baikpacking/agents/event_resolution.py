import re
from typing import Dict, List, Optional, Tuple

from baikpacking.agents.orchestration_models import EventResolutionResult


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
    "kromvojoj": "Kromvojoj",
    "kromvojoj race": "Kromvojoj",
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

_EVENT_LEADING_FILLER_RE = re.compile(
    r"^(?:for\s+|at\s+|in\s+|doing\s+|ride\s+|riding\s+|race\s+|racing\s+)+",
    re.IGNORECASE,
)

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
        "long distance across Spain or Europe",
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

_BAD_EVENT_CANDIDATE_PREFIXES = (
    "recommend",
    "show",
    "give",
    "find",
    "suggest",
    "tell",
    "what",
    "which",
    "best",
)

_REQUESTED_COUNT_WORDS = (
    "bike",
    "bikes",
    "tyre",
    "tyres",
    "tire",
    "tires",
    "bag",
    "bags",
    "wheel",
    "wheels",
    "option",
    "options",
    "setup",
    "setups",
)

_REQUESTED_COUNT_PREFIX_RE = re.compile(
    rf"\b(?:give me|give|recommend|suggest|show me|show|find me|find|list|top|best)\s+(?:me\s+)?(?P<count>[1-9]|1\d|20)\s+(?P<unit>{'|'.join(_REQUESTED_COUNT_WORDS)})(?:\s+options?)?\b",
    re.IGNORECASE,
)

_REQUESTED_COUNT_GENERAL_RE = re.compile(
    rf"\b(?P<count>[1-9]|1\d|20)\s+(?P<unit>{'|'.join(_REQUESTED_COUNT_WORDS)})(?:\s+options?)?\b",
    re.IGNORECASE,
)


def _count_titleish_words(words: List[str]) -> int:
    return sum(
        1
        for word in words
        if word.lower() not in _EVENT_CONNECTORS and _TITLEISH_TOKEN_RE.match(word)
    )


def _clean_event_candidate(text: str) -> str:
    candidate = re.sub(r"\s+", " ", (text or "").strip(" \t\r\n?.,:;!()[]{}\"'"))
    candidate = _EVENT_FRAGMENT_RE.sub("", candidate).strip(" \t\r\n?.,:;!()[]{}\"'")
    candidate = _EVENT_LEADING_FILLER_RE.sub("", candidate).strip()
    return candidate


def _looks_like_event_name(candidate: str) -> bool:
    if not candidate:
        return False

    candidate = candidate.strip()
    if candidate.replace(" ", "").isdigit():
        return False
    lowered = candidate.lower()

    if any(lowered.startswith(prefix + " ") or lowered == prefix for prefix in _BAD_EVENT_CANDIDATE_PREFIXES):
        return False

    words = candidate.split()
    if not (1 <= len(words) <= 8):
        return False

    lowered_words = [w.lower() for w in words]
    if all(w in _EVENT_STOPWORDS for w in lowered_words):
        return False

    if len(words) == 2 and lowered_words[0] in _BAD_EVENT_CANDIDATE_PREFIXES and words[1].isdigit():
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


def _extract_known_event_alias(user_query: str) -> Optional[str]:
    match = _find_known_event_alias_match(user_query)
    return match[1] if match else None


def _extract_requested_count(user_query: str) -> Optional[int]:
    text = (user_query or "").strip()
    if not text:
        return None

    for pattern in (_REQUESTED_COUNT_PREFIX_RE, _REQUESTED_COUNT_GENERAL_RE):
        match = pattern.search(text)
        if match:
            value = int(match.group("count"))
            if value > 0:
                return value
    return None


def _strip_requested_count_phrase(user_query: str) -> str:
    text = (user_query or "").strip()
    if not text:
        return text

    for pattern in (_REQUESTED_COUNT_PREFIX_RE, _REQUESTED_COUNT_GENERAL_RE):
        text = pattern.sub(" ", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


def _find_known_event_alias_match(user_query: str) -> Optional[Tuple[str, str]]:
    text = (user_query or "").strip().lower()
    if not text:
        return None

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())

    for alias in sorted(KNOWN_EVENTS, key=len, reverse=True):
        alias_norm = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
        if not alias_norm:
            continue

        if re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", text):
            return alias, KNOWN_EVENTS[alias]

    return None


def _extract_event_name(user_query: str) -> str:
    text = _strip_requested_count_phrase(user_query).strip()
    if not text:
        return "Unknown event"

    alias_hit = _extract_known_event_alias(text)
    if alias_hit:
        return alias_hit

    lowered = text.lower()

    cleaned_lowered = _clean_event_candidate(lowered)
    alias_hit = _extract_known_event_alias(cleaned_lowered)
    if alias_hit:
        return alias_hit

    query_tokens = set(re.findall(r"[a-z0-9]+", cleaned_lowered))
    best_name: Optional[str] = None
    best_score = 0.0

    for alias, canonical in KNOWN_EVENTS.items():
        alias_tokens = set(re.findall(r"[a-z0-9]+", alias.lower()))
        if not alias_tokens:
            continue

        inter = len(query_tokens & alias_tokens)
        union = len(query_tokens | alias_tokens)
        score = inter / union if union else 0.0

        if alias_tokens.issubset(query_tokens):
            score += 0.5

        if score > best_score:
            best_score = score
            best_name = canonical

    if best_name and best_score >= 0.4:
        return best_name

    candidates: List[str] = []

    for pattern in _EVENT_CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            candidate = _clean_event_candidate(match.group(1))
            if _looks_like_event_name(candidate):
                candidates.append(candidate)

    candidates.extend(_extract_capitalized_spans(text))
    candidates = [c for c in candidates if _looks_like_event_name(c)]

    if not candidates:
        return "Unknown event"

    normalized = [KNOWN_EVENTS.get(candidate.lower(), candidate) for candidate in candidates]
    return max(normalized, key=_score_event_candidate)


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


def resolve_event(user_query: str) -> EventResolutionResult:
    requested_count = _extract_requested_count(user_query)
    cleaned_query = _strip_requested_count_phrase(user_query)
    display_name = _extract_event_name(cleaned_query or user_query)
    alias_match = _find_known_event_alias_match(cleaned_query or user_query)

    if alias_match:
        matched_alias, canonical_name = alias_match
        alias_norm = re.sub(r"[^a-z0-9]+", " ", matched_alias.lower()).strip()
        canonical_norm = re.sub(r"[^a-z0-9]+", " ", canonical_name.lower()).strip()

        if alias_norm == canonical_norm:
            return EventResolutionResult(
            raw_query_event=matched_alias,
            canonical_name=canonical_name,
            display_name=display_name,
            requested_count=requested_count,
            match_type="exact",
            confidence=0.98,
            is_trusted_exact=True,
            )

        return EventResolutionResult(
            raw_query_event=matched_alias,
            canonical_name=canonical_name,
            display_name=display_name,
            requested_count=requested_count,
            match_type="alias",
            confidence=0.93,
            is_trusted_exact=False,
        )

    canonical_name = display_name if display_name in KNOWN_EVENTS.values() else None

    if canonical_name:
        return EventResolutionResult(
            raw_query_event=display_name,
            canonical_name=canonical_name,
            display_name=display_name,
            requested_count=requested_count,
            match_type="exact",
            confidence=0.85,
            is_trusted_exact=True,
        )

    return EventResolutionResult(
        raw_query_event=None if display_name == "Unknown event" else display_name,
        canonical_name=None,
        display_name=display_name,
        requested_count=requested_count,
        match_type="unknown",
        confidence=0.15 if display_name == "Unknown event" else 0.35,
        is_trusted_exact=False,
    )
