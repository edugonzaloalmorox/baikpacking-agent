import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from baikpacking.agents.orchestration_models import EventCandidate, EventResolutionResult
from baikpacking.db.db_connection import get_pg_connection


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

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MIN_FUZZY_SCORE = 0.82
_MIN_WEAK_SCORE = 0.58

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


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_event_text(text: Optional[str]) -> str:
    candidate = _strip_accents(text or "").lower()
    candidate = _YEAR_RE.sub(" ", candidate)
    candidate = re.sub(r"[’'`´]", "", candidate)
    candidate = re.sub(r"[^a-z0-9]+", " ", candidate)
    return " ".join(candidate.split())


def _canonicalize_event_title(title: Optional[str]) -> str:
    candidate = _strip_accents(title or "")
    candidate = _YEAR_RE.sub(" ", candidate)
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate.strip(" \t\r\n?.,:;!()[]{}\"'")


def _event_query_variants(user_query: str) -> List[str]:
    variants: List[str] = []
    cleaned = _clean_event_candidate(_strip_requested_count_phrase(user_query))
    raw = (user_query or "").strip()
    extracted = _extract_event_name(raw)

    for value in (cleaned, raw, extracted):
        if value and value not in variants:
            variants.append(value)

    for span in _extract_capitalized_spans(raw):
        if span and span not in variants:
            variants.append(span)

    alias_match = _find_known_event_alias_match(raw)
    if alias_match:
        _, canonical_name = alias_match
        if canonical_name and canonical_name not in variants:
            variants.append(canonical_name)

    return variants


def _title_match_score(query_text: str, title: str) -> Tuple[float, str]:
    normalized_query = _normalize_event_text(query_text)
    normalized_title = _normalize_event_text(title)

    if not normalized_query or not normalized_title:
        return 0.0, "empty"

    if normalized_query == normalized_title:
        return 1.0, "normalized_exact"

    query_tokens = normalized_query.split()
    title_tokens = normalized_title.split()
    if not query_tokens or not title_tokens:
        return 0.0, "empty"

    query_set = set(query_tokens)
    title_set = set(title_tokens)
    overlap = len(query_set & title_set)
    union = len(query_set | title_set)
    jaccard = overlap / union if union else 0.0
    containment = overlap / min(len(query_set), len(title_set)) if query_set and title_set else 0.0
    ratio = SequenceMatcher(None, normalized_query, normalized_title).ratio()

    score = max(
        ratio,
        jaccard,
        0.65 * ratio + 0.35 * jaccard,
        0.55 * ratio + 0.45 * containment,
    )

    reason = "fuzzy"
    if normalized_title in normalized_query or normalized_query in normalized_title:
        score = max(score, 0.93)
        reason = "substring"

    if title_set.issubset(query_set):
        score = max(score, 0.9)
        reason = "semantic_subset"
    elif query_set.issubset(title_set):
        score = max(score, 0.88)
        reason = "semantic_superset"
    elif containment >= 0.66:
        score = max(score, 0.84)
        reason = "semantic_overlap"
    elif jaccard >= 0.45:
        score = max(score, 0.76)
        reason = "weak_overlap"

    if any(token.isdigit() for token in title_tokens) and not any(token.isdigit() for token in query_tokens):
        score = max(score, 0.8)

    return min(score, 1.0), reason


@lru_cache(maxsize=1)
def _load_kb_event_titles() -> Tuple[str, ...]:
    titles = list(dict.fromkeys(KNOWN_EVENTS.values()))
    try:
        with get_pg_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT title
                    FROM public.articles
                    WHERE title IS NOT NULL AND btrim(title) <> ''
                    ORDER BY title;
                    """
                )
                for row in cur.fetchall():
                    title = row[0] if row else None
                    if isinstance(title, str) and title.strip():
                        titles.append(title.strip())
    except Exception:
        # The resolver still works from the hard-coded aliases when the DB is unavailable.
        pass

    normalized_seen: set[str] = set()
    ordered: List[str] = []
    for title in titles:
        normalized = _normalize_event_text(title)
        if normalized and normalized not in normalized_seen:
            normalized_seen.add(normalized)
            ordered.append(title)
    return tuple(ordered)


def _candidate_tier(score: float, exact_hit: bool = False) -> str:
    if exact_hit or score >= 0.95:
        return "trusted_exact"
    if score >= _MIN_FUZZY_SCORE:
        return "fuzzy_candidate"
    if score >= _MIN_WEAK_SCORE:
        return "weak_candidate"
    return "unknown"


def _build_event_candidates(user_query: str) -> List[EventCandidate]:
    candidates: List[EventCandidate] = []
    query_variants = _event_query_variants(user_query)
    kb_titles = _load_kb_event_titles()

    for title in kb_titles:
        canonical_name = _canonicalize_event_title(title)
        if not canonical_name:
            continue

        best_score = 0.0
        best_reason = "unknown"
        for variant in query_variants or [user_query]:
            score, reason = _title_match_score(variant, canonical_name)
            if score > best_score:
                best_score = score
                best_reason = reason

        if not best_score:
            continue

        match_type = _candidate_tier(best_score, exact_hit=best_score >= 0.98)

        candidates.append(
            EventCandidate(
                title=canonical_name,
                canonical_name=canonical_name,
                score=round(best_score, 4),
                match_type=match_type,
                source=f"kb_title:{best_reason}",
            )
        )

    alias_match = _find_known_event_alias_match(user_query)
    if alias_match:
        alias_text, canonical_name = alias_match
        canonical_display = _canonicalize_event_title(canonical_name)
        candidates.append(
            EventCandidate(
                title=canonical_display,
                canonical_name=canonical_display,
                score=1.0,
                match_type="trusted_exact",
                source=f"alias:{alias_text}",
            )
        )

    if not candidates:
        return []

    deduped: Dict[str, EventCandidate] = {}
    for candidate in candidates:
        key = _normalize_event_text(candidate.title)
        existing = deduped.get(key)
        if existing is None or candidate.score > existing.score:
            deduped[key] = candidate

    return sorted(deduped.values(), key=lambda item: (-item.score, item.title))[:5]


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
    text = _normalize_event_text(user_query)
    if not text:
        return None

    for alias in sorted(KNOWN_EVENTS, key=len, reverse=True):
        alias_norm = _normalize_event_text(alias)
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
    query_text = cleaned_query or user_query or ""
    candidate_events = _build_event_candidates(query_text)
    alias_match = _find_known_event_alias_match(query_text)

    if alias_match:
        matched_alias, canonical_name = alias_match
        canonical_display = _canonicalize_event_title(canonical_name)

        alias_candidate = EventCandidate(
            title=canonical_display,
            canonical_name=canonical_display,
            score=1.0,
            match_type="trusted_exact",
            source=f"alias:{matched_alias}",
        )
        candidate_events = [alias_candidate] + [c for c in candidate_events if _normalize_event_text(c.title) != _normalize_event_text(canonical_display)]

        return EventResolutionResult(
            raw_query_event=matched_alias,
            canonical_name=canonical_display,
            display_name=canonical_display,
            requested_count=requested_count,
            match_type="trusted_exact",
            confidence=1.0,
            is_trusted_exact=True,
            candidate_events=candidate_events[:5],
        )

    best_candidate = candidate_events[0] if candidate_events else None
    if best_candidate and best_candidate.score >= _MIN_WEAK_SCORE:
        canonical_name = best_candidate.canonical_name or best_candidate.title
        match_type = best_candidate.match_type
        if match_type == "unknown":
            match_type = _candidate_tier(best_candidate.score)
        return EventResolutionResult(
            raw_query_event=query_text,
            canonical_name=canonical_name,
            display_name=canonical_name,
            requested_count=requested_count,
            match_type=match_type,
            confidence=min(0.99, best_candidate.score),
            is_trusted_exact=match_type == "trusted_exact",
            candidate_events=candidate_events[:5],
        )

    display_name = _extract_event_name(query_text)
    return EventResolutionResult(
        raw_query_event=None if display_name == "Unknown event" else display_name,
        canonical_name=None,
        display_name=display_name,
        requested_count=requested_count,
        match_type="unknown",
        confidence=best_candidate.score if best_candidate else (0.15 if display_name == "Unknown event" else 0.35),
        is_trusted_exact=False,
        candidate_events=candidate_events[:5],
    )
