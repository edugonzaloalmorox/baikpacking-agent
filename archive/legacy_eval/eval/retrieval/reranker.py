import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from baikpacking.eval.retrievers import RetrievedHit


# -------------------------
# Query parsing (heuristics)
# -------------------------

@dataclass(frozen=True)
class QueryConstraints:
    require_mechanical: bool = False
    avoid_electronic: bool = False
    require_garmin: bool = False
    avoid_suspension: bool = False
    prefer_rigid: bool = False

    # tyres
    tyre_min_mm: Optional[int] = None
    tyre_max_mm: Optional[int] = None
    tyre_min_mtb_in: Optional[float] = None
    tyre_max_mtb_in: Optional[float] = None

    # event hint (optional)
    event_key_hint: Optional[str] = None

    # intent hints
    prioritize_puncture: bool = False
    need_wide_gearing: bool = False


_TYRE_MM_RE = re.compile(r"(\d{2})\s*[-–]\s*(\d{2})\s*mm", re.IGNORECASE)
_TYRE_MIN_MM_RE = re.compile(r"(\d{2})\s*mm\+", re.IGNORECASE)
_TYRE_IN_RE = re.compile(r"(\d(?:\.\d)?)\s*[-–]\s*(\d(?:\.\d)?)\s*['\"]", re.IGNORECASE)


def parse_constraints(query: str) -> QueryConstraints:
    q = query.lower()

    require_mech = ("mechanical shifting" in q) or ("mechanical only" in q)
    avoid_elec = ("avoid electronic" in q) or ("no electronic" in q) or require_mech

    require_garmin = ("garmin only" in q) or ("navigation: garmin" in q) or ("nav: garmin" in q)

    avoid_susp = ("avoid suspension" in q) or ("no suspension" in q)
    prefer_rigid = ("rigid" in q) or avoid_susp

    prioritize_puncture = ("puncture" in q) or ("puncture resistance" in q) or ("flat" in q)
    need_wide_gearing = ("wide range gearing" in q) or ("steep climbs" in q) or ("wide gear range" in q)

    tyre_min_mm = tyre_max_mm = None
    m = _TYRE_MM_RE.search(q)
    if m:
        tyre_min_mm, tyre_max_mm = int(m.group(1)), int(m.group(2))
    else:
        m2 = _TYRE_MIN_MM_RE.search(q)
        if m2:
            tyre_min_mm = int(m2.group(1))
            tyre_max_mm = None  # open upper bound

    tyre_min_in = tyre_max_in = None
    mi = _TYRE_IN_RE.search(q)
    if mi:
        tyre_min_in, tyre_max_in = float(mi.group(1)), float(mi.group(2))

    event_key_hint = None
    if "transcontinental" in q or "tcr" in q:
        event_key_hint = "transcontinental"
    elif "gb duro" in q or "gbduro" in q:
        event_key_hint = "gbd"

    return QueryConstraints(
        require_mechanical=require_mech,
        avoid_electronic=avoid_elec,
        require_garmin=require_garmin,
        avoid_suspension=avoid_susp,
        prefer_rigid=prefer_rigid,
        tyre_min_mm=tyre_min_mm,
        tyre_max_mm=tyre_max_mm,
        tyre_min_mtb_in=tyre_min_in,
        tyre_max_mtb_in=tyre_max_in,
        event_key_hint=event_key_hint,
        prioritize_puncture=prioritize_puncture,
        need_wide_gearing=need_wide_gearing,
    )


# -------------------------
# Payload helpers
# -------------------------

def _payload_text(payload: Dict[str, Any]) -> str:
    return (payload.get("text") or "").lower()

def _payload_event_key(payload: Dict[str, Any]) -> str:
    return (payload.get("event_key") or "").lower()

def _payload_frame_type(payload: Dict[str, Any]) -> str:
    return (payload.get("frame_type") or "").lower()

def _payload_tyre_width(payload: Dict[str, Any]) -> str:
    return (payload.get("tyre_width") or "").lower()

def _payload_electronic(payload: Dict[str, Any]) -> Optional[bool]:
    v = payload.get("electronic_shifting")
    if v is None:
        return None
    return bool(v)


def tyre_bucket_to_range_mm(tyre_width: str) -> Tuple[Optional[int], Optional[int]]:
    s = tyre_width.strip().lower()
    if not s:
        return (None, None)
    if s.endswith("mm+"):
        try:
            return (int(s.replace("mm+", "")), None)
        except ValueError:
            return (None, None)
    m = re.search(r"(\d{2})\s*mm\s*-\s*(\d{2})\s*mm", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = re.search(r"(\d{2})\s*-\s*(\d{2})\s*mm", s)
    if m2:
        return (int(m2.group(1)), int(m2.group(2)))
    m3 = re.search(r"(\d{2})\s*mm", s)
    if m3:
        v = int(m3.group(1))
        return (v, v)
    return (None, None)


def overlaps_range(
    a_min: Optional[int], a_max: Optional[int], b_min: Optional[int], b_max: Optional[int]
) -> bool:
    if a_min is None and a_max is None:
        return True
    if b_min is None and b_max is None:
        return True
    lo1 = a_min if a_min is not None else -10**9
    hi1 = a_max if a_max is not None else 10**9
    lo2 = b_min if b_min is not None else -10**9
    hi2 = b_max if b_max is not None else 10**9
    return not (hi1 < lo2 or hi2 < lo1)


# -------------------------
# Reranker (soft-boost, clamped)
# -------------------------

@dataclass(frozen=True)
class RerankerConfig:
    """
    Soft boosts are additive on top of dense, then clamped so dense always dominates.

    Recommended: keep per-feature boosts tiny (e.g. 0.01-0.03) and clamp total delta.
    """
    oversample: int = 5

    # per-feature deltas (small!)
    w_event_match: float = 0.02

    w_mechanical_ok: float = 0.02
    w_electronic_violation: float = -0.02

    w_garmin_ok: float = 0.02
    w_garmin_violation: float = -0.02

    w_rigid_ok: float = 0.01
    w_suspension_violation: float = -0.02

    w_tyre_match: float = 0.02
    w_tyre_mismatch: float = -0.02

    w_puncture_terms: float = 0.01
    w_wide_gearing_terms: float = 0.01

    w_has_structured_fields: float = 0.005

    # global clamp: rules can at most shift score by this amount
    max_rule_delta_abs: float = 0.05


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rerank_hits(
    query: str,
    hits: List[RetrievedHit],
    cfg: RerankerConfig,
    *,
    return_debug: bool = False,
) -> List[RetrievedHit] | Tuple[List[RetrievedHit], List[Dict[str, float]]]:
    """
    Deterministically rerank hits using query constraints + payload fields/text.

    - Dense score remains the base.
    - Rules contribute small deltas which are globally clamped to ±cfg.max_rule_delta_abs.
    - Optionally returns per-hit debug dict with feature deltas (useful for regressions).
    """
    qc = parse_constraints(query)

    def rule_deltas(h: RetrievedHit) -> Dict[str, float]:
        payload = h.payload or {}
        text = _payload_text(payload)
        event_key = _payload_event_key(payload)
        frame_type = _payload_frame_type(payload)
        tyre_bucket = _payload_tyre_width(payload)
        electronic = _payload_electronic(payload)

        d: Dict[str, float] = {}

        # event match
        if qc.event_key_hint and qc.event_key_hint in event_key:
            d["event_match"] = cfg.w_event_match

        # electronic / mechanical
        if qc.avoid_electronic:
            if electronic is True:
                d["electronic_violation"] = cfg.w_electronic_violation
            elif electronic is False:
                d["mechanical_ok"] = cfg.w_mechanical_ok

        # garmin constraint
        if qc.require_garmin:
            if "garmin" in text:
                d["garmin_ok"] = cfg.w_garmin_ok
            else:
                # only penalize if navigation is explicitly something else
                if any(x in text for x in ["wahoo", "hammerhead", "bryton", "karoo", "coros"]):
                    d["garmin_violation"] = cfg.w_garmin_violation

        # rigid / suspension
        if qc.avoid_suspension or qc.prefer_rigid:
            if "suspension" in frame_type or "full suspension" in frame_type:
                d["suspension_violation"] = cfg.w_suspension_violation
            else:
                if frame_type:
                    d["rigid_ok"] = cfg.w_rigid_ok

        # tyre match (mm buckets only)
        if qc.tyre_min_mm is not None or qc.tyre_max_mm is not None:
            t_min, t_max = tyre_bucket_to_range_mm(tyre_bucket)
            if not (t_min is None and t_max is None):
                ok = overlaps_range(qc.tyre_min_mm, qc.tyre_max_mm, t_min, t_max)
                d["tyre_match" if ok else "tyre_mismatch"] = cfg.w_tyre_match if ok else cfg.w_tyre_mismatch

        # puncture resistance terms
        if qc.prioritize_puncture:
            if any(x in text for x in ["tubeless", "sealant", "plugs", "dynaplug", "dart", "cushcore", "inserts"]):
                d["puncture_terms"] = cfg.w_puncture_terms

        # wide gearing terms
        if qc.need_wide_gearing:
            if any(x in text for x in ["10-52", "10-50", "11-51", "11-50", "11-46", "11-42", "mullet"]):
                d["wide_gearing_terms"] = cfg.w_wide_gearing_terms

        # structured fields presence
        if any(payload.get(k) for k in ["frame_type", "wheel_size", "tyre_width", "electronic_shifting"]):
            d["has_structured"] = cfg.w_has_structured_fields

        return d

    rescored: List[Tuple[float, RetrievedHit, Dict[str, float]]] = []
    for h in hits:
        base = float(h.score)
        deltas = rule_deltas(h)
        delta_sum = sum(deltas.values())
        delta_sum = _clamp(delta_sum, -cfg.max_rule_delta_abs, cfg.max_rule_delta_abs)
        total = base + delta_sum
        rescored.append((total, h, deltas))

    rescored.sort(key=lambda x: x[0], reverse=True)
    ranked = [h for _, h, _ in rescored]

    if return_debug:
        debug = [d for _, _, d in rescored]
        return ranked, debug

    return ranked
