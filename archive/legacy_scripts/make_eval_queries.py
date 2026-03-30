
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Event catalog
# -----------------------------

EVENTS: List[Tuple[str, str, str, str]] = [
    ("transcontinental", "Transcontinental Race", "road/mixed", "endurance road / light gravel"),
    ("gbduro", "GB Duro", "rough gravel/off-road", "hardtail / rigid"),
    ("two-volcano-sprint", "Two Volcano Sprint", "road/fast", "road"),
    ("dorset-divide", "Dorset Divide", "mixed/rough", "hardtail / gravel"),
    ("granguanche-audax-trail", "GranGuanche Audax Trail", "mixed/volcanic", "hardtail / rigid"),
    ("further-elements", "Further Elements", "mixed", "gravel / hardtail"),
    ("tour-de-farce", "Tour de Farce", "mixed", "gravel"),
    ("peninsular-divide", "Peninsular Divide", "off-road", "hardtail / rigid"),
]

DEFAULT_K = 10


# -----------------------------
# Diversity building blocks
# -----------------------------

RIDER_PROFILES = [
    "I’m 37, based in Spain.",
    "I’m 49, based in Belgium.",
    "I’m 61, based in the Netherlands.",
    "I’m 35, based in Switzerland.",
    "I’m 42, based in the UK.",
    "I’m 28, based in Germany.",
]

# Different "intent styles" so embeddings spread out
INTENT_STYLES = [
    "Recommend a complete setup (bike, tyres, drivetrain, bags, sleep, nav).",
    "Give me 3 setup archetypes and when to choose each.",
    "I have a bike already; recommend the rest (tyres/gearing/bags).",
    "Compare two options: gravel vs hardtail for this race.",
    "Show examples of riders/setups that match my constraints.",
    "What would you change for reliability (not speed)?",
    "Optimize for speed (racing), not comfort.",
    "Optimize for comfort and sleep, not speed.",
]

BIKE_PREFS = [
    "I prefer a hardtail.",
    "I want a rigid bike (no suspension).",
    "I’m leaning gravel bike.",
    "I want an endurance road setup with aerobars.",
    "I’m undecided between gravel and hardtail.",
]

DRIVETRAIN_PREFS = [
    "Mechanical shifting only (no electronic).",
    "Electronic shifting is fine.",
    "1x drivetrain only.",
    "2x drivetrain preferred.",
    "Wide range gearing needed for steep climbs.",
]

TYRE_PREFS = [
    "Tyres: 50mm+.",
    "Tyres: 45–50mm.",
    "Tyres: 32–38mm.",
    "Tyres: 2.2–2.4\" MTB.",
    "Prioritize puncture resistance.",
    "Prioritize rolling speed.",
]

BAG_PREFS = [
    "Bags: minimal (fast & light).",
    "Bags: full bikepacking (frame + saddle + bar roll).",
    "Bags: rack / Tailfin style.",
    "Avoid saddle bag (seatpost dropper / clearance).",
]

SLEEP_PREFS = [
    "Sleep: power naps only (no sleep system).",
    "Sleep: minimal (bivy + quilt).",
    "Sleep: comfort (bag + pad).",
]

NAV_POWER_PREFS = [
    "Navigation: Garmin only.",
    "Power: dynamo hub + USB charging.",
    "Power: power banks only (no dynamo).",
]

# “Conditions” add strong signal and help separate queries
CONDITIONS = [
    "Expect lots of rain and mud.",
    "Long night riding, cold temperatures.",
    "Many steep climbs, low cadence.",
    "Rough washboard and chunky gravel.",
    "Hike-a-bike sections likely.",
    "Fast tarmac stretches mixed with gravel.",
]

# Negations / constraints that change embedding a lot
AVOIDS = [
    "Avoid electronic shifting.",
    "Avoid ultralight fragile parts.",
    "Avoid aero bars.",
    "Avoid suspension (keep it simple).",
    "Avoid huge tyres (keep it fast).",
    "Avoid heavy racks.",
]

MUST_HAVES = [
    "Must include a wide gear range.",
    "Must include lights for night riding.",
    "Must include a reliable charging plan.",
    "Must include puncture protection strategy.",
    "Must include a comfortable cockpit.",
]


# Different prompt forms (not just 1 template)
PROMPT_FORMS = [
    # long-form
    "I’m doing {event_name}. {intent} {profile} {condition}\n"
    "Preferences: {prefs}\n"
    "Must-haves: {musts}\n"
    "Avoid: {avoids}",
    # concise
    "{event_name}: {intent} {profile} Constraints: {prefs}. {condition}",
    # compare
    "For {event_name}, compare these setups for my constraints:\n"
    "- Option A: gravel bike\n"
    "- Option B: hardtail\n"
    "{profile} {condition}\n"
    "Constraints: {prefs}\n"
    "Tell me which you’d pick and why.",
    # examples-first
    "Find rider setup examples for {event_name} matching:\n"
    "{prefs}\n"
    "{profile} {condition}",
]


@dataclass(frozen=True)
class EvalQuery:
    qid: str
    query: str
    k: int = DEFAULT_K
    topic: str = "race_setup"
    event_slug: Optional[str] = None
    event_name: Optional[str] = None
    terrain: Optional[str] = None
    archetype: Optional[str] = None


def _sample_unique(rng: random.Random, pool: Sequence[str], n: int) -> List[str]:
    """Sample up to n unique items from pool."""
    n = max(0, min(n, len(pool)))
    return rng.sample(list(pool), k=n)


def _build_prefs(rng: random.Random) -> str:
    """
    Build a preference block with variability in count and categories.
    This is the main driver of embedding diversity.
    """
    parts: List[str] = []

    # Vary how many categories appear (2–5)
    buckets = [BIKE_PREFS, DRIVETRAIN_PREFS, TYRE_PREFS, BAG_PREFS, SLEEP_PREFS, NAV_POWER_PREFS]
    rng.shuffle(buckets)

    # Always include tyres or drivetrain to keep it concrete
    forced = [rng.choice(TYRE_PREFS), rng.choice(DRIVETRAIN_PREFS)]
    parts.extend(forced)

    # Add 1–3 additional buckets
    n_extra = rng.randint(1, 3)
    for bucket in buckets[:n_extra]:
        parts.append(rng.choice(bucket))

    # Occasionally add an "avoid" inside prefs (strong negative signal)
    if rng.random() < 0.35:
        parts.append(rng.choice(AVOIDS))

    # De-dup
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)

    return " ".join(out)


def generate_queries(
    n_per_event: int,
    seed: int = 7,
    start_index: int = 1,
) -> List[EvalQuery]:
    rng = random.Random(seed)
    out: List[EvalQuery] = []
    idx = start_index

    for event_slug, event_name, terrain, archetype in EVENTS:
        for _ in range(n_per_event):
            profile = rng.choice(RIDER_PROFILES)
            intent = rng.choice(INTENT_STYLES)
            condition = rng.choice(CONDITIONS)

            prefs = _build_prefs(rng)

            musts = "; ".join(_sample_unique(rng, MUST_HAVES, rng.randint(1, 2)))
            avoids = "; ".join(_sample_unique(rng, AVOIDS, rng.randint(0, 2)))

            form = rng.choice(PROMPT_FORMS)
            query = form.format(
                event_name=event_name,
                intent=intent,
                profile=profile,
                condition=condition,
                prefs=prefs,
                musts=musts,
                avoids=avoids,
            )

            qid = f"q{idx:04d}"
            idx += 1

            out.append(
                EvalQuery(
                    qid=qid,
                    query=query,
                    k=DEFAULT_K,
                    topic="race_setup",
                    event_slug=event_slug,
                    event_name=event_name,
                    terrain=terrain,
                    archetype=archetype,
                )
            )

    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-per-event", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", type=str, default="data/eval")
    p.add_argument("--write-meta", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    queries = generate_queries(n_per_event=args.n_per_event, seed=args.seed, start_index=1)

    write_jsonl(
        out_dir / "queries.jsonl",
        ({"qid": q.qid, "query": q.query, "k": q.k, "topic": q.topic} for q in queries),
    )

    if args.write_meta:
        write_jsonl(out_dir / "query_meta.jsonl", (asdict(q) for q in queries))


if __name__ == "__main__":
    main()
