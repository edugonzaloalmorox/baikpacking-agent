import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib


# ---------------------------------------------------------------------------
# Synthetic response ground truth generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Expectation:
    """High-level expectations for a bikepacking setup answer."""
    must_cover: List[str]
    must_include: List[str]
    must_avoid: List[str]
    expected: Dict[str, str]
    tags: List[str]


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _infer_year(name: str, default_year: int) -> int:
    m = re.search(r"\b(20\d{2})\b", name)
    return int(m.group(1)) if m else default_year


def _strip_year(name: str) -> str:
    return re.sub(r"\s*\b(20\d{2})\b\s*", " ", name).strip()


def _extract_distance_km(name: str) -> int | None:
    """
    Extract common race distance suffixes like '200', '360', '500', '550', '600'.
    """
    m = re.search(r"\b(200|300|303|360|500|550|600)\b", name)
    return int(m.group(1)) if m else None


def _infer_surface_and_style(event: str) -> Tuple[str, List[str]]:
    """
    Returns (style, tags). style is one of: road|gravel|trail|mixed|unknown.
    """
    e = event.lower()

    tags: List[str] = []

    # strong cues
    if "audax road" in e or "road" in e or "liège" in e or "lpl" in e or "superbrevet" in e or "supergrevet" in e:
        tags.append("road")
    if "gravel" in e or "gx" in e:
        tags.append("gravel")
    if "trail" in e or "colorado trail" in e:
        tags.append("trail")

    # softer cues
    if "divide" in e or "unknown" in e or "balkan" in e or "transatlantic" in e:
        tags.append("mixed")
    if "sprint" in e or "quick bite" in e:
        tags.append("short")
    if "monster" in e or "accursed" in e:
        tags.append("tough")

    # decide style priority
    if "trail" in tags:
        return "trail", tags
    if "road" in tags and "gravel" not in tags:
        return "road", tags
    if "gravel" in tags and "road" not in tags:
        return "gravel", tags
    if "road" in tags and "gravel" in tags:
        return "mixed", tags
    if "mixed" in tags:
        return "mixed", tags
    return "unknown", tags


def _normalize_question(q: str) -> str:
    """Normalize question text for duplicate detection."""
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = q.replace("’", "'").replace("–", "-").replace("—", "-")
    return q


def _stable_id(prefix: str, text: str) -> str:
    """Stable id derived from content so reruns are idempotent."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _question_variations() -> List[str]:
    """
    Extra suffixes that:
      1) keep questions unique if templates repeat
      2) encourage different answers (constraints)
    """
    return [
        "Constraints: avoid electronic shifting.",
        "Constraints: prioritize waterproof packing.",
        "Constraints: prioritize easy field repairs.",
        "Constraints: optimize for comfort/contact points.",
        "Constraints: focus on night riding (lights + power).",
        "Constraints: minimize kit (only essentials).",
        "Constraints: prioritize low gearing for climbs.",
        "Constraints: prioritize puncture resistance.",
    ]



def _infer_environment_tags(event: str) -> List[str]:
    e = event.lower()
    tags: List[str] = []

    # mountains / cold cues
    if "pyren" in e or "alps" in e or "peaks" in e or "highland" in e or "andes" in e or "vercors" in e:
        tags += ["mountains", "cold"]

    # night / brevet cues
    if "brevet" in e or "audax" in e or "xl" in e or "ultra" in e:
        tags += ["night", "power", "navigation"]

    # wet UK-ish cues
    if "british" in e or "dales" in e or "cotswolds" in e or "celtic" in e or "albion" in e:
        tags += ["wet", "rain"]

    # heat / tropics cues (rough heuristic)
    if "rwanda" in e or "java" in e or "fuego" in e or "patagonia" in e:
        tags += ["heat_or_extremes", "hydration"]

    return list(dict.fromkeys(tags))


def _expectations_for(event: str, default_year: int) -> Expectation:
    """
    Build a constraint-based ground truth for response evaluation.

    This is intentionally *coarse*: it's meant to be stable across different
    valid recommendations, while still checkable by rules or an LLM judge.
    """
    year = _infer_year(event, default_year)
    distance = _extract_distance_km(event)
    base_event = _strip_year(event)

    style, style_tags = _infer_surface_and_style(base_event)
    env_tags = _infer_environment_tags(base_event)

    must_cover = ["bike_type", "wheels", "tyres", "drivetrain", "bags", "sleep_system"]

    must_include = ["tyres", "drivetrain", "bags"]
    must_avoid: List[str] = []

    expected: Dict[str, str] = {}
    tags = list(dict.fromkeys(style_tags + env_tags))

    # Distance-driven expectations
    if distance is not None:
        if distance >= 500:
            tags.append("multi-day")
            must_include += ["lights", "power", "layers"]
            expected["lighting_power"] = "redundant lights + charging plan"
            expected["comfort"] = "sustainable fit/contact points"
        elif 300 <= distance < 500:
            tags.append("overnight_possible")
            must_include += ["nutrition", "spares"]
            expected["spares"] = "tube/plug/boot/quick link"
        else:
            tags.append("shorter")
            # short events: no heavy sleep emphasis
            expected["sleep_system_style"] = "minimal or none (if no overnight)"

    # Style-driven expectations
    if style == "road":
        expected["tyres_style"] = "endurance road, puncture protection"
        expected["bags_style"] = "small + easy food access (top tube)"
        expected["bike_type_hint"] = "endurance road / all-road"
    elif style == "gravel":
        expected["tyres_style"] = "fast rolling gravel + durability"
        expected["bike_type_hint"] = "gravel / ATB"
        expected["bags_style"] = "stable, minimal sway"
    elif style == "trail":
        expected["tyres_style"] = "strong casing + traction"
        expected["bike_type_hint"] = "rigid/hardtail MTB capable"
        expected["bags_style"] = "secure mounting + stability on rough"
        must_include += ["repair", "traction"]
    elif style == "mixed":
        expected["tyres_style"] = "all-round durable mixed-terrain"
        expected["bike_type_hint"] = "all-road / ATB"
        expected["bags_style"] = "balanced access + stability"

    # Environment cues
    if "mountains" in tags:
        must_include += ["low gears", "warm layers"]
        expected["gearing"] = "wide range + low climbing gear"
        expected["layers"] = "cold descent warmth + waterproof"
    if "wet" in tags or "rain" in tags:
        must_include += ["waterproof"]
        expected["weather_protection"] = "rain shell + waterproof bagging/liners"
    if "night" in tags:
        must_include += ["navigation", "backup"]
        expected["navigation"] = "GPS + phone backup"
        expected["redundancy"] = "backup light + backup plan"
    if "hydration" in tags:
        must_include += ["hydration"]
        expected["hydration"] = "capacity plan + electrolytes"

    # De-duplicate must_include while preserving order
    seen = set()
    must_include = [x for x in must_include if not (x in seen or seen.add(x))]

    # A few universal “bad fits” to avoid (light-touch)
    if style == "trail":
        must_avoid += ["slick tyres", "time trial"]
    if style == "road":
        must_avoid += ["MTB tyres"]
    if "short" in tags or "shorter" in tags:
        must_avoid += ["tent", "heavy sleeping bag"]

    return Expectation(
        must_cover=must_cover,
        must_include=must_include,
        must_avoid=must_avoid,
        expected=expected,
        tags=tags,
    )


def _question_templates() -> List[str]:
    """
    Templates to create recommender-style questions.
    We intentionally mix: full setup, subset (tyres/drivetrain/bags), and constraints.
    """
    return [
        "I’m doing {event} {year}. Recommend a complete bikepacking setup for finishing confidently.",
        "For {event} {year}, what bikepacking setup would you recommend (bike, tyres, drivetrain, bags, sleep)?",
        "I’m riding {event} {year}. I want something reliable and easy to repair—what setup should I use?",
        "For {event} {year}, I care most about comfort over long days. Recommend a setup.",
        "I’m doing {event} {year} and I expect harsh conditions. What tyres + drivetrain + bags would you choose?",
        "For {event} {year}, I want a minimal kit. What should I carry and what would you avoid?",
        "I’m preparing {event} {year}. Suggest an optimal lighting + power + navigation strategy and how to pack it.",
    ]


def _create_gt_rows(
    races: List[str],
    *,
    default_year: int,
    per_event: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    templates = _question_templates()
    variations = _question_variations()

    # 1) Dedupe races by (event, year) so repeated items don't create duplicates
    seen_event_keys: set[str] = set()
    unique_races: List[str] = []
    for race in races:
        year = _infer_year(race, default_year)
        event = _strip_year(race)
        key = f"{_slugify(event)}__{year}"
        if key in seen_event_keys:
            continue
        seen_event_keys.add(key)
        unique_races.append(race)

    rows: List[dict] = []
    seen_questions: set[str] = set()

    for race in unique_races:
        year = _infer_year(race, default_year)
        event = _strip_year(race)

        exp = _expectations_for(race, default_year)

        # 2) Prefer unique templates per event when possible
        if per_event <= len(templates):
            chosen_templates = rng.sample(templates, k=per_event)
        else:
            chosen_templates = [rng.choice(templates) for _ in range(per_event)]

        created = 0
        attempts = 0
        max_attempts = per_event * 30  # safety cap

        while created < per_event and attempts < max_attempts:
            attempts += 1

            tmpl = chosen_templates[created] if created < len(chosen_templates) else rng.choice(templates)
            base_q = tmpl.format(event=event, year=year).strip()

            # Add a suffix when templates might collide (or to increase diversity)
            suffix = ""
            if per_event > len(templates) or created > 0:
                suffix = " " + rng.choice(variations)

            question = (base_q + suffix).strip()
            norm_q = _normalize_question(question)

            if norm_q in seen_questions:
                continue

            seen_questions.add(norm_q)

            # 3) Stable row id based on question text (prevents duplicated ids across reruns)
            row_id = _stable_id("resp_gt", f"{event}|{year}|{question}")

            rows.append(
                {
                    "id": row_id,
                    "event": event,
                    "year": year,
                    "question": question,
                    "must_cover": exp.must_cover,
                    "must_include": exp.must_include,
                    "must_avoid": exp.must_avoid,
                    "expected": exp.expected,
                    "tags": exp.tags,
                    "meta": {
                        "generator": "synth_response_gt",
                        "seed": seed,
                        "per_event": per_event,
                        "attempts": attempts,
                    },
                }
            )

            created += 1

        if created < per_event:
            raise RuntimeError(
                f"Could not generate {per_event} unique questions for event={event} year={year}. "
                f"Generated={created}. Try lowering per_event or adding more variations/templates."
            )

    return rows


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _default_races_from_prompt() -> List[str]:
    """
    A compact default list. Replace/extend this with your full list, or pass a file via --races-file.
    """
    return [
        "Further Perseverance Pyrénées 2025",
        "the Log Drivers Waltz 2025",
        "The Land Between 2025",
        "Ardennes Monster 2025",
        "Bentang Jawa 2025",
        "Super-Brevet Berlin-Munich-Berlin 2025",
        "Basajaun 2025",
        "VIA Race 2025",
        "Utrecht Ultra XL 2025",
        "Three Peaks Bike Race 2025",
        "the Bright Midnight 2025",
        "Andean Raid 2025",
        "Transpyrenees by Transibérica 2025",
        "Trans Balkan Race 2025",
        "Dales Divide 2025",
        "GranGuanche Audax Gravel 2025",
        "Race Around Rwanda 2025",
        "GranGuanche Audax Road 2025",
        "GranGuanche Audax Trail 2025",
        "Peninsular Divide 2025",
        "MittelgebirgeClassique 2025",
        "Istra Land 2025",
        "Bohemian Border Bash Race 2025",
        "Race Around The Netherlands GX 2025",
        "GBDURO25",
    ]


def _load_races_file(path: Path) -> List[str]:
    """
    Load races from a text file (one race per line). Empty lines and # comments ignored.
    """
    races: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            races.append(line)
    return races


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic response ground truth JSONL for bikepacking recommender evaluation."
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/eval/response_ground_truth.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--default-year",
        type=int,
        default=2025,
        help="Default year to use when not present in race name.",
    )
    p.add_argument(
        "--per-event",
        type=int,
        default=1,
        help="Number of question variants per event.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--races-file",
        type=str,
        default="",
        help="Optional path to a text file listing races (one per line). If omitted, a small default list is used.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.races_file:
        races = _load_races_file(Path(args.races_file))
    else:
        races = _default_races_from_prompt()

    rows = _create_gt_rows(
        races,
        default_year=args.default_year,
        per_event=args.per_event,
        seed=args.seed,
    )

    out_path = Path(args.out)
    _write_jsonl(out_path, rows)

    print(f"Wrote {len(rows)} synthetic response GT rows to: {out_path}")
    print(f"Events: {len(races)} | Variants/event: {args.per_event} | Seed: {args.seed}")


if __name__ == "__main__":
    main()