import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from baikpacking.agents.recommender_agent import recommend_setup


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file into a list of dicts."""
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict], *, append: bool = False) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Log sanitization (remove web/tool noise)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Lines that often appear in tool / http logs
_WEB_LOG_LINE_RE = re.compile(
    r"(http request:|primp:|duckduckgo|wikipedia\.org|yandex\.com|brave\.com|ddgs|opensearch|GET\s+https?://|POST\s+https?://)",
    re.IGNORECASE,
)

def redact_web_traces(text: str) -> str:
    """
    Remove typical web/tool traces from a reasoning/log blob.
    Keeps the reasoning content if it's not obviously web-log output.
    """
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        if _WEB_LOG_LINE_RE.search(line):
            continue
        # If a line is basically just a URL or includes one, drop it
        if _URL_RE.search(line):
            # Keep the line only if it has meaningful non-url content
            # (heuristic: remove urls then check length)
            stripped = _URL_RE.sub("", line).strip(" -:\t")
            if len(stripped) < 25:
                continue
            # otherwise keep the line but with urls removed
            line = _URL_RE.sub("[url removed]", line)
        cleaned_lines.append(line)

    # Drop excessive blank lines
    out = "\n".join(cleaned_lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


# ---------------------------------------------------------------------------
# Formatting (answer + log)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormattedRec:
    answer: str
    log: str
    structured: Dict[str, Any]


def _safe_get(obj: Any, name: str, default: Any = "") -> Any:
    return getattr(obj, name, default)


def format_recommendation(
    rec: Any,
    *,
    log_mode: str = "riders+structured",
    include_reasoning: bool = False,
    redact_web: bool = True,
) -> FormattedRec:
    """
    Convert recommend_setup() return object into:
      - answer: what the user sees
      - log: grounding info for the judge (riders-only by default)
    """
    event = _safe_get(rec, "event", "") or ""
    summary = _safe_get(rec, "summary", "") or ""

    bike_type = _safe_get(rec, "bike_type", "") or ""
    wheels = _safe_get(rec, "wheels", "") or ""
    tyres = _safe_get(rec, "tyres", "") or ""
    drivetrain = _safe_get(rec, "drivetrain", "") or ""
    bags = _safe_get(rec, "bags", "") or ""
    sleep_system = _safe_get(rec, "sleep_system", "") or ""

    similar_riders = _safe_get(rec, "similar_riders", []) or []

    grounding_lines: List[str] = []
    for r in similar_riders:
        name = _safe_get(r, "name", None) or "Unknown"
        event_title = _safe_get(r, "event_title", None) or "Unknown event"
        year = _safe_get(r, "year", None)
        best_score = _safe_get(r, "best_score", None)

        year_s = str(year) if year is not None else "?"
        score_s = f"{best_score:.3f}" if isinstance(best_score, (int, float)) else "?"
        grounding_lines.append(f"- {name} @ {event_title} (year={year_s}, score={score_s})")

    reasoning = _safe_get(rec, "reasoning", "") or ""
    if redact_web:
        reasoning = redact_web_traces(reasoning)

    answer = "\n".join(
        [
            "EVENT:",
            event.strip(),
            "",
            "SUMMARY:",
            summary.strip(),
            "",
            "RECOMMENDED SETUP:",
            f"- Bike type: {bike_type}".strip(),
            f"- Wheels: {wheels}".strip(),
            f"- Tyres: {tyres}".strip(),
            f"- Drivetrain: {drivetrain}".strip(),
            f"- Bags: {bags}".strip(),
            f"- Sleep system: {sleep_system}".strip(),
        ]
    ).strip()

    structured = {
        "event": event,
        "summary": summary,
        "bike_type": bike_type,
        "wheels": wheels,
        "tyres": tyres,
        "drivetrain": drivetrain,
        "bags": bags,
        "sleep_system": sleep_system,
        "n_similar_riders": len(similar_riders),
    }

    log_parts: List[str] = []

    # Always include rider grounding
    if grounding_lines:
        log_parts.append("GROUNDING RIDERS:\n" + "\n".join(grounding_lines))
    else:
        log_parts.append("GROUNDING RIDERS:\n(none returned)")

    if log_mode in {"riders+structured", "full"}:
        log_parts.append("STRUCTURED:\n" + json.dumps(structured, ensure_ascii=False))

    # Only include reasoning if explicitly requested
    if include_reasoning and reasoning:
        log_parts.append("REASONING:\n" + reasoning.strip())

    # Safety: never include raw URLs in logs unless user explicitly wants them
    # (we already redact by default)
    log = "\n\n".join([p for p in log_parts if p.strip()]).strip()

    return FormattedRec(answer=answer, log=log, structured=structured)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build response_eval_rows.jsonl by running the recommender on GT questions."
    )
    p.add_argument("--gt", type=str, default="data/eval/response_ground_truth.jsonl",
                   help="Path to response ground truth JSONL.")
    p.add_argument("--out", type=str, default="data/eval/response_eval_rows.jsonl",
                   help="Output JSONL path.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only process the first N rows.")
    p.add_argument("--append", action="store_true",
                   help="Append to output file (and skip existing ids).")
    p.add_argument("--skip-duplicate-answers", action="store_true",
                   help="Skip rows whose answer hash already exists in output.")

    # New controls
    p.add_argument("--log-mode", type=str, default="riders+structured",
                   choices=["riders-only", "riders+structured", "full"],
                   help="What to include in the eval row log.")
    p.add_argument("--include-reasoning", action="store_true",
                   help="Include agent reasoning in log (sanitized). Default off.")
    p.add_argument("--no-redact-web", action="store_true",
                   help="Disable web-log/url redaction. Not recommended for judge eval.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    gt_path = Path(args.gt)
    out_path = Path(args.out)

    gt_rows = read_jsonl(gt_path)
    if args.limit and args.limit > 0:
        gt_rows = gt_rows[: args.limit]

    existing_ids: Set[str] = set()
    existing_answer_hashes: Set[str] = set()

    if args.append and out_path.exists():
        existing = read_jsonl(out_path)
        for r in existing:
            rid = r.get("id")
            if rid:
                existing_ids.add(rid)
            if args.skip_duplicate_answers:
                ah = r.get("answer_hash")
                if ah:
                    existing_answer_hashes.add(ah)

    run_ts = datetime.now(timezone.utc).isoformat()
    results: List[dict] = []

    processed = 0
    skipped = 0

    for gt in gt_rows:
        rid = gt.get("id")
        question = gt.get("question")

        if not rid or not question:
            skipped += 1
            continue

        if args.append and rid in existing_ids:
            skipped += 1
            continue

        rec = recommend_setup(question)

        formatted = format_recommendation(
            rec,
            log_mode=args.log_mode,
            include_reasoning=args.include_reasoning,
            redact_web=not args.no_redact_web,
        )

        answer_hash = sha256_text(formatted.answer)

        if args.skip_duplicate_answers and answer_hash in existing_answer_hashes:
            skipped += 1
            continue

        row = {
            "id": rid,
            "question": question,
            "instructions": "You are a bikepacking recommender agent. Provide a practical setup recommendation.",
            "answer": formatted.answer,
            "log": formatted.log,
            "answer_hash": answer_hash,
            "meta": {
                "event": gt.get("event"),
                "year": gt.get("year"),
                "tags": gt.get("tags", []),
                "run_at": run_ts,
            },
        }

        results.append(row)
        processed += 1

        if args.skip_duplicate_answers:
            existing_answer_hashes.add(answer_hash)

    write_jsonl(out_path, results, append=args.append)

    print(f"Wrote {processed} eval rows to: {out_path}")
    print(f"Skipped: {skipped} (missing fields / already exists / duplicate answer)")
    print(f"Source GT: {gt_path} | Total GT read: {len(gt_rows)}")


if __name__ == "__main__":
    main()