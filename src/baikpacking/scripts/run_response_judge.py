import argparse
import asyncio
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence

from baikpacking.agents.response_judge_agent import EvaluationResult, judge_one


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_WEB_TOOL_NAMES: set[str] = {
    "search_similar_riders",
    "render_grounding_riders",
    "event_web_search",
}


# ---------------------------------------------------------------------------
# CLI / IO
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    input_path: Path
    output_path: Path
    ground_truth_path: Path
    max_rows: int
    concurrency: int


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/eval/sample_eval_rows.bin",
        help="Path to .bin (pickle) eval rows",
    )
    parser.add_argument(
        "--output",
        default="reports/response/response_judge_report.json",
        help="Path to JSON report",
    )
    parser.add_argument(
        "--ground-truth",
        default="data/eval/response_ground_truth.jsonl",
        help="Path to response_ground_truth.jsonl",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max rows to judge (0 = all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of parallel judge calls (bounded).",
    )
    args = parser.parse_args()
    return RunConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        ground_truth_path=Path(getattr(args, "ground_truth")),
        max_rows=int(args.max),
        concurrency=int(args.concurrency),
    )



def load_rows(path: Path, max_rows: int = 0) -> list[dict]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    if max_rows and max_rows > 0:
        return rows[:max_rows]
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ground truth join
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundTruthIndex:
    by_id: dict[str, dict]
    by_question: dict[str, dict]


def load_ground_truth_index(path: Path) -> GroundTruthIndex:
    """
    Load response_ground_truth.jsonl and build indices:
      - by_id: id -> gt_obj
      - by_question: question -> gt_obj
    """
    by_id: dict[str, dict] = {}
    by_question: dict[str, dict] = {}

    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from e

            gt_id = obj.get("id")
            q = obj.get("question")
            if isinstance(gt_id, str) and gt_id:
                by_id[gt_id] = obj
            if isinstance(q, str) and q:
                by_question[q] = obj

    return GroundTruthIndex(by_id=by_id, by_question=by_question)


def fallback_ground_truth_for_row(question: str) -> dict:
    return {
        "type": "rubric_only",
        "must_cover": ["bike_type", "wheels", "tyres", "drivetrain", "bags", "sleep_system"],
        "must_include": ["tyres", "drivetrain", "bags"],
        "must_avoid": [],
        "notes": "No event-specific ground truth available. Judge only for coverage + safety (no invented event facts) + usefulness.",
        "question": question,
    }

def find_ground_truth_obj(row: dict, gt_index: GroundTruthIndex) -> Optional[dict]:
    """
    Resolve a ground truth object for a given eval row.

    Priority:
      1) explicit id keys in the eval row
      2) match by exact question string
    """
    for key in ("ground_truth_id", "gt_id", "response_gt_id", "gt", "ground_truth"):
        val = row.get(key)
        # If val is already a dict (sometimes pipelines embed GT directly)
        if isinstance(val, dict) and val:
            return val
        # If val is an id
        if isinstance(val, str) and val in gt_index.by_id:
            return gt_index.by_id[val]

    q = row.get("question")
    if isinstance(q, str) and q:
        return gt_index.by_question.get(q)

    return None


def serialize_ground_truth(gt_obj: Optional[dict]) -> str:
    """Serialize GT for the judge prompt."""
    if not gt_obj:
        return ""
    return json.dumps(gt_obj, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Trace + tool detection
# ---------------------------------------------------------------------------

def get_trace(row: dict) -> Any:
    """Prefer structured messages if present, else fallback to text log."""
    if row.get("messages") is not None:
        return row["messages"]
    return row.get("log", "")


def _extract_tool_name(m: dict) -> Optional[str]:
    tool = m.get("tool_name") or m.get("name") or m.get("tool")
    return tool if isinstance(tool, str) and tool else None


def _extract_openai_tool_calls(m: dict) -> list[str]:
    out: list[str] = []
    for tc in (m.get("tool_calls") or []):
        if isinstance(tc, dict):
            fn = (tc.get("function") or {}).get("name")
            if isinstance(fn, str) and fn:
                out.append(fn)
    return out


def detect_tool_calls(messages_or_log: Any, names_for_text_scan: Sequence[str]) -> list[str]:
    """
    Extract tool names from an agent trace.

    Supports:
    - list[dict] structured messages (recommended)
    - dict
    - plain text log fallback (regex scan for known tool names)

    Returns stable-order deduplicated list.
    """
    tool_calls: list[str] = []

    if isinstance(messages_or_log, list):
        for m in messages_or_log:
            if not isinstance(m, dict):
                continue
            t = _extract_tool_name(m)
            if t:
                tool_calls.append(t)
            tool_calls.extend(_extract_openai_tool_calls(m))

    elif isinstance(messages_or_log, dict):
        t = _extract_tool_name(messages_or_log)
        if t:
            tool_calls.append(t)
        tool_calls.extend(_extract_openai_tool_calls(messages_or_log))

    else:
        text = str(messages_or_log)
        for name in names_for_text_scan:
            if re.search(rf"\b{re.escape(name)}\b", text):
                tool_calls.append(name)

    # Deduplicate, stable order
    seen: set[str] = set()
    out: list[str] = []
    for t in tool_calls:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def used_web_tools(tool_calls: Sequence[str], web_tool_names: set[str]) -> bool:
    return any(t in web_tool_names for t in tool_calls)


def make_trace_for_judge(tool_calls: Sequence[str]) -> dict:
    """
    Keep judge prompts small: pass only a compact summary, not full tool outputs/logs.
    """
    return {"tool_calls": list(tool_calls)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def to_checklist_dict(res: EvaluationResult) -> dict:
    """
    Keep reporting logic in one place.
    Update this when EvaluationResult changes.
    """
    return {
        "meets_constraints": res.meets_constraints,
        "covers_required_components": res.covers_required_components,
        "includes_required_items": res.includes_required_items,
        "avoids_forbidden_items": res.avoids_forbidden_items,
        "does_not_invent_event_facts": res.does_not_invent_event_facts,
        "output_is_actionable": res.output_is_actionable,
    }


def to_row_report(
    *,
    row_index: int,
    question: str,
    res: EvaluationResult,
    tool_calls: list[str],
    web_used: bool,
    gt_id: Optional[str],
) -> dict:
    return {
        "row_index": row_index,
        "question": question,
        "ground_truth_id": gt_id,
        "score_0_to_6": res.score_0_to_6,
        "checklist": to_checklist_dict(res),
        "notes": res.notes,
        "tags": res.tags,
        "tool_calls": tool_calls,
        "web_tools_used": web_used,
    }


def build_report(*, input_path: Path, judged_rows: list[dict], scores: list[int]) -> dict:
    n = len(judged_rows)
    pct_web = (sum(1 for x in judged_rows if x["web_tools_used"]) / n) if n else 0.0
    return {
        "input": str(input_path),
        "n_rows": n,
        "avg_score_0_to_6": mean(scores) if scores else 0.0,
        "pct_web_tools_used": pct_web,
        "rows": judged_rows,
    }


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

async def judge_single_row(
    *,
    sem: asyncio.Semaphore,
    row_index: int,
    row: dict,
    gt_index: GroundTruthIndex,
    web_tool_names: set[str],
    names_for_text_scan: Sequence[str],
) -> tuple[dict, int]:
    async with sem:
        question = row.get("question", "")
        answer = row.get("answer", "")
        instructions = row.get("instructions", "")

        trace = get_trace(row)
        tool_calls = detect_tool_calls(trace, names_for_text_scan=names_for_text_scan)
        web_used = used_web_tools(tool_calls, web_tool_names=web_tool_names)

        # Prompt shrinking: pass only tool-call summary to the judge
        trace_for_judge = make_trace_for_judge(tool_calls)

        # Join ground truth
        gt_obj = find_ground_truth_obj(row, gt_index)
        if not gt_obj:
            gt_obj = fallback_ground_truth_for_row(question)
        ground_truth = serialize_ground_truth(gt_obj)

        gt_id = gt_obj.get("id") if isinstance(gt_obj, dict) else None

        res: EvaluationResult = await judge_one(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            messages=trace_for_judge, 
            instructions=instructions,
        )

        row_report = to_row_report(
            row_index=row_index,
            question=question,
            res=res,
            tool_calls=tool_calls,
            web_used=web_used,
            gt_id=gt_id if isinstance(gt_id, str) else None,
        )
        return row_report, res.score_0_to_6


async def judge_rows(
    rows: list[dict],
    *,
    concurrency: int,
    gt_index: GroundTruthIndex,
    web_tool_names: set[str],
) -> tuple[list[dict], list[int]]:
    sem = asyncio.Semaphore(concurrency)
    names_for_text_scan = sorted(web_tool_names)

    tasks = [
        judge_single_row(
            sem=sem,
            row_index=i,
            row=r,
            gt_index=gt_index,
            web_tool_names=web_tool_names,
            names_for_text_scan=names_for_text_scan,
        )
        for i, r in enumerate(rows)
    ]
    results = await asyncio.gather(*tasks)

    judged = [rr for rr, _ in results]
    scores = [sc for _, sc in results]
    return judged, scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    cfg = parse_args()

    gt_index = load_ground_truth_index(cfg.ground_truth_path)
    rows = load_rows(cfg.input_path, max_rows=cfg.max_rows)

    judged, scores = await judge_rows(
        rows,
        concurrency=cfg.concurrency,
        gt_index=gt_index,
        web_tool_names=DEFAULT_WEB_TOOL_NAMES,
    )

    report = build_report(input_path=cfg.input_path, judged_rows=judged, scores=scores)
    save_json(cfg.output_path, report)

    print(f"Judged {report['n_rows']} rows | avg score: {report['avg_score_0_to_6']:.2f}/6")
    print(f"Web tools used in {report['pct_web_tools_used']*100:.1f}% of rows")
    print(f"Saved report: {cfg.output_path}")


if __name__ == "__main__":
    asyncio.run(main())