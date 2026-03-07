import json
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

from baikpacking.agents.recommender_agent import recommend_setup_with_trace

load_dotenv()


def fmt_score(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "NA"


def _get_trace_entries(trace):
    """
    Support both naming conventions:
      - trace.entries (older)
      - trace.calls   (newer, json-serializable)
    """
    if trace is None:
        return []
    entries = getattr(trace, "entries", None)
    if isinstance(entries, list):
        return entries
    calls = getattr(trace, "calls", None)
    if isinstance(calls, list):
        return calls
    return []


def _missing_setup_fields(rs):
    required = ["bike_type", "wheels", "tyres", "drivetrain", "bags", "sleep_system"]
    missing = []
    for k in required:
        v = getattr(rs, k, None)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(k)
    return missing


def main() -> None:
    query = "What lights should I use for the Atlas Mountain Race?"
    rec, trace = recommend_setup_with_trace(query)

    rs = rec.recommended_setup

    # -------------------------
    # Human-readable answer blob (for eval rows)
    # -------------------------
    answer = (
        f"{rec.summary}\n\n"
        f"Bike type: {rs.bike_type}\n"
        f"Wheels: {rs.wheels}\n"
        f"Tyres: {rs.tyres}\n"
        f"Drivetrain: {rs.drivetrain}\n"
        f"Bags: {rs.bags}\n"
        f"Sleep system: {rs.sleep_system}"
    )

    # -------------------------
    # Log blob
    # -------------------------
    log_lines = []
    log_lines.append("Grounding riders:" if rec.similar_riders else "No grounding riders returned")

    for r in rec.similar_riders or []:
        log_lines.append(
            f"- {r.name or 'Unknown'} @ {r.event_title or 'Unknown event'} "
            f"(year={r.year}, score={fmt_score(r.best_score)})"
        )

    if rec.reasoning:
        log_lines.append("\nReasoning:")
        log_lines.append(rec.reasoning)

    missing = _missing_setup_fields(rs)
    if missing:
        log_lines.append("\n⚠ Missing recommended_setup fields:")
        log_lines.append("  " + ", ".join(missing))

    log = "\n".join(log_lines)

    # -------------------------
    # Tool trace
    # -------------------------
    trace_entries = _get_trace_entries(trace)

    # -------------------------
    # Eval row write
    # -------------------------
    eval_row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": query,
        "answer": answer,
        "log": log,
        "output": rec.model_dump(),
        "tool_trace": trace_entries,
    }

    out_path = Path("data/eval/sample_eval_rows.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(eval_row, ensure_ascii=False) + "\n")

    print(f"\nSaved eval row to {out_path}")

    # -------------------------
    # Console output
    # -------------------------
    
    print("\n====================")
    print("BIKEPACKING SETUP")
    print("====================\n")

    print("EVENT:")
    print(rec.event or "(missing)")

    print("\nSUMMARY:")
    print(rec.summary)

    print("\nRECOMMENDED SETUP:")
    print("  Bike type   :", rs.bike_type or "(missing)")
    print("  Wheels      :", rs.wheels or "(missing)")
    print("  Tyres       :", rs.tyres or "(missing)")
    print("  Drivetrain  :", rs.drivetrain or "(missing)")
    print("  Bags        :", rs.bags or "(missing)")
    print("  Sleep system:", rs.sleep_system or "(missing)")

    if missing:
        print("\n⚠ Missing fields:", ", ".join(missing))

    print("\nGROUNDING RIDERS:")
    if not rec.similar_riders:
        print("  (no riders returned)")
    else:
        for r in rec.similar_riders:
            print(
                f"  - {r.name or 'Unknown'} @ {r.event_title or 'Unknown event'} "
                f"(year={r.year}, score={fmt_score(r.best_score)})"
            )

    if rec.reasoning:
        print("\nREASONING:")
        print(rec.reasoning)

    # -------------------------
    # LOOP / TOOL TRACE PRINT
    # -------------------------
    print("\n====================")
    print("AGENT TOOL TRACE")
    print("====================")

    if not trace_entries:
        print("  (no tool calls recorded)")
    else:
        for i, c in enumerate(trace_entries, 1):
            tool = c.get("tool") or c.get("tool_name") or "(unknown)"
            ms = c.get("elapsed_ms", "?")
            args = c.get("args", {})
            result = c.get("result", {})

            print(f"[{i}] {tool} ({ms} ms)")
            print(f"     args   = {args}")
            print(f"     result = {result}")
            print()

    print("--------------------------------------------------")


if __name__ == "__main__":
    main()