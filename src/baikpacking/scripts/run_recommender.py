
from baikpacking.agents.recommender_agent import recommend_setup
from pathlib import Path
import pickle


def main() -> None:
    query = "What lights should I use Atlas Mountain Race?"

    rec = recommend_setup(query)


    answer = (
        f"{rec.summary}\n\n"
        f"Bike type: {rec.bike_type}\n"
        f"Wheels: {rec.wheels}\n"
        f"Tyres: {rec.tyres}\n"
        f"Drivetrain: {rec.drivetrain}\n"
        f"Bags: {rec.bags}\n"
        f"Sleep system: {rec.sleep_system}"
    )

    
    log_lines = []

    if rec.similar_riders:
        log_lines.append("Grounding riders:")
        for r in rec.similar_riders:
            log_lines.append(
                f"- {r.name or 'Unknown'} "
                f"@ {r.event_title or 'Unknown event'} "
                f"(year={r.year}, score={r.best_score:.3f})"
            )
    else:
        log_lines.append("No grounding riders returned")

    if rec.reasoning:
        log_lines.append("\nReasoning:")
        log_lines.append(rec.reasoning)

    log = "\n".join(log_lines)


    eval_row = {
        "question": query,
        "instructions": "You are a bikepacking recommender agent.",
        "answer": answer,
        "log": log,
    }

   
    out_path = Path("data/eval/sample_eval_rows.bin")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with out_path.open("rb") as f:
            rows = pickle.load(f)
    else:
        rows = []

    rows.append(eval_row)

    with out_path.open("wb") as f:
        pickle.dump(rows, f)

    print(f"\n Saved eval row to {out_path}")
    print(f" Total rows: {len(rows)}")

  
    print("\n====================")
    print("BIKEPACKING SETUP")
    print("====================\n")

    print("EVENT:")
    print(rec.event)

    print("\nSUMMARY:")
    print(rec.summary)

    print("\nRECOMMENDED SETUP:")
    print("  Bike type   :", rec.bike_type)
    print("  Wheels      :", rec.wheels)
    print("  Tyres       :", rec.tyres)
    print("  Drivetrain  :", rec.drivetrain)
    print("  Bags        :", rec.bags)
    print("  Sleep system:", rec.sleep_system)

    print("\nGROUNDING RIDERS:")
    if not rec.similar_riders:
        print("  (no riders returned)")
    else:
        for r in rec.similar_riders:
            print(
                f"  - {r.name or 'Unknown'} "
                f"@ {r.event_title or 'Unknown event'} "
                f"(year={r.year}, score={r.best_score:.3f})"
            )

    if rec.reasoning:
        print("\nREASONING:")
        print(rec.reasoning)


if __name__ == "__main__":
    main()
