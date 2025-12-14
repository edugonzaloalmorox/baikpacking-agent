import json
from pathlib import Path

LABELS_PATH = Path("data/eval/labels.jsonl")
OUT_PATH = Path("data/eval/qrels.jsonl")

import json
from pathlib import Path
from typing import Iterator, Dict, Any, List


def iter_json_objects(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Read a file that is *intended* to be JSONL but may contain:
      - blank lines
      - pretty-printed multi-line JSON objects

    We recover by accumulating lines until we have a complete JSON object.
    """
    buf: List[str] = []
    depth = 0
    in_str = False
    esc = False

    def update_depth(s: str) -> None:
        nonlocal depth, in_str, esc
        for ch in s:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # start collecting
            buf.append(raw)
            update_depth(raw)

            if depth == 0 and buf:
                chunk = "".join(buf).strip()
                buf.clear()
                try:
                    yield json.loads(chunk)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON object near:\n{chunk[:500]}") from e

    if buf:
        chunk = "".join(buf).strip()
        raise ValueError(f"Trailing incomplete JSON object:\n{chunk[:500]}")


def main(
    labels_path: str = "data/eval/labels.jsonl",
    out_path: str = "data/eval/qrels.jsonl",
    min_rel: int = 1,
) -> None:
    labels_file = Path(labels_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_file.open("w", encoding="utf-8") as out:
        for row in iter_json_objects(labels_file):
            qid = row["qid"]
            relevants = row.get("relevants", [])
            relevant_ids = [
                int(r["rider_id"])
                for r in relevants
                if int(r.get("rel", 0)) >= min_rel
            ]
            out.write(json.dumps({"qid": qid, "relevant_ids": relevant_ids}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote qrels: {out_path} (rows={n}, min_rel={min_rel})")


if __name__ == "__main__":
    main()
