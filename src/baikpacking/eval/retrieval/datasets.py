from pathlib import Path
from typing import Dict, List
import json

def _load_jsonl(path: str | Path) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def load_queries(path: str | Path) -> List[dict]:
    return _load_jsonl(path)

def load_qrels(path: str | Path) -> Dict[str, List[int]]:
  
    rows = _load_jsonl(path)
    return {r["qid"]: r.get("relevant_ids", []) for r in rows}
