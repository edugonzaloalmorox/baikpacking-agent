import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from baikpacking.eval.datasets import load_queries
from baikpacking.embedding.config import Settings
from baikpacking.embedding.qdrant_utils import get_qdrant_client
from baikpacking.embedding.embed import embed_texts
from baikpacking.eval.retrievers_qdrant import DenseQdrantRetriever


def payload_summary(payload: Dict[str, Any]) -> str:
    """
    Build a readable one-liner for labeling.

    Prefer payload['text'] because in your collection it often contains a compact rider line
    including bike/tyres/bags/navigation. Fallback to stitched structured fields.
    """
    txt = (payload.get("text") or "").strip()
    if txt:
        return txt 

    parts = []
    for k in [
        "name",
        "event_title",
        "frame_type",
        "frame_material",
        "wheel_size",
        "tyre_width",
        "electronic_shifting",
    ]:
        v = payload.get(k)
        if v not in (None, "", "None"):
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def overlap_diagnostics(all_rankings: List[List[int]], top_k: int = 20) -> Dict[str, Any]:
    """
    Diagnose 'everything looks the same' by measuring overlap in top-k doc ids across queries.
    """
    if not all_rankings:
        return {"n_queries": 0}

    # frequency of each rider_id in top-k across queries
    freq = Counter()
    for r in all_rankings:
        for rid in r[:top_k]:
            freq[rid] += 1

    n_queries = len(all_rankings)
    most_common = freq.most_common(10)
    # average pairwise Jaccard overlap (approx with first 30 queries to keep it cheap)
    sample = all_rankings[: min(30, n_queries)]
    jaccs = []
    for i in range(len(sample)):
        si = set(sample[i][:top_k])
        for j in range(i + 1, len(sample)):
            sj = set(sample[j][:top_k])
            denom = len(si | sj) or 1
            jaccs.append(len(si & sj) / denom)
    avg_jacc = sum(jaccs) / len(jaccs) if jaccs else 0.0

    return {
        "n_queries": n_queries,
        "top_k_for_diag": top_k,
        "avg_pairwise_jaccard_topk": avg_jacc,
        "most_common_riders_in_topk": [{"rider_id": rid, "count": c} for rid, c in most_common],
    }


def main(
    queries_path: str = "data/eval/queries.jsonl",
    out_path: str = "data/eval/candidates.jsonl",
    candidate_k: int = 50,
    write_diag: bool = True,
) -> None:
    settings = Settings()
    client = get_qdrant_client()

    retriever = DenseQdrantRetriever(
        client=client,
        collection_name=settings.qdrant_collection,
        embed_fn=embed_texts,
    )

    queries = load_queries(queries_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rankings: List[List[int]] = []

    with out.open("w", encoding="utf-8") as f:
        for q in queries:
            qid = q["qid"]
            text = q["query"]
            hits = retriever.search(text, k=candidate_k)

            ranking = [int(h.doc_id) for h in hits]
            all_rankings.append(ranking)

            candidates = []
            for h in hits:
                payload = getattr(h, "payload", None) or {}
                candidates.append(
                    {
                        "rider_id": int(h.doc_id),
                        "point_id": h.point_id,
                        "score": float(h.score),
                        # payload bits for labeling & traceability
                        "name": payload.get("name"),
                        "event_title": payload.get("event_title"),
                        "event_url": payload.get("event_url"),
                        "event_key": payload.get("event_key"),
                        "chunk_index": payload.get("chunk_index"),
                        "frame_type": payload.get("frame_type"),
                        "frame_material": payload.get("frame_material"),
                        "wheel_size": payload.get("wheel_size"),
                        "tyre_width": payload.get("tyre_width"),
                        "electronic_shifting": payload.get("electronic_shifting"),
                        "summary": payload_summary(payload),
                    }
                )

            row = {
                "qid": qid,
                "query": text,
                "candidate_k": candidate_k,
                "retriever": "dense_qdrant",
                "candidates": candidates,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved candidates to {out_path}")

    if write_diag:
        diag = overlap_diagnostics(all_rankings, top_k=min(20, candidate_k))
        diag_path = out.with_suffix(".diag.json")
        diag_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved diagnostics to {diag_path}")


if __name__ == "__main__":
    main()