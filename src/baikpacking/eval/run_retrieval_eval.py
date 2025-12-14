from datetime import datetime
from pathlib import Path
from statistics import mean
import json
import time
from typing import Any, Dict, List, Optional, Sequence

from baikpacking.eval.datasets import load_queries, load_qrels
from baikpacking.eval.metrics import (
    hitrate_at_k,
    recall_at_k as set_recall_at_k,
    precision_at_k,
    mrr_at_k,
)
from baikpacking.eval.retrievers_qdrant import DenseQdrantRetriever
from baikpacking.embedding.config import Settings
from baikpacking.embedding.embed import embed_texts
from baikpacking.embedding.qdrant_utils import get_qdrant_client

# Optional reranker
try:
    from baikpacking.eval.reranker import RerankerConfig, rerank_hits
except Exception:  # keep eval runnable even if reranker not present
    RerankerConfig = None  # type: ignore
    rerank_hits = None  # type: ignore


settings = Settings()
client = get_qdrant_client()


def _load_rerank_config(path: str) -> "RerankerConfig":
    """Load reranker configuration from JSON."""
    if RerankerConfig is None:
        raise RuntimeError("Reranker not available. Ensure baikpacking.eval.reranker exists.")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RerankerConfig(**data)


def _dedupe_by_doc_id_keep_best(hits: Sequence[Any]) -> List[Any]:
    """Deduplicate by doc_id (rider_id). Keep the hit with the best dense score."""
    best: Dict[int, Any] = {}
    for h in hits:
        did = int(h.doc_id)
        prev = best.get(did)
        if prev is None or float(h.score) > float(prev.score):
            best[did] = h
    return list(best.values())


def _ids(hits: Sequence[Any], k: int) -> List[int]:
    """Extract top-k doc_ids as ints."""
    return [int(h.doc_id) for h in hits[:k]]


def _preview(hits: Sequence[Any], k: int) -> List[dict]:
    """Small human-readable preview for debugging / reports."""
    out: List[dict] = []
    for h in hits[:k]:
        p = h.payload or {}
        out.append(
            {
                "rider_id": int(h.doc_id),
                "score": float(h.score),
                "event_key": (p.get("event_key") or ""),
                "tyre_width": (p.get("tyre_width") or ""),
                "frame_type": (p.get("frame_type") or ""),
                "electronic_shifting": p.get("electronic_shifting"),
                "text": (p.get("text") or "")[:220],
            }
        )
    return out


def _diag_metrics(retrieved: List[int], relevant: set[int], ks: Sequence[int]) -> Dict[str, float]:
    """Compute diagnostics at multiple cutoffs."""
    diag: Dict[str, float] = {}
    for dk in ks:
        diag[f"hitrate@{dk}"] = hitrate_at_k(retrieved, relevant, dk)
        diag[f"recall@{dk}"] = set_recall_at_k(retrieved, relevant, dk)
        diag[f"precision@{dk}"] = precision_at_k(retrieved, relevant, dk)
        diag[f"mrr@{dk}"] = mrr_at_k(retrieved, relevant, dk)
    return diag


def run(
    queries_path: str,
    qrels_path: str,
    out_dir: str,
    *,
    apply_rerank: bool = True,
    rerank_config_path: str = "data/eval/rerank_config.json",
    oversample_default: int = 5,
    diag_ks: Optional[List[int]] = None,
    mrr_fallback_epsilon: float = 1e-12,
    worst_n: int = 10,
) -> Path:
    """
    Run retrieval evaluation (dense baseline + optional deterministic rerank).

    Notes:
    - Retrieval is oversampled if reranking is enabled.
    - Dedupe is by rider_id (doc_id), keeping the best dense-scoring chunk.
    - Safe fallback:
        * If rerank hurts hitrate@k -> fallback to baseline
        * If hitrate ties, but rerank hurts MRR@k -> fallback to baseline
    - Debug deltas from reranker are stored only on regressions to keep reports small.
    """
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)  # expects {qid: [relevant_ids...]}

    retriever = DenseQdrantRetriever(
        client=client,
        collection_name=settings.qdrant_collection,
        embed_fn=embed_texts,
    )

    cfg: Optional["RerankerConfig"] = None
    if apply_rerank:
        cfg = _load_rerank_config(rerank_config_path)

    per_q: List[dict] = []
    hitrates: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []
    mrrs: List[float] = []

    lat_retrieval: List[float] = []
    lat_rerank: List[float] = []
    lat_total: List[float] = []

    diag_ks = diag_ks or [1, 3, 5, 10, 20]

    for q in queries:
        qid = q["qid"]
        text = q["query"]
        k = int(q.get("k", 10))

        relevant = set(map(int, qrels.get(qid, [])))
        is_labeled = len(relevant) > 0

        oversample = int(getattr(cfg, "oversample", oversample_default)) if cfg is not None else 1
        retrieve_k = max(k, k * oversample)

        t0 = time.perf_counter()

        # 1) Retrieval
        t_retr0 = time.perf_counter()
        hits = retriever.search(text, k=retrieve_k)
        t_retr1 = time.perf_counter()

        # 2) Dedupe
        hits_deduped = _dedupe_by_doc_id_keep_best(hits)

        # 3) Baseline rank (dense only)
        baseline_ranked = sorted(hits_deduped, key=lambda h: float(h.score), reverse=True)
        baseline_ids = _ids(baseline_ranked, k)

        # 4) Rerank
        t_rr0 = time.perf_counter()
        reranked_ranked = baseline_ranked
        reranked_debug: Optional[List[Dict[str, float]]] = None

        if cfg is not None and rerank_hits is not None:
            reranked_ranked, reranked_debug = rerank_hits(  # type: ignore[assignment]
                text, baseline_ranked, cfg, return_debug=True
            )

        reranked_ids = _ids(reranked_ranked, k)
        t_rr1 = time.perf_counter()

        # 5) Safe fallback (hitrate first, then MRR if hitrate ties)
        used_fallback = False
        if is_labeled:
            base_hr = hitrate_at_k(baseline_ids, relevant, k)
            rr_hr = hitrate_at_k(reranked_ids, relevant, k)

            base_mrr = mrr_at_k(baseline_ids, relevant, k)
            rr_mrr = mrr_at_k(reranked_ids, relevant, k)

            rerank_hurts = (rr_hr < base_hr) or (
                rr_hr == base_hr and (rr_mrr + mrr_fallback_epsilon) < base_mrr
            )

            if rerank_hurts:
                final_ids = baseline_ids
                used_fallback = True
            else:
                final_ids = reranked_ids
        else:
            base_hr = rr_hr = None
            base_mrr = rr_mrr = None
            final_ids = reranked_ids

        retrieved = final_ids[:k]
        t1 = time.perf_counter()

        retr_ms = (t_retr1 - t_retr0) * 1000.0
        rr_ms = (t_rr1 - t_rr0) * 1000.0
        total_ms = (t1 - t0) * 1000.0

        lat_retrieval.append(retr_ms)
        lat_rerank.append(rr_ms)
        lat_total.append(total_ms)

        # Final metrics
        if is_labeled:
            hr = hitrate_at_k(retrieved, relevant, k)
            rec = set_recall_at_k(retrieved, relevant, k)
            prec = precision_at_k(retrieved, relevant, k)
            mrr = mrr_at_k(retrieved, relevant, k)

            hitrates.append(hr)
            recalls.append(rec)
            precisions.append(prec)
            mrrs.append(mrr)

            diag = _diag_metrics(retrieved, relevant, diag_ks)
        else:
            hr = rec = prec = mrr = None
            diag = None

        rerank_regressed = bool(
            is_labeled
            and base_hr is not None
            and rr_hr is not None
            and base_mrr is not None
            and rr_mrr is not None
            and (
                rr_hr < base_hr
                or (rr_hr == base_hr and (rr_mrr + mrr_fallback_epsilon) < base_mrr)
            )
        )

        per_q.append(
            {
                "qid": qid,
                "query": text,
                "k": k,
                "labeled": is_labeled,
                "apply_rerank": cfg is not None,
                "retrieve_k": int(retrieve_k),
                "dedupe_in": int(len(hits)),
                "dedupe_out": int(len(hits_deduped)),
                "hitrate_at_k": hr,
                "set_recall_at_k": rec,
                "precision_at_k": prec,
                "mrr_at_k": mrr,
                "latency_ms_retrieval": retr_ms,
                "latency_ms_rerank": rr_ms,
                "latency_ms_total": total_ms,
                "relevant_ids": sorted(relevant),
                "diag": diag,
                "baseline": {
                    "retrieved": baseline_ids,
                    "preview": _preview(baseline_ranked, min(k, 5)),
                    "hitrate_at_k": base_hr,
                    "mrr_at_k": base_mrr,
                },
                "reranked": {
                    "retrieved": reranked_ids,
                    "preview": _preview(reranked_ranked, min(k, 5)),
                    "hitrate_at_k": rr_hr,
                    "mrr_at_k": rr_mrr,
                },
                "final": {
                    "retrieved": retrieved,
                    "used_fallback": used_fallback,
                },
                "rerank_regressed": rerank_regressed,
                "rerank_debug_topk": (
                    (reranked_debug[: min(k, 10)] if reranked_debug is not None else None)
                    if rerank_regressed
                    else None
                ),
            }
        )

    # --- Summary diagnostics (must be AFTER the loop) ---
    labeled_rows = [r for r in per_q if r["labeled"] and r["mrr_at_k"] is not None]
    worst_by_mrr = sorted(labeled_rows, key=lambda r: float(r["mrr_at_k"]))[:worst_n]

    n_fallback = sum(1 for r in per_q if r.get("final", {}).get("used_fallback"))
    n_regressed = sum(1 for r in per_q if r.get("rerank_regressed"))

    worst_queries_by_mrr = [
        {
            "qid": r["qid"],
            "k": r["k"],
            "mrr_at_k": r["mrr_at_k"],
            "hitrate_at_k": r["hitrate_at_k"],
            "baseline_mrr_at_k": r.get("baseline", {}).get("mrr_at_k"),
            "reranked_mrr_at_k": r.get("reranked", {}).get("mrr_at_k"),
            "used_fallback": r.get("final", {}).get("used_fallback"),
            "query": r["query"],
        }
        for r in worst_by_mrr
    ]

    report = {
        "retriever": retriever.name,
        "apply_rerank": cfg is not None,
        "rerank_config_path": rerank_config_path if cfg is not None else None,
        "n_queries": len(queries),
        "n_labeled": len(hitrates),
        "summary": {
            "hitrate_at_k": float(mean(hitrates)) if hitrates else 0.0,
            "set_recall_at_k": float(mean(recalls)) if recalls else 0.0,
            "precision_at_k": float(mean(precisions)) if precisions else 0.0,
            "mrr_at_k": float(mean(mrrs)) if mrrs else 0.0,
            "avg_latency_ms_retrieval": float(mean(lat_retrieval)) if lat_retrieval else 0.0,
            "avg_latency_ms_rerank": float(mean(lat_rerank)) if lat_rerank else 0.0,
            "avg_latency_ms_total": float(mean(lat_total)) if lat_total else 0.0,
            "n_fallback": int(n_fallback),
            "n_rerank_regressed": int(n_regressed),
            "worst_queries_by_mrr": worst_queries_by_mrr,
        },
        "per_query": per_q,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out / f"retrieval_report_{retriever.name}_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved report: {out_path}")
    print(report["summary"])
    return out_path


if __name__ == "__main__":
    run(
        queries_path="data/eval/queries.jsonl",
        qrels_path="data/eval/qrels.jsonl",
        out_dir="reports/retrieval",
        apply_rerank=True,
        rerank_config_path="data/eval/rerank_config.json",
    )