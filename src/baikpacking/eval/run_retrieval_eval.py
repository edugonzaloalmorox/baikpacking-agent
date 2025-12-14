from pathlib import Path
from statistics import mean
import json
import time
from datetime import datetime

from baikpacking.eval.datasets import load_queries, load_qrels
from baikpacking.eval.metrics import (
    hitrate_at_k,
    recall_at_k as set_recall_at_k,
    precision_at_k,
    mrr_at_k,
    # ndcg_at_k,  # optional (requires graded relevance mapping)
)

from baikpacking.eval.retrievers_qdrant import DenseQdrantRetriever
from baikpacking.embedding.qdrant_utils import get_qdrant_client
from baikpacking.embedding.config import Settings
from baikpacking.embedding.embed import embed_texts

settings = Settings()
client = get_qdrant_client()


def run(queries_path: str, qrels_path: str, out_dir: str) -> Path:
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)  # expects {qid: [relevant_ids...]}

    retriever = DenseQdrantRetriever(
        client=client,
        collection_name=settings.qdrant_collection,
        embed_fn=embed_texts,
    )

    per_q = []
    hitrates = []
    recalls = []
    precisions = []
    mrrs = []
    lats = []

    # If you want additional fixed-k diagnostics, set these:
    diag_ks = [1, 3, 5, 10, 20]

    for q in queries:
        qid = q["qid"]
        text = q["query"]
        k = int(q.get("k", 10))
        relevant = set(map(int, qrels.get(qid, [])))
        is_labeled = len(relevant) > 0

        t0 = time.perf_counter()
        hits = retriever.search(text, k=k)
        retrieved = [h.doc_id for h in hits]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if is_labeled:
            hr = hitrate_at_k(retrieved, relevant, k)
            rec = set_recall_at_k(retrieved, relevant, k)
            prec = precision_at_k(retrieved, relevant, k)
            mrr = mrr_at_k(retrieved, relevant, k)

            hitrates.append(hr)
            recalls.append(rec)
            precisions.append(prec)
            mrrs.append(mrr)
            lats.append(latency_ms)
        else:
            hr = rec = prec = mrr = None

        # Optional: compute multiple ks for diagnostics even if query has its own k
        diag = None
        if is_labeled:
            diag = {
                f"hitrate@{dk}": hitrate_at_k(retrieved, relevant, dk)
                for dk in diag_ks
            }
            diag.update({
                f"recall@{dk}": set_recall_at_k(retrieved, relevant, dk)
                for dk in diag_ks
            })
            diag.update({
                f"precision@{dk}": precision_at_k(retrieved, relevant, dk)
                for dk in diag_ks
            })
            diag.update({
                f"mrr@{dk}": mrr_at_k(retrieved, relevant, dk)
                for dk in diag_ks
            })

        per_q.append(
            {
                "qid": qid,
                "k": k,
                "labeled": is_labeled,
                "hitrate_at_k": hr,
                "set_recall_at_k": rec,
                "precision_at_k": prec,
                "mrr_at_k": mrr,
                "latency_ms": latency_ms,
                "retrieved": retrieved[:k],
                "relevant_ids": sorted(relevant),
                "diag": diag,  # can be None
            }
        )

    report = {
        "retriever": retriever.name,
        "n_queries": len(queries),
        "n_labeled": len(hitrates),
        "summary": {
            "hitrate_at_k": float(mean(hitrates)) if hitrates else 0.0,
            "set_recall_at_k": float(mean(recalls)) if recalls else 0.0,
            "precision_at_k": float(mean(precisions)) if precisions else 0.0,
            "mrr_at_k": float(mean(mrrs)) if mrrs else 0.0,
            "avg_latency_ms": float(mean(lats)) if lats else 0.0,
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
    )
