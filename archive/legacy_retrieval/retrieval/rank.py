from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


try:
    from baikpacking.eval.retrieval_eval.reranker import RerankerConfig, rerank_hits
except Exception:
    RerankerConfig = None  # type: ignore
    rerank_hits = None  # type: ignore


@dataclass(frozen=True)
class RankConfig:
    k: int = 10
    oversample: int = 5
    apply_rerank: bool = True
    rerank_cfg: Optional["RerankerConfig"] = None


def dedupe_by_doc_id_keep_best(hits: Sequence[Any]) -> List[Any]:
    best: Dict[int, Any] = {}
    for h in hits:
        did = int(h.doc_id)
        prev = best.get(did)
        if prev is None or float(h.score) > float(prev.score):
            best[did] = h
    return list(best.values())


def rank_candidates(
    query: str,
    hits: Sequence[Any],
    cfg: RankConfig,
    *,
    return_debug: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Given raw hits (oversampled), produce final top-k ranked hits and debug info.

    This mirrors eval behavior but without using qrels (no fallback based on labels).
    """
    # 1) dedupe
    deduped = dedupe_by_doc_id_keep_best(hits)

    # 2) baseline dense
    baseline = sorted(deduped, key=lambda h: float(h.score), reverse=True)

    # 3) rerank (optional)
    reranked = baseline
    rerank_debug = None
    if cfg.apply_rerank and cfg.rerank_cfg is not None and rerank_hits is not None:
        if return_debug:
            reranked, rerank_debug = rerank_hits(query, baseline, cfg.rerank_cfg, return_debug=True)
        else:
            reranked = rerank_hits(query, baseline, cfg.rerank_cfg)

    final = reranked[: cfg.k]

    info = {
        "k": cfg.k,
        "oversample": cfg.oversample,
        "dedupe_in": int(len(hits)),
        "dedupe_out": int(len(deduped)),
        "applied_rerank": bool(cfg.apply_rerank and cfg.rerank_cfg is not None and rerank_hits is not None),
        "rerank_debug_topk": (rerank_debug[: min(cfg.k, 10)] if rerank_debug else None),
    }
    return final, info