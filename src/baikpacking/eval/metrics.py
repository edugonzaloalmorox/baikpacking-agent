from typing import Dict, List, Set
import math

def hitrate_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """HitRate@k (success@k): 1 if any relevant doc appears in top-k, else 0."""
    topk = retrieved[:k]
    return 1.0 if any(d in relevant for d in topk) else 0.0

def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Set Recall@k: |relevant ∩ topk| / |relevant|."""
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / float(len(relevant))

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Precision@k: |relevant ∩ topk| / k."""
    if k <= 0:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / float(k)

def mrr_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """MRR@k: reciprocal rank of first relevant doc in top-k, else 0."""
    for i, d in enumerate(retrieved[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def average_precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """AP@k for binary relevance."""
    if not relevant:
        return 0.0
    hit_count = 0
    sum_precisions = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in relevant:
            hit_count += 1
            sum_precisions += hit_count / i
    return sum_precisions / float(len(relevant))

def dcg_at_k(retrieved: List[int], rels: Dict[int, int], k: int) -> float:
    """DCG@k with graded relevance and log2 discount."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = max(0, int(rels.get(doc_id, 0)))
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(retrieved: List[int], rels: Dict[int, int], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k."""
    dcg = dcg_at_k(retrieved, rels, k)
    ideal_rels = sorted((max(0, int(r)) for r in rels.values()), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:k], start=1):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(i + 1)
    return 0.0 if idcg == 0.0 else dcg / idcg
