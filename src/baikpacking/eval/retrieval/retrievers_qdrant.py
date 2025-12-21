from typing import List

from baikpacking.eval.retrieval.retrievers import RetrievedHit, BaseRetriever


class DenseQdrantRetriever(BaseRetriever):
    """
    Dense semantic retriever backed by Qdrant.

    Uses payload["rider_id"] as the stable doc_id for evaluation.
    Compatible with qdrant-client versions that expose either:
      - client.search(...)
      - client.query_points(...)
    """
    name = "dense_qdrant"

    def __init__(self, client, collection_name: str, embed_fn):
        self.client = client
        self.collection = collection_name
        self.embed_fn = embed_fn

    def search(self, query: str, k: int) -> List[RetrievedHit]:
        vec = self.embed_fn([query])[0]

        # Optional oversampling so we can still return k unique riders
        # even if multiple chunks from the same rider dominate the top.
        retrieve_k = max(k * 5, k)

        # --- Compatibility across qdrant-client versions ---
        if hasattr(self.client, "search"):
            raw_hits = self.client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=retrieve_k,
                with_payload=True,
                with_vectors=False,
            )
        elif hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection,
                query=vec,
                limit=retrieve_k,
                with_payload=True,
                with_vectors=False,
            )
            raw_hits = res.points
        else:
            raise AttributeError(
                "Unsupported qdrant-client: neither 'search' nor 'query_points' exists on QdrantClient."
            )

        # Build hits
        hits: List[RetrievedHit] = []
        for h in raw_hits:
            payload = getattr(h, "payload", None) or {}
            if "rider_id" not in payload:
                raise KeyError("Qdrant payload missing 'rider_id'. Cannot run retrieval eval.")

            hits.append(
                RetrievedHit(
                    doc_id=int(payload["rider_id"]),
                    score=float(getattr(h, "score", 0.0)),
                    point_id=int(getattr(h, "id")),
                    payload=payload,
                )
            )

        # Deduplicate by rider_id (doc_id): keep best-scoring hit per rider
        best_by_rider: dict[int, RetrievedHit] = {}
        for hit in hits:
            prev = best_by_rider.get(hit.doc_id)
            if prev is None or hit.score > prev.score:
                best_by_rider[hit.doc_id] = hit

        deduped = sorted(best_by_rider.values(), key=lambda x: x.score, reverse=True)

        # Return top-k unique riders
        return deduped[:k]