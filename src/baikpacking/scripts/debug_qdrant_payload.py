from __future__ import annotations

from typing import Any, Optional

from qdrant_client.http import models as rest

from baikpacking.embedding.qdrant_utils import get_qdrant_client
from baikpacking.embedding.config import Settings


def _print_point(point: Any) -> None:
    print("\n--- POINT ID (Qdrant internal) ---")
    print(point.id)
    print("\n--- PAYLOAD ---")
    for k, v in (point.payload or {}).items():
        print(f"{k}: {v}")


def main(rider_id: int, page_size: int = 256, max_pages: int = 200) -> None:
    settings = Settings()
    client = get_qdrant_client()

    print("Collection:", settings.qdrant_collection)
    target_int = int(rider_id)
    target_str = str(rider_id)

    # -------- Attempt 1: server-side filter (fast) --------
    try:
        flt = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="rider_id",
                    match=rest.MatchValue(value=target_int),
                )
            ]
        )

        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=flt,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if points:
            _print_point(points[0])
            return

        # If no results, try string match (sometimes payload stored as string)
        flt = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="rider_id",
                    match=rest.MatchValue(value=target_str),
                )
            ]
        )
        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=flt,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if points:
            _print_point(points[0])
            return

        print(f"No point found via filter for rider_id={rider_id}. Falling back to scan...")

    except Exception as e:
        # This catches the 400 Bad Request you’re seeing
        print(f"Filter scroll failed ({type(e).__name__}). Falling back to scan...")

    # -------- Attempt 2: client-side scan (always works) --------
    offset: Optional[Any] = None
    for page in range(max_pages):
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for p in points:
            payload = p.payload or {}
            rid = payload.get("rider_id")
            if rid == target_int or str(rid) == target_str:
                print(f"Found rider_id={rider_id} on page {page+1}")
                _print_point(p)
                return

        if offset is None:
            break

    raise RuntimeError(f"Could not find rider_id={rider_id} after scanning {max_pages} pages")


if __name__ == "__main__":
    main(rider_id=15875524168037015678)
