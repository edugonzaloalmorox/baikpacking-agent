from baikpacking.embedding.qdrant_utils import search_riders_grouped

query = "Best brand for dynamo lights"
riders = search_riders_grouped(query, top_k_riders=5)

print(f"Query: {query!r}")
print(f"Got {len(riders)} riders:\n")

for i, r in enumerate(riders, start=1):
    print(f"{i}. score={r['best_score']:.4f} | rider_id={r['rider_id']}")
    print(f"   name:  {r.get('name')}")
    print(f"   event: {r.get('event_title')}")
    print(f"   url:   {r.get('event_url')}")
    print("   top chunks:")
    for ch in r["chunks"]:
        snippet = ch["text"].replace("\n", " ")[:160]
        print(f"     - ({ch['score']:.4f}) {snippet!r}")
    print()
