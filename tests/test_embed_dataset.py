from baikpacking.embedding.embed import embed_riders_from_json

chunks = embed_riders_from_json()
print(f"Total embedded chunks: {len(chunks)}")
print("Example:", chunks[0]["rider_id"], len(chunks[0]["vector"]))

