from baikpacking.embedding.embed import embed_texts

vecs = embed_texts(["hello world", "bikepacking is awesome"])
print(len(vecs), len(vecs[0]))