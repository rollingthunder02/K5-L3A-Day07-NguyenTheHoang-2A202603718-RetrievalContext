"""Three retrieval methods over the same Day 6 snapshot."""
import os
import sys
from pathlib import Path

from baseline import TfidfIndex


class DenseRetriever:
    def __init__(self, day6_root):
        from dotenv import load_dotenv

        day6_starter = Path(day6_root).resolve() / "starter"
        sys.path.insert(0, str(day6_starter))
        try:
            from src.embeddings import MyEmbedder
            from src.store import PgVectorStore
        finally:
            sys.path.pop(0)

        load_dotenv(day6_starter / ".env")
        self.embedder = MyEmbedder(os.environ["EMBEDDING_MODEL"])
        self.store = PgVectorStore(os.environ["DATABASE_URL"], self.embedder.dimensions)

    def search(self, query, *, filters=None, top_k=3):
        vector = self.embedder.embed_query(query)
        rows = self.store.search("B", vector, top_k=top_k, filters=filters)
        return [{**row, "id": row["chunk_id"], "source_id": row["document_id"]}
                for row in rows]


class LexicalRetriever:
    def __init__(self, chunks):
        self.index = TfidfIndex(chunks)

    def search(self, query, *, filters=None, top_k=3):
        if filters:
            raise ValueError("TF-IDF trong lab này chưa hỗ trợ filter.")
        return self.index.search(query, top_k=top_k)


def fuse(dense_hits, lexical_hits, top_k, constant=60):
    chunks, scores, ranks = {}, {}, {}
    for method, hits in (("dense", dense_hits), ("tfidf", lexical_hits)):
        for rank, chunk in enumerate(hits, start=1):
            cid = chunk["id"]
            chunks[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1 / (constant + rank)
            ranks.setdefault(cid, {})[method] = rank
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [{**chunks[cid], "score": scores[cid], "strategy": "hybrid",
             "branch_ranks": ranks[cid]} for cid in ordered[:top_k]]


class HybridRetriever:
    def __init__(self, dense, lexical, candidate_k=10):
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k

    def search(self, query, *, filters=None, top_k=3):
        if filters:
            raise ValueError("Hybrid trong lab này chưa hỗ trợ filter.")
        limit = max(self.candidate_k, top_k)
        dense = self.dense.search(query, top_k=limit)
        lexical = self.lexical.search(query, top_k=limit)
        return fuse(dense, lexical, top_k)
