"""Sentence embeddings (all-MiniLM-L6-v2, 384-d, L2-normalised) and the embedding job."""
from __future__ import annotations

import numpy as np
import psycopg

from .db import vector_backend

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


class Embedder:
    """Lazy wrapper over fastembed (ONNX runtime — no PyTorch needed)."""

    def __init__(self, model: str = MODEL):
        self.model_name = model
        self._m = None

    def _load(self):
        if self._m is None:
            from fastembed import TextEmbedding
            self._m = TextEmbedding(self.model_name)
        return self._m

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._load().embed(texts)), dtype="f4")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


def embed_missing(conn: psycopg.Connection, embedder: Embedder | None = None, batch: int = 256) -> int:
    """Embed every profile summary that has no embedding yet. Returns the number embedded."""
    embedder = embedder or Embedder()
    kind = vector_backend(conn)
    n = 0
    while True:
        rows = conn.execute("SELECT profile_id, summary FROM profiles WHERE embedding IS NULL AND summary IS NOT NULL "
                            "ORDER BY profile_id LIMIT %s", (batch,)).fetchall()
        if not rows:
            return n
        vecs = embedder.embed([r[1] for r in rows])
        with conn.cursor() as cur:
            for (pid, _), v in zip(rows, vecs):
                if kind == "pgvector":
                    cur.execute("UPDATE profiles SET embedding = %s::vector WHERE profile_id = %s",
                                ("[" + ",".join(f"{x:.6f}" for x in v) + "]", pid))
                else:
                    cur.execute("UPDATE profiles SET embedding = %s WHERE profile_id = %s", (v.tolist(), pid))
        conn.commit()
        n += len(rows)
