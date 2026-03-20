"""
vector_store.py
---------------
FAISS-backed vector store for XYZ chatbot chunk retrieval.

Uses FlatIP (inner product on L2-normalised vectors) which is equivalent
to cosine similarity — ideal for sentence-transformer embeddings.

Supports:
  - Filtered retrieval by intent / channel / outcome
  - Top-k similarity search
  - Persistence (save / load index + metadata)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import faiss

from data_parser import Chunk


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """
    In-memory FAISS IndexFlatIP with a parallel metadata sidecar.

    Parameters
    ----------
    dim : int
        Embedding dimensionality (inferred on first add_chunks call if not set).
    """

    def __init__(self, dim: Optional[int] = None):
        self.dim: Optional[int] = dim
        self._index: Optional[faiss.Index] = None
        self._chunks: List[Chunk] = []          # parallel list, same order as FAISS rows
        self._metadata: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
    ) -> "FAISSVectorStore":
        """
        Add chunks and their pre-computed embeddings to the index.

        Parameters
        ----------
        chunks     : list of Chunk objects (order must match embeddings rows)
        embeddings : float32 array of shape (N, D), L2-normalised
        """
        assert len(chunks) == embeddings.shape[0], (
            f"Chunk count ({len(chunks)}) != embedding rows ({embeddings.shape[0]})"
        )

        dim = embeddings.shape[1]
        if self._index is None:
            self.dim = dim
            self._index = faiss.IndexFlatIP(dim)   # cosine via inner product on unit vecs

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        self._metadata.extend([c.metadata | {"chunk_id": c.chunk_id,
                                              "chunk_type": c.chunk_type,
                                              "interaction_id": c.interaction_id,
                                              "text": c.text}
                                for c in chunks])
        return self

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        filter_intent: Optional[str] = None,
        filter_channel: Optional[str] = None,
        filter_outcome: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k most similar chunks to query_vec.

        Filtering is done post-retrieval on an over-fetch pool (5×k).
        Each result dict contains: score, chunk_id, text, metadata fields.

        Parameters
        ----------
        query_vec      : (1, D) float32 L2-normalised query embedding
        k              : number of results to return
        filter_intent  : only return chunks matching this intent label
        filter_channel : only return chunks matching this channel
        filter_outcome : only return chunks matching this outcome
        chunk_type     : "interaction" | "turn_window" | None (both)
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Over-fetch so we have headroom after filtering
        fetch_k = min(self._index.ntotal, k * 5)
        scores, indices = self._index.search(query_vec.reshape(1, -1), fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]

            # Apply filters
            if filter_intent and meta.get("intent") != filter_intent:
                continue
            if filter_channel and meta.get("channel") != filter_channel:
                continue
            if filter_outcome and meta.get("outcome") != filter_outcome:
                continue
            if chunk_type and meta.get("chunk_type") != chunk_type:
                continue

            results.append({
                "score": float(score),
                "chunk_id": meta["chunk_id"],
                "interaction_id": meta["interaction_id"],
                "chunk_type": meta["chunk_type"],
                "intent": meta.get("intent", "unknown"),
                "channel": meta.get("channel", "unknown"),
                "outcome": meta.get("outcome", "unknown"),
                "text": meta["text"],
            })

            if len(results) >= k:
                break

        return results

    def get_diverse_context(
        self,
        query_vec: np.ndarray,
        k: int = 3,
    ) -> str:
        """
        Retrieve top-k chunks and format them as a single context string
        for the LLM prompt. Deduplicates by interaction_id.
        """
        raw = self.search(query_vec, k=k * 2)

        # Deduplicate: prefer interaction-level chunk per interaction
        seen_interactions: set[int] = set()
        unique: List[Dict] = []
        for r in raw:
            iid = r["interaction_id"]
            if iid not in seen_interactions:
                seen_interactions.add(iid)
                unique.append(r)
            if len(unique) >= k:
                break

        context_parts = []
        for i, r in enumerate(unique, 1):
            context_parts.append(
                f"[Context {i} | Intent: {r['intent']} | Score: {r['score']:.3f}]\n"
                f"{r['text']}"
            )
        return "\n\n---\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """Save FAISS index + metadata sidecar to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "faiss.index"))
        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        print(f"Vector store saved → {directory}")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "FAISSVectorStore":
        """Load a previously saved FAISS vector store."""
        directory = Path(directory)
        index = faiss.read_index(str(directory / "faiss.index"))
        with open(directory / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(directory / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        store = cls(dim=index.d)
        store._index = index
        store._chunks = chunks
        store._metadata = metadata
        print(f"Vector store loaded ← {directory}  ({index.ntotal} vectors)")
        return store

    def __len__(self) -> int:
        return self._index.ntotal if self._index else 0

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore(vectors={len(self)}, "
            f"dim={self.dim}, chunks={len(self._chunks)})"
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from data_parser import parse_interactions, build_chunks
    from metadata_tagger import tag_chunks
    from embedder import EmbeddingPipeline

    input_path = Path(__file__).parent / "Input Decs" / "input.txt"
    interactions = parse_interactions(input_path)
    chunks = build_chunks(interactions, window_size=2)
    chunks = tag_chunks(chunks, interactions)

    pipeline = EmbeddingPipeline()
    pipeline.fit(chunks)

    store = FAISSVectorStore()
    store.add_chunks(chunks, pipeline.embeddings)
    print(store)

    # Test retrieval
    test_query = "i lost my card and need a replacement"
    qvec = pipeline.encode_query(test_query)
    results = store.search(qvec, k=3)
    print(f"\nTop 3 results for: '{test_query}'")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['chunk_id']} | intent={r['intent']}")
        print(f"    {r['text'][:120]}...")
