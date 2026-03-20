"""
embedder.py
-----------
Benchmarks three sentence-transformer models on the XYZ chatbot chunks and picks the
best one based on Silhouette Score on intent clusters.

Models evaluated:
  1. all-MiniLM-L6-v2        — fast, lightweight baseline
  2. BAAI/bge-small-en-v1.5  — strong asymmetric retrieval
  3. intfloat/e5-base-v2     — instruction-tuned, best for Q&A

Usage:
  from embedder import EmbeddingPipeline
  pipeline = EmbeddingPipeline()
  pipeline.fit(chunks)          # benchmarks + picks best model
  embeddings = pipeline.embeddings   # np.ndarray (N, D)
"""

from __future__ import annotations

import os
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from data_parser import Chunk


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CANDIDATE_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-base-v2",
]

# e5 family requires a task prefix for asymmetric retrieval
E5_QUERY_PREFIX = "passage: "   # used when encoding document chunks


def _get_model_cache_dir() -> str:
    """Return local HuggingFace cache dir."""
    return str(Path.home() / ".cache" / "huggingface" / "hub")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _encode_chunks(
    model: SentenceTransformer,
    chunks: List[Chunk],
    model_name: str,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode chunk texts using the given model.
    Applies e5 passage prefix when needed.
    Returns L2-normalised vectors (required for cosine via inner product).
    """
    texts = []
    for chunk in chunks:
        text = chunk.text
        if "e5" in model_name.lower():
            text = E5_QUERY_PREFIX + text
        texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype(np.float32)


def _silhouette(embeddings: np.ndarray, labels: List[str]) -> float:
    """
    Compute Silhouette Score for the embedding space given intent labels.
    Returns -1.0 if fewer than 2 distinct labels.
    """
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    if len(set(encoded)) < 2:
        return -1.0
    return silhouette_score(embeddings, encoded, metric="cosine")


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    """
    Benchmarks multiple sentence-transformer models and selects the best one
    by Silhouette Score on interaction-level intent clusters.

    Attributes
    ----------
    best_model_name : str
        Name of the winning model.
    embeddings : np.ndarray
        Final embeddings for ALL chunks (interaction + turn-window), shape (N, D).
    benchmark_results : list[dict]
        Per-model benchmark details.
    """

    def __init__(self, candidate_models: List[str] = CANDIDATE_MODELS):
        self.candidate_models = candidate_models
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.benchmark_results: list[dict] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        chunks: List[Chunk],
        save_path: Optional[Union[str, Path]] = None,
    ) -> "EmbeddingPipeline":
        """
        Benchmark all candidate models on interaction-level chunks,
        pick winner by Silhouette Score, then embed ALL chunks.

        Parameters
        ----------
        chunks     : full list of Chunk objects (interaction + turn_window)
        save_path  : if given, saves embeddings as .npy to this path
        """
        # Separate interaction-level chunks for benchmarking (one per intent)
        interaction_chunks = [c for c in chunks if c.chunk_type == "interaction"]
        intent_labels = [c.metadata.get("intent", "general_enquiry") for c in interaction_chunks]

        print("\n" + "=" * 60)
        print("  Embedding Model Benchmark")
        print("=" * 60)

        best_score = -999.0
        best_name = None
        results = []

        for model_name in self.candidate_models:
            print(f"\n→ Loading: {model_name}")
            t0 = time.time()
            model = SentenceTransformer(model_name, cache_folder=_get_model_cache_dir())
            load_time = time.time() - t0

            t1 = time.time()
            emb = _encode_chunks(model, interaction_chunks, model_name)
            enc_time = time.time() - t1

            score = _silhouette(emb, intent_labels)
            dim = emb.shape[1]

            result = {
                "model": model_name,
                "silhouette_score": round(float(score), 4),
                "embedding_dim": dim,
                "load_time_s": round(load_time, 2),
                "encode_time_s": round(enc_time, 2),
            }
            results.append(result)

            print(f"   Silhouette Score : {score:.4f}")
            print(f"   Embedding dim    : {dim}")
            print(f"   Load time        : {load_time:.2f}s | Encode: {enc_time:.2f}s")

            if score > best_score:
                best_score = score
                best_name = model_name
                self.best_model = model

            # Free memory for eliminated models
            del model

        self.benchmark_results = results
        self.best_model_name = best_name

        print("\n" + "=" * 60)
        print(f"  ✓ Best model : {best_name}")
        print(f"  ✓ Best score : {best_score:.4f}")
        print("=" * 60 + "\n")

        # Encode ALL chunks with the best model
        print("Encoding all chunks with best model...")
        self.embeddings = _encode_chunks(self.best_model, chunks, self.best_model_name)
        print(f"Embeddings shape: {self.embeddings.shape}")

        if save_path:
            np.save(str(save_path), self.embeddings)
            print(f"Saved embeddings → {save_path}")

        return self

    # ------------------------------------------------------------------
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single retrieval query at inference time.
        Applies e5 query prefix if needed.
        """
        if self.best_model is None:
            raise RuntimeError("Call fit() before encode_query()")

        text = query
        if "e5" in self.best_model_name.lower():
            text = "query: " + query       # asymmetric: query prefix for queries

        vec = self.best_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    def print_benchmark_table(self):
        """Pretty-print benchmark results."""
        print(f"\n{'Model':<35} {'Silhouette':>12} {'Dim':>6} {'Load(s)':>9} {'Enc(s)':>8}")
        print("-" * 75)
        for r in self.benchmark_results:
            mark = " ✓" if r["model"] == self.best_model_name else "  "
            print(
                f"{mark}{r['model']:<33} {r['silhouette_score']:>12.4f} "
                f"{r['embedding_dim']:>6} {r['load_time_s']:>9.2f} "
                f"{r['encode_time_s']:>8.2f}"
            )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from data_parser import parse_interactions, build_chunks
    from metadata_tagger import tag_chunks

    input_path = Path(__file__).parent / "Input Decs" / "input.txt"
    interactions = parse_interactions(input_path)
    chunks = build_chunks(interactions, window_size=2)
    chunks = tag_chunks(chunks, interactions)

    pipeline = EmbeddingPipeline()
    pipeline.fit(chunks, save_path=Path(__file__).parent / "embeddings.npy")
    pipeline.print_benchmark_table()
