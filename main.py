"""
main.py
-------
Orchestrates the full XYZ chatbot RAG pipeline end-to-end:

  1. Parse chatbot conversation transcripts
  2. Tag intents & outcomes
  3. Benchmark embedding models → select best
  4. Build FAISS vector store
  5. Run a conversational RAG demo session
  6. Visualize embeddings with UMAP and t-SNE

Run:
  python main.py
  python main.py --no-rag   # skip interactive demo, viz only
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "Input Decs" / "input.txt"

sys.path.insert(0, str(BASE_DIR))

from data_parser import parse_interactions, build_chunks
from metadata_tagger import tag_chunks, tag_interactions
from embedder import EmbeddingPipeline
from vector_store import FAISSVectorStore
from rag_chain import ConversationalRAGChain
from visualize import run_visualization


# ---------------------------------------------------------------------------
# Demo queries that test the chatbot's known failure points
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    # Card replacement (Interaction 1, 5)
    "I lost my debit card",
    # Travel (Interaction 7) — follow-up tests condensation
    "and I'm travelling to Spain next week",
    # Address change — free text (Interaction 12 failure)
    "I need to change my address to 16 Sycamore Tree, Haswell DH6 2AG",
    # Savings account (Interaction 4 failure — premature handoff)
    "what savings accounts do XYZ offer?",
    # Fraud alert (Interaction 8)
    "I received a suspicious text asking for my PIN",
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(skip_rag: bool = False) -> dict:
    """
    Execute the full pipeline.
    Returns a dict with key artefact paths and metrics.
    """
    print("\n" + "═" * 65)
    print("  XYZ Chatbot — Improved Conversational RAG Pipeline")
    print("═" * 65 + "\n")

    # ── Step 1: Parse ──────────────────────────────────────────────
    print("Step 1 │ Parsing transcripts...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    interactions = parse_interactions(INPUT_FILE)
    interactions = tag_interactions(interactions)
    print(f"         ✓ {len(interactions)} interactions loaded")

    # ── Step 2: Chunk ──────────────────────────────────────────────
    print("\nStep 2 │ Building chunks...")
    chunks = build_chunks(interactions, window_size=2)
    chunks = tag_chunks(chunks, interactions)

    n_int = sum(1 for c in chunks if c.chunk_type == "interaction")
    n_win = sum(1 for c in chunks if c.chunk_type == "turn_window")
    print(f"         ✓ {len(chunks)} chunks total")
    print(f"           ({n_int} interaction-level, {n_win} turn-window)")

    intent_dist = {}
    for c in chunks:
        if c.chunk_type == "interaction":
            intent = c.metadata.get("intent", "unknown")
            intent_dist[intent] = intent_dist.get(intent, 0) + 1
    print(f"\n         Intent distribution (interaction chunks):")
    for intent, cnt in sorted(intent_dist.items()):
        print(f"           {intent:<25s} {cnt}")

    # ── Step 3: Embed ──────────────────────────────────────────────
    print("\nStep 3 │ Benchmarking & fitting embedding models...")
    emb_path = BASE_DIR / "embeddings.npy"
    pipeline = EmbeddingPipeline()
    pipeline.fit(chunks, save_path=emb_path)
    pipeline.print_benchmark_table()

    # ── Step 4: Vector Store ───────────────────────────────────────
    print("\nStep 4 │ Building FAISS vector store...")
    store = FAISSVectorStore()
    store.add_chunks(chunks, pipeline.embeddings)
    store.save(BASE_DIR / "vector_store")
    print(f"         ✓ {store}")

    # ── Step 5: RAG Demo ───────────────────────────────────────────
    if not skip_rag:
        print("\nStep 5 │ Running conversational RAG demo...")
        chain = ConversationalRAGChain(store, pipeline, k=3)
        chain.run_demo(DEMO_QUERIES)
    else:
        print("\nStep 5 │ RAG demo skipped (--no-rag flag)")

    # ── Step 6: Visualize ──────────────────────────────────────────
    print("\nStep 6 │ Generating UMAP + t-SNE visualizations...")
    viz_paths = run_visualization(
        embeddings=pipeline.embeddings,
        chunks=chunks,
        output_dir=BASE_DIR,
        save_png=True,
    )

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ✅  Pipeline complete!")
    print("═" * 65)
    print(f"\n  Best embedding model : {pipeline.best_model_name}")
    print(f"  Total vectors        : {len(store)}")
    print(f"  Embeddings saved     : {emb_path}")
    print(f"\n  Visualizations:")
    for key, path in viz_paths.items():
        print(f"    {key:<15s}: {path}")
    print()

    return {
        "pipeline": pipeline,
        "store": store,
        "viz_paths": viz_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XYZ Chatbot Conversational RAG Pipeline"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG demo session (useful for viz-only runs)",
    )
    args = parser.parse_args()

    run_pipeline(skip_rag=args.no_rag)
