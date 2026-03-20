"""
visualize.py
------------
UMAP and t-SNE visualization of the XYZ chatbot chunk embeddings.

Produces:
  - umap_plot.html    : interactive Plotly scatter (coloured by intent)
  - tsne_plot.html    : interactive Plotly scatter (coloured by intent)
  - umap_plot.png     : static PNG export
  - tsne_plot.png     : static PNG export

Design goals:
  ✓ Inter-document separation  — different intent clusters should be visually distinct
  ✓ Intra-document cohesion    — same-interaction chunks should cluster tightly together
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 10,
    min_dist: float = 0.15,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project embeddings to 2D with UMAP.

    n_neighbors: Controls local vs global structure balance.
                  Smaller = tighter local clusters (good for intra-doc cohesion).
    min_dist:     Controls how tightly points pack together.
    """
    from umap import UMAP
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def run_tsne(
    embeddings: np.ndarray,
    perplexity: Optional[float] = None,
    n_iter: int = 1200,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project embeddings to 2D with t-SNE.

    perplexity is automatically set to min(30, N/3) to avoid having
    perplexity ≥ N which would cause t-SNE to fail on small datasets.
    """
    from sklearn.manifold import TSNE

    n_samples = embeddings.shape[0]
    if perplexity is None:
        # Safe perplexity: must be < n_samples
        perplexity = min(30.0, max(5.0, (n_samples - 1) / 3.0))

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        metric="cosine",
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Build visualization dataframe
# ---------------------------------------------------------------------------

def build_viz_df(
    embeddings: np.ndarray,
    chunks: list,
    projection_2d: np.ndarray,
    method: str,
) -> pd.DataFrame:
    """Build a tidy DataFrame for Plotly from chunks + 2D projections."""
    records = []
    for i, chunk in enumerate(chunks):
        text_preview = chunk.text[:120].replace("\n", " ")
        records.append({
            "x": float(projection_2d[i, 0]),
            "y": float(projection_2d[i, 1]),
            "chunk_id": chunk.chunk_id,
            "interaction_id": chunk.interaction_id,
            "chunk_type": chunk.chunk_type,
            "intent": chunk.metadata.get("intent", "general_enquiry"),
            "channel": chunk.metadata.get("channel", "unknown"),
            "outcome": chunk.metadata.get("outcome", "unknown"),
            "text_preview": text_preview,
            "method": method,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotly visualization
# ---------------------------------------------------------------------------

INTENT_COLOR_MAP = {
    "card_replacement":    "#e74c3c",
    "balance_enquiry":     "#3498db",
    "password_reset":      "#9b59b6",
    "savings_account":     "#2ecc71",
    "direct_debit":        "#f39c12",
    "fraud_alert":         "#e67e22",
    "branch_locator":      "#1abc9c",
    "mortgage":            "#34495e",
    "address_change":      "#c0392b",
    "travel_notification": "#16a085",
    "general_enquiry":     "#95a5a6",
}

SYMBOL_MAP = {
    "interaction": "circle",
    "turn_window": "diamond",
}


def _make_figure(df: pd.DataFrame, title: str) -> go.Figure:
    """
    Build a rich Plotly scatter figure with:
      - Colour by intent
      - Symbol by chunk_type (circle=interaction, diamond=turn_window)
      - Size-encoded by chunk_type (interaction chunks are larger)
      - Hover: chunk_id, intent, channel, outcome, text preview
    """
    df = df.copy()
    df["size"] = df["chunk_type"].map({"interaction": 16, "turn_window": 9})
    df["color"] = df["intent"].map(INTENT_COLOR_MAP).fillna("#95a5a6")
    df["symbol"] = df["chunk_type"].map(SYMBOL_MAP).fillna("circle")

    # Convex hull hulls per interaction (to show intra-doc cohesion visually)
    fig = go.Figure()

    # Add one trace per intent so legend is grouped by intent
    for intent, grp in df.groupby("intent"):
        color = INTENT_COLOR_MAP.get(intent, "#95a5a6")
        for ctype, cgrp in grp.groupby("chunk_type"):
            symbol = SYMBOL_MAP.get(ctype, "circle")
            size = 16 if ctype == "interaction" else 9
            fig.add_trace(go.Scatter(
                x=cgrp["x"],
                y=cgrp["y"],
                mode="markers",
                name=f"{intent} ({ctype})",
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                customdata=np.stack([
                    cgrp["chunk_id"],
                    cgrp["interaction_id"],
                    cgrp["intent"],
                    cgrp["channel"],
                    cgrp["outcome"],
                    cgrp["text_preview"],
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Interaction: %{customdata[1]}<br>"
                    "Intent: %{customdata[2]}<br>"
                    "Channel: %{customdata[3]}<br>"
                    "Outcome: %{customdata[4]}<br>"
                    "<i>%{customdata[5]}...</i>"
                    "<extra></extra>"
                ),
            ))

    # Ellipses per interaction (intra-doc cohesion indicator)
    _add_interaction_ellipses(fig, df)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=22, color="#2c3e50"),
            x=0.5,
        ),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13),
        legend=dict(
            title="Intent (shape = chunk type)",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#dce1e7",
            borderwidth=1,
        ),
        xaxis=dict(showgrid=True, gridcolor="#e0e0e0", zeroline=False, title="Dim 1"),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0", zeroline=False, title="Dim 2"),
        width=1100,
        height=750,
        margin=dict(l=60, r=60, t=80, b=60),
    )
    return fig


def _add_interaction_ellipses(fig: go.Figure, df: pd.DataFrame) -> None:
    """Draw a light ellipse around each interaction's points (intra-doc cohesion)."""
    try:
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        # We'll approximate with a scatter + error ellipse using PCA
        # For Plotly we'll draw a simple convex-hull polygon instead
        import scipy.spatial as spatial

        for iid, grp in df.groupby("interaction_id"):
            if len(grp) < 3:
                continue
            pts = grp[["x", "y"]].values
            try:
                hull = spatial.ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close
                fig.add_trace(go.Scatter(
                    x=hull_pts[:, 0],
                    y=hull_pts[:, 1],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(180,180,220,0.10)",
                    line=dict(color="rgba(100,100,180,0.30)", width=1, dash="dot"),
                    name=f"Interaction {iid}",
                    showlegend=False,
                    hoverinfo="skip",
                ))
            except Exception:
                pass
    except ImportError:
        pass   # scipy optional — skip ellipses if not available


# ---------------------------------------------------------------------------
# Side-by-side comparison figure
# ---------------------------------------------------------------------------

def make_combined_figure(umap_df: pd.DataFrame, tsne_df: pd.DataFrame) -> go.Figure:
    """
    Create a 1×2 subplot comparing UMAP and t-SNE side by side.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["UMAP Projection", "t-SNE Projection"],
        horizontal_spacing=0.08,
    )

    intents = sorted(set(umap_df["intent"].tolist() + tsne_df["intent"].tolist()))

    for col, df in [(1, umap_df), (2, tsne_df)]:
        for intent in intents:
            grp = df[df["intent"] == intent]
            color = INTENT_COLOR_MAP.get(intent, "#95a5a6")
            for ctype, cgrp in grp.groupby("chunk_type"):
                symbol = SYMBOL_MAP.get(ctype, "circle")
                size = 14 if ctype == "interaction" else 8
                fig.add_trace(
                    go.Scatter(
                        x=cgrp["x"], y=cgrp["y"],
                        mode="markers",
                        name=f"{intent} ({ctype})",
                        showlegend=(col == 1),
                        marker=dict(
                            size=size, color=color, symbol=symbol,
                            opacity=0.85,
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate=(
                            f"<b>{intent}</b><br>%{{text}}<extra></extra>"
                        ),
                        text=cgrp["chunk_id"],
                    ),
                    row=1, col=col,
                )

    fig.update_layout(
        title=dict(
            text="XYZ Chatbot Embeddings — UMAP vs t-SNE<br>"
                 "<sup>● interaction-level chunk &nbsp;◆ turn-window chunk</sup>",
            font=dict(size=20, color="#2c3e50"),
            x=0.5,
        ),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(
            title="Intent",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#dce1e7",
            borderwidth=1,
        ),
        width=1400, height=700,
        margin=dict(l=60, r=60, t=110, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_visualization(
    embeddings: np.ndarray,
    chunks: list,
    output_dir: Union[str, Path] = ".",
    save_png: bool = True,
) -> dict:
    """
    Run UMAP and t-SNE, produce Plotly figures, save HTML + PNG.

    Returns dict with output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Visualization] Running UMAP...")
    umap_2d = run_umap(embeddings)

    print("[Visualization] Running t-SNE...")
    tsne_2d = run_tsne(embeddings)

    umap_df = build_viz_df(embeddings, chunks, umap_2d, "UMAP")
    tsne_df = build_viz_df(embeddings, chunks, tsne_2d, "t-SNE")

    # Individual plots
    umap_fig = _make_figure(
        umap_df,
        "XYZ Chatbot — UMAP Embedding Space<br>"
        "<sup>● interaction-level   ◆ turn-window   | coloured by intent</sup>",
    )
    tsne_fig = _make_figure(
        tsne_df,
        "XYZ Chatbot — t-SNE Embedding Space<br>"
        "<sup>● interaction-level   ◆ turn-window   | coloured by intent</sup>",
    )
    combined_fig = make_combined_figure(umap_df, tsne_df)

    paths = {}

    # --- Save HTML (interactive)
    umap_html = output_dir / "umap_plot.html"
    tsne_html = output_dir / "tsne_plot.html"
    combined_html = output_dir / "combined_plot.html"

    umap_fig.write_html(str(umap_html), include_plotlyjs="cdn")
    tsne_fig.write_html(str(tsne_html), include_plotlyjs="cdn")
    combined_fig.write_html(str(combined_html), include_plotlyjs="cdn")

    paths["umap_html"] = umap_html
    paths["tsne_html"] = tsne_html
    paths["combined_html"] = combined_html

    print(f"  ✓ UMAP plot     → {umap_html}")
    print(f"  ✓ t-SNE plot    → {tsne_html}")
    print(f"  ✓ Combined plot → {combined_html}")

    # --- Save PNG (static)
    if save_png:
        try:
            import kaleido   # noqa: F401
            umap_png  = output_dir / "umap_plot.png"
            tsne_png  = output_dir / "tsne_plot.png"
            comb_png  = output_dir / "combined_plot.png"

            umap_fig.write_image(str(umap_png), scale=2)
            tsne_fig.write_image(str(tsne_png), scale=2)
            combined_fig.write_image(str(comb_png), scale=2)

            paths["umap_png"]  = umap_png
            paths["tsne_png"]  = tsne_png
            paths["combined_png"] = comb_png

            print(f"  ✓ PNGs saved    → {output_dir}")
        except Exception as e:
            print(f"  ⚠ PNG export skipped ({e}). Install kaleido for PNG support.")

    # Save projection data as CSV for further analysis
    csv_path = output_dir / "embeddings_projected.csv"
    combined_df = pd.concat([
        umap_df.assign(method="UMAP"),
        tsne_df.assign(method="tSNE"),
    ])
    combined_df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    print(f"  ✓ Projections CSV → {csv_path}")

    return paths


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from data_parser import parse_interactions, build_chunks
    from metadata_tagger import tag_chunks
    from embedder import EmbeddingPipeline

    base = Path(__file__).parent
    input_path = base / "Input Decs" / "input.txt"

    interactions = parse_interactions(input_path)
    chunks = build_chunks(interactions, window_size=2)
    chunks = tag_chunks(chunks, interactions)

    pipeline = EmbeddingPipeline()
    pipeline.fit(chunks)

    run_visualization(
        embeddings=pipeline.embeddings,
        chunks=chunks,
        output_dir=base,
        save_png=True,
    )
    print("\nDone! Open umap_plot.html or combined_plot.html in your browser.")
