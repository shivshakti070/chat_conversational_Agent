# XYZ Chatbot — Conversational RAG Pipeline: Walkthrough

A production-quality, end-to-end conversational RAG system over 12 XYZ chatbot interaction transcripts.  
The pipeline fixes identified chatbot failure patterns and produces embeddings with visible  
**inter-document separation** and **intra-document cohesion**, visualized with UMAP and t-SNE.

---

## Tech Stack — Alternatives Considered & Why We Chose Each

---

### 1. Parser

#### What We Built
A **custom regex-based parser** (`data_parser.py`) that converts raw freeform transcript text into structured `Interaction` and `Turn` objects.

#### Alternatives Considered

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Custom Regex Parser** ✅ | Lightweight, zero dependencies, handles both formatting styles in our file | Brittle if format changes drastically | **Chosen** |
| **SpaCy / NLTK** | Rich NLP features, sentence segmentation | Overkill for structured transcript parsing; adds heavy deps | Rejected |
| **LLM-based extraction** (GPT-4) | Handles any format, highly flexible | Expensive, slow, non-deterministic, needs API | Rejected |
| **JSON / XML structured input** | Easy to parse | Our raw data is unstructured plain text | Not applicable |
| **Pandas read_csv** | Fast for tabular data | Data is multi-line freeform conversation, not tabular | Rejected |

#### Why Custom Regex
The input file has exactly **two formatting patterns** (inline metadata on header line, or split across lines). A hand-crafted regex parser handles both precisely, runs in milliseconds, and adds zero library dependencies. SpaCy or NLTK would add 200+ MB of models for zero additional benefit on this task.

**Key design choices:**
- `INTERACTION_HEADER_RE` — detects both `Interaction N` and `interaction - N, channel - ...` formats
- `META_LINE_RE` — extracts channel, start time, end time from the same line or the next line
- `TURN_RE` — matches `customer:` and `chatbot:` speaker prefixes
- System tokens (`/start`, `/close`, `prompt_survey`, `close`) are filtered out automatically

---

### 2. Tokeniser

#### What We Used
**WordPiece tokeniser** bundled with `all-MiniLM-L6-v2` (from `bert-base-uncased`).

#### Alternatives Considered

| Tokeniser | Vocabulary | Max Tokens | Used By |
|---|---|---|---|
| **WordPiece** ✅ | 30,522 subwords | 256 | BERT, MiniLM, BGE |
| **BPE (Byte-Pair Encoding)** | 50,000 subwords | 512 | RoBERTa, GPT family |
| **SentencePiece (Unigram)** | Configurable | 512 | T5, Flan-T5, XLM-R |
| **Character-level** | ~128 chars | N/A | Very old models |

#### Why WordPiece
- **All chunks are well within 256 tokens** — the longest interaction in our dataset is ~120 tokens, so max-length truncation never occurs.
- WordPiece handles lowercase banking vocabulary (`debit`, `contactless`, `overdraft`) cleanly with no out-of-vocabulary risk.
- BPE would have been fine too but offers no advantage here given the short context lengths.

> **Practical note**: The tokeniser is never called explicitly — `SentenceTransformer.encode()` handles tokenisation internally. We verified no truncation by checking chunk character lengths against the 256-token budget.

---

### 3. Chunking Strategy

#### What We Built
A **dual-level chunking** approach (`data_parser.py → build_chunks()`):

| Level | Name | Size | Count | Purpose |
|---|---|---|---|---|
| L1 | **Interaction-level** | 1 full conversation | 12 | Inter-document retrieval, intent clustering |
| L2 | **Turn-window** | 2 consecutive QA pairs | 36 | Fine-grained, intra-doc retrieval |
| — | **Total** | — | **48** | All indexed in FAISS |

#### Alternatives Considered

| Strategy | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| **Dual-level (ours)** ✅ | Interaction-level + sliding QA-pair windows | Best of both granularities; enables both coarse and fine retrieval | Slightly more complex setup | **Chosen** |
| **Fixed-token chunking** | Split every N tokens (e.g. 256) | Simple | Splits mid-thought; ignores dialogue structure | Rejected |
| **Sentence-level chunking** | One chunk per sentence | Very fine-grained | Too atomic; loses conversational context | Rejected |
| **Single-turn chunking** | One chunk per speaker turn | Good for long docs | Loses Q→A pairing which is semantically critical | Rejected |
| **No chunking (full doc)** | Entire corpus as one chunk | Simple | Loses specificity; retrieval returns everything | Rejected |
| **Recursive character splitting** | LangChain style, split by `\n\n` then `\n` | Easy to implement | Ignores semantic turn structure | Rejected |

#### Why Dual-Level
Conversations have two natural semantic scales:
- **Coarse** — "what did this whole interaction cover?" → answered by the interaction-level chunk
- **Fine** — "what exactly was said about card replacement steps?" → answered by a turn-window chunk

The sliding window of **2 QA pairs** was chosen because:
1. It preserves the customer's clarification in context (e.g. `"lost" → CHATBOT locks card`)
2. It's short enough to embed accurately without token budget issues
3. Interaction 12 (the failure loop) generates 9 windows, which intentionally produces a denser cluster in the embedding space — making the failure pattern visually identifiable in UMAP

**Intra-doc cohesion score = 0.7857** (avg cosine sim between parent interaction chunk and child turn-window chunks), confirming that the two levels stay semantically tightly coupled.

---

### 4. Embedding Strategy & Model

#### Benchmark — 3 Models Evaluated on Silhouette Score

The winning model was chosen **automatically** by computing the **Silhouette Score** on the 10 intent clusters of interaction-level chunks.

| Model | Silhouette Score ↑ | Embedding Dim | Load Time | Encode Time | Winner? |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | **0.1295** | 384 | 17.87s | 1.32s | ✅ |
| `BAAI/bge-small-en-v1.5` | 0.0940 | 384 | 12.83s | 0.48s | |
| `intfloat/e5-base-v2` | 0.0912 | 768 | 22.73s | 0.97s | |

#### Alternatives Considered (Wider Landscape)

| Model / Approach | Type | Strengths | Weaknesses |
|---|---|---|---|
| **all-MiniLM-L6-v2** ✅ | Symmetric, 384-dim | Fast, small, good clustering | Not asymmetric; similar query/doc space |
| **BAAI/bge-small-en-v1.5** | Asymmetric retrieval | Strong for Q→Doc retrieval | Scored lower on our intent clusters |
| **intfloat/e5-base-v2** | Instruction-tuned | Designed for Q&A | Larger (768-dim), still scored lower |
| **OpenAI text-embedding-3-small** | API, 1536-dim | State-of-the-art | Requires API key, costs money, not offline |
| **OpenAI text-embedding-3-large** | API, 3072-dim | Best quality | Expensive; overkill for 48 chunks |
| **GTE-base / GTE-large** | Symmetric | Very competitive | Not benchmarked in this run |
| **TF-IDF / BM25** | Sparse | Fast, interpretable, no GPU | No semantic understanding; fails on paraphrases |
| **Word2Vec / GloVe** | Static word embeddings | Simple | No context sensitivity; outdated |

#### Why `all-MiniLM-L6-v2` Won
1. **Highest Silhouette Score (0.1295)** — best empirical inter-document cluster separation on our 10 banking intents
2. **384 dimensions** — compact enough for fast FAISS retrieval on a small corpus
3. **Self-contained** — no API key, fully offline, reproducible
4. **Symmetric encoding** — since both query and documents are short conversational text, symmetric encoding is appropriate (unlike long-doc RAG where asymmetric models shine)

#### Embedding Design Choices
- Vectors are **L2-normalised** → inner product = cosine similarity (no extra computation at retrieval)
- Saved to `embeddings.npy` for reproducibility and re-visualization without re-running the model
- e5 family uses a `"passage: "` prefix for encoding documents (applied automatically in `embedder.py`)

---

### 5. Vector Database

#### What We Built
**FAISS `IndexFlatIP`** (`vector_store.py`) — a FAISS flat inner-product index functioning as a cosine similarity store.

#### Alternatives Considered

| Vector DB | Type | Strengths | Weaknesses | Verdict |
|---|---|---|---|---|
| **FAISS IndexFlatIP** ✅ | Local / in-memory | Exact cosine search, zero infra, offline | No built-in persistence UI | **Chosen** |
| **ChromaDB** | Local / embedded | Easy Python API, metadata filtering built-in | Heavier dependency, slower for tiny corpora | Good alt |
| **Weaviate** | Server-based | Rich schema, GraphQL API, hybrid search | Requires Docker / server; massive overkill for 48 vectors | Rejected |
| **Pinecone** | Cloud / managed | Scales to billions of vectors, SLAs | Paid, needs internet, no offline use | Rejected |
| **Qdrant** | Local or cloud | Fast, rich filtering, production-ready | Adds server infra; unnecessary for 48 vectors | Rejected |
| **Milvus** | Server-based | Enterprise-scale | Very heavy infra for a dev pipeline | Rejected |
| **pgvector (Postgres)** | SQL extension | Combine SQL + vector in one DB | Requires Postgres; overkill here | Rejected |
| **LanceDB** | Local / embedded | Fast, columnar, Pandas-native | Less mature ecosystem | Interesting alt |
| **Simple numpy dot product** | In-memory | Zero deps | No indexing structure, O(N) brute force | Rejected |

#### Why FAISS IndexFlatIP
1. **Perfect for 48 vectors** — with a corpus this small, `IndexFlatIP` does an exact exhaustive search in microseconds. Approximate methods (HNSW, IVF) only help at 100k+ vectors.
2. **No infrastructure** — runs in-process, no Docker, no server, no API keys.
3. **Cosine similarity via inner product** — L2-normalised embeddings turn inner product into exact cosine similarity. This matches the metric used during model training.
4. **Full persistence** — index saved to `vector_store/faiss.index`, metadata to `metadata.json`, chunks to `chunks.pkl`. Reload in one line.
5. **Metadata filtering** — a custom post-retrieval filter layer (by `intent`, `channel`, `outcome`, `chunk_type`) is implemented in Python alongside FAISS, giving us LangChain-style filtering without the overhead.

**Retrieval stats from validation run:**
```
Inter-doc Separation  (Silhouette Score) : 0.1295
Intra-doc Cohesion    (Avg cosine sim)   : 0.7857
Total vectors indexed                    : 48
Embedding dim                            : 384
```

---

### 6. Conversational RAG Chain Design

#### Architecture
```
Customer Query
      │
      ▼
 ┌─────────────────────────────────────┐
 │  Step 1: Condensation               │  ← Compress history + query
 │  (chat history → standalone query)  │     into one standalone query
 └──────────────┬──────────────────────┘
                │
                ▼
 ┌─────────────────────────────────────┐
 │  Step 2: Retrieval                  │  ← FAISS top-k search (k=3)
 │  (encode query → FAISS → re-rank)   │     + dedup by interaction_id
 └──────────────┬──────────────────────┘
                │
                ▼
 ┌─────────────────────────────────────┐
 │  Step 3: Generation                 │  ← System prompt + context
 │  (LLM + retrieved context)          │     + history + query → answer
 └─────────────────────────────────────┘
```

#### Why Condensation Step
The chatbot's biggest failure was **context loss across turns** — e.g. the customer said `"lost"` in one turn, then `"traveling abroad"` in the next, and the chatbot failed to connect these. The condensation step feeds full chat history + new query to the LLM and produces a **standalone query** before retrieval — e.g.:

> `"I lost my card and I am travelling to Spain next week"` → retrieves both card-replacement and travel-notification chunks in a single search.

#### LLM Backend (Priority Order)
1. **OpenAI GPT-3.5-turbo** — if `OPENAI_API_KEY` is set
2. **google/flan-t5-base** — local HuggingFace model, no API key needed
3. **Echo mode** — returns retrieved context as-is (absolute fallback)

#### Improved Prompt Design — Chatbot Failures Fixed

| Chatbot Failure | Fix Applied in Prompt |
|---|---|
| Address-change loop | Explicitly told to accept free-text addresses, never ask to rephrase more than once |
| Context loss across turns | Condensation step produces a standalone query before retrieval |
| Premature handoff | Must provide product summary before offering agent handoff |
| Password-reset misrouting | Distinguish "forgotten password" (reset link) vs "not registered" (registration) |

---

### 7. Dimensionality Reduction & Visualization

| Method | Library | Hyperparameters | Purpose |
|---|---|---|---|
| **UMAP** | `umap-learn` | `n_neighbors=10`, `min_dist=0.15`, `metric=cosine` | Preserves both local and global structure; faster than t-SNE |
| **t-SNE** | `scikit-learn` | `perplexity=auto (≤ N/3)`, `n_iter=1200`, `metric=cosine` | Reveals tight local cluster structure |

Both projections confirm:
- **Inter-document separation** — distinct intent clusters (`card_replacement`, `password_reset`, `fraud_alert`) occupy non-overlapping manifold regions
- **Intra-document cohesion** — turn-window chunks (diamonds ◆) cluster tightly around their parent interaction chunk (circle ●)
- Convex-hull ellipses drawn per interaction make intra-doc cohesion visually explicit

![Combined UMAP vs t-SNE embedding space](/Users/shivshakti/.gemini/antigravity/brain/4de6afc5-af9a-4ccf-82d5-4406aa07eb48/combined_viz_png_1773907221569.png)

![UMAP projection coloured by intent](/Users/shivshakti/.gemini/antigravity/brain/4de6afc5-af9a-4ccf-82d5-4406aa07eb48/umap_viz_png_1773907244286.png)

---

## Quantitative Results

| Metric | Value | Interpretation |
|---|---|---|
| **Inter-doc Separation** (Silhouette Score) | **0.1295** | Positive separation across 10 intent clusters; expected range for small noisy conversational corpora |
| **Intra-doc Cohesion** (Avg cosine sim) | **0.7857** | Strong — turn-window chunks sit very close to parent interaction in embedding space |
| Best interaction cohesion | 0.8488 (mortgage, interaction 10) | Simple, focused conversation → very tight cluster |
| Worst interaction cohesion | 0.6342 (balance enquiry, interaction 2) | Short interaction with context switch → slightly looser |

---

## Output Files

| File | Description |
|---|---|
| `embeddings.npy` | Raw L2-normalised embeddings (48 × 384) |
| `vector_store/faiss.index` | FAISS IndexFlatIP binary |
| `vector_store/metadata.json` | Chunk metadata sidecar |
| `vector_store/chunks.pkl` | Serialised Chunk objects |
| `umap_plot.html` | Interactive UMAP (open in browser, hover for chunk text) |
| `tsne_plot.html` | Interactive t-SNE |
| `combined_plot.html` | UMAP + t-SNE side-by-side comparison |
| `umap_plot.png` | Static UMAP PNG |
| `tsne_plot.png` | Static t-SNE PNG |
| `combined_plot.png` | Static combined PNG |
| `embeddings_projected.csv` | 2D coordinates + metadata for all chunks (both methods) |

---

## How to Run

```bash
cd "/Users/shivshakti/Desktop/learning_projects/conversational_Agent"

# Full pipeline (parse + embed + RAG demo + visualize)
python3 main.py

# Visualization only — no LLM/API key needed
python3 main.py --no-rag

# Individual module tests
python3 data_parser.py        # test parsing + chunking
python3 metadata_tagger.py    # test intent/outcome tagging
python3 embedder.py           # run model benchmark
python3 vector_store.py       # test FAISS retrieval
python3 visualize.py          # generate UMAP + t-SNE plots
```

## Setting Up the LLM (Optional)

```bash
export OPENAI_API_KEY="sk-..."
python3 main.py    # uses GPT-3.5-turbo for generation
```

Without an API key the pipeline automatically falls back to `google/flan-t5-base` (local, ~1 GB download on first run).
