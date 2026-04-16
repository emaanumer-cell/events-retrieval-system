# Event Retrieval System — Full Project Flow & Parameter Reference

## Overview

A RAG-based retrieval system that takes a natural language query (e.g. "acquisition events"), expands it with an LLM, retrieves candidate GA4 events via hybrid search, and reranks them with the Cohere Rerank API. The Streamlit UI runs both pipelines (bi-encoder vs reranker) side-by-side for comparison.

---

## 1. Data Ingestion & Parsing (`src/parser.py`)

**Input:** An `.xlsx` tracking plan file from `data/` directory.

**Column layout (0-indexed):**

| Col | Field | Example |
|-----|-------|---------|
| 1 | Event Name | `creating_screen_shown` |
| 2 | Event Definition | "Fired when the AI Creating Animation loading screen is displayed." |
| 3 | Screen Name | `AI Creating Animation` |
| 4 | Parameter Name | `image_id` |
| 5 | Parameter Description | "ID of the image being animated" |
| 6 | Sample Values | `img_abc123` |
| 7 | Key Event | `Yes` / `No` |
| 8 | Detailed Event Definition | Long-form description with upstream/downstream context |

**Parsing logic:**
- One row per parameter; a new event starts when column 1 (Event Name) is non-empty.
- Continuation rows (empty Event Name) append parameters to the current event.
- Uses `openpyxl` in `read_only=True, data_only=True` mode.
- Output: `list[Event]` — each event has a deterministic `doc_id` = `md5(event_name + "_" + screen_name)[:16]`.

**Decisions / things to tune:**
- The `doc_id` hash is MD5 truncated to 16 hex chars. Collision-safe for small corpora but not cryptographically unique.
- No deduplication logic — if two rows have the same event name + screen name, they get the same `doc_id` and the second upsert overwrites the first in Pinecone.

---

## 2. Text Representations (`src/models.py`)

Each `Event` produces **two different text representations** used at different stages:

### 2a. `to_index_text()` — used for embedding + BM25

```
{event_name} | {screen_name} | {event_definition} | {detailed_event_definition} | {param1_name}: {desc} (e.g. {samples}); {param2_name}: ...
```

- Pipe-delimited, compact.
- Used by: Gemini embedding (indexing), BM25 tokenization.
- **Includes parameters** (name, description, sample values) — appended if present.
- `detailed_event_definition` only appended if non-empty.

### 2b. `to_document()` — used for cross-encoder reranking

```
Event: {event_name}. Definition: {event_definition} Screen: {screen_name}. 
Parameters: {p.name}: {p.description} (e.g. {p.sample_values}); ...
Detailed definition: {detailed_event_definition}. Key event: {key_event}.
```

- Full prose, richer than `to_index_text()`.
- **Includes parameters and key_event flag.**
- Used by: Cohere Rerank API as the `documents` input.

**Symmetry note:**
- Both `to_index_text()` and `to_document()` now include parameters.
- The bi-encoder and reranker pipelines see comparable information.
- `to_document()` is slightly richer (includes `key_event` flag), but the core content is aligned.

---

## 3. Indexing Pipeline (`src/indexer.py`, `build_indexes.py`)

### 3a. Dense Index — Gemini Embedding + Pinecone

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Embedding model** | `gemini-embedding-001` | Google's production embedding model. Config key: `GEMINI_EMBED_MODEL` |
| **Embedding dimension** | `3072` | Full dimension. Config key: `GEMINI_EMBED_DIMENSION` |
| **Input token limit** | `2048` | Gemini's max input tokens. Config key: `GEMINI_EMBED_TOKEN_LIMIT`. Enforced via intelligent truncation. |
| **Truncation strategy** | Parameters first, then detailed_definition | Priority: event_name + screen + definition always kept. Logs warnings when truncation occurs. |
| **Vector DB** | Pinecone Serverless | Cloud: `aws`, Region: `us-east-1` |
| **Index name** | `events-index-gemini` | Config key: `PINECONE_INDEX_NAME` |
| **Distance metric** | `cosine` | Set at index creation time |
| **Upsert batch size** | `100` | Pinecone recommended max per call |
| **API** | `genai.Client.models.embed_content()` | From `google-genai` SDK. Same client used for indexing and query embedding |

**Metadata stored per vector:**
```json
{
  "event_name": "...",
  "screen_name": "...",
  "event_definition": "...",
  "detailed_definition": "...",
  "key_event": "Yes/No",
  "parameters_json": "[{name, description, sample_values}, ...]",
  "corpus_hash": "sha256[:16]",
  "embedding_model": "gemini-embedding-001",
  "embedding_dim": 3072,
  "indexed_at": 1234567890
}
```

**Why this structure:**
- No redundancy — each field stored once (no separate JSON blob containing duplicates)
- Minimal metadata bloat — parameters as compact JSON, not full Event serialization
- Observability — tracks embedding model version and indexing timestamp
- Efficient reconstruction — `deserialize_event_from_metadata()` reads fields directly

**Idempotency guard:** 
- Computes `corpus_hash = sha256(to_index_text() of all events)[:16]`
- Checks if stored hash (from first vector's metadata) matches current hash
- Only skips re-embedding if vector count AND hash both match
- Handles content changes: if one event changes, hash changes, forces re-index

**Token limit enforcement:**
- Texts exceeding 2048 tokens are intelligently truncated before embedding
- Truncation priority: 
  1. Drop parameters first (if needed)
  2. Then drop detailed_definition (if still over)
  3. Always keep: event_name + screen_name + event_definition
- Logs warnings whenever truncation occurs (includes token count before/after)

**Things to tune:**
- `cosine` vs `dotproduct` — test whether Gemini embeddings are pre-normalized.
- No asymmetric input types (unlike Cohere's `search_document`/`search_query`) — same `embed_content()` call for both indexing and querying.
- **Metadata field selection**: Currently storing all event fields in metadata. If corpus grows very large, consider storing only `screen_name` and `parameters_json` (the minimal set needed for `deserialize_event_from_metadata()` to reconstruct Event).
- **Token truncation tuning**: Current threshold is 2048 (Gemini's limit). Could use lower threshold (e.g. 1800) to add safety margin if needed.

### 3b. Sparse Index — BM25 (In-Memory)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Algorithm** | `BM25Okapi` | From `rank_bm25` library |
| **Tokenization** | Porter Stemming + whitespace split | Stemmer from `nltk`. Example: "animations" + "animation" both → "anim" |
| **k1 (term frequency saturation)** | `1.5` (library default) | Not explicitly set — using `rank_bm25` defaults |
| **b (length normalization)** | `0.75` (library default) | Not explicitly set — using `rank_bm25` defaults |
| **Corpus text** | `tokenize(event.to_index_text())` | Same tokenization as query time |
| **Persistence** | None — rebuilt on every app start and every search | See note below |

**Critical note on BM25 rebuild:**
- BM25 is rebuilt **twice**: once at startup in `build_indexes()`, and again **per query** inside `biencoder_search()` from the Pinecone-fetched event subset. The per-query rebuild means BM25 IDF statistics are computed only over the fetched events, not the full corpus. This is intentional — it scores only the events already retrieved by Pinecone.

**Things to tune:**
- `k1` and `b` are at library defaults. Higher `k1` = more weight on term frequency. Lower `b` = less length normalization.
- Tokenization uses Porter Stemmer — handles inflections (animations → animation). Could add stopword removal for better signal.
- No IDF smoothing or custom preprocessing.

---

## 4. Query Expansion (`src/query.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | `gemini-3-pro-preview` | Config key: `GEMINI_MODEL`, env-overridable |
| **Temperature** | `0.5` | Moderate creativity. Config key: `GEMINI_TEMPERATURE` |
| **Max output tokens** | `256` | Config key: `GEMINI_MAX_TOKENS` |
| **Prompt strategy** | Keyword expansion | Returns comma-separated keywords only |

**Prompt (verbatim):**
```
User query: {query}

You are helping search a GA4 mobile app tracking plan. 
List common app event names, screen names, and analytics keywords 
that are related to this query. Include synonyms, abbreviations, 
and variations commonly used in mobile app analytics. 
Return ONLY a comma-separated list of keywords, nothing else.
```

**Output format:** `"{original_query} {comma-separated keywords from Gemini}"`

**Fallback:** If `GEMINI_API_KEY` is unset or the call fails, returns the original query unchanged.

**Things to tune:**
- Temperature: 0.5 is a middle ground. Lower (0.1-0.3) for more deterministic expansions, higher (0.7-1.0) for broader coverage.
- Max tokens: 256 limits expansion breadth. For short queries this is plenty; for complex multi-concept queries it might truncate.
- The expanded query is a raw string concatenation — no deduplication of terms, no weighting of original vs expanded terms.
- The expanded query is used for **both** the dense embedding and BM25. The Gemini keywords may dilute the original intent in the embedding space.
- Cross-encoder uses the **original query** (not expanded) — see app.py line 148. This is a deliberate asymmetry: expansion helps recall in bi-encoder, but cross-encoder sees the raw user intent.

---

## 5. Retrieval Pipeline (`src/retriever.py`)

### 5a. Fetch from Pinecone (`fetch_all_events`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **top_k** | `10000` | Fetches essentially all events. Hardcoded, not configurable |
| **Query embedding** | `genai.Client.models.embed_content()` | Same model + dimension as indexing (gemini-embedding-001, 1536d) |
| **Screen filter** | `{"screen_name": {"$in": [...]}}` | Optional metadata filter |

**Decision:** Fetches ALL events (up to 10K) rather than a selective top-K. This means the bi-encoder pipeline scores everything, which is expensive but ensures full recall. The dense cosine scores from Pinecone are preserved and used as one leg of RRF.

### 5b. Hybrid Search + RRF Fusion (`biencoder_search`)

**Architecture:** Two-leg hybrid retrieval with Reciprocal Rank Fusion.

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Leg 1: Dense retrieval** | Cosine scores from Pinecone | Already computed during fetch |
| **Leg 2: BM25** | Rebuilt from fetched events | Scored with `.get_scores(tokenized_query)` |
| **BM25 query tokenization** | `query.lower().split()` | Must match index tokenization |
| **RRF smoothing constant (k)** | `60` | Standard default. Config key: `RRF_K` |
| **RRF formula** | `alpha * 1/(k + rank_dense) + (1-alpha) * 1/(k + rank_bm25)` | Weighted by RRF_ALPHA (0.7 = 70% dense) |
| **RRF alpha** | `0.7` | Config key: `RRF_ALPHA`. Favors dense/semantic over BM25/keyword |
| **Score normalization** | Min-max to [0, 1] | Applied after RRF fusion |
| **RRF threshold** | `0.0` (pass everything) | Config key: `RRF_THRESHOLD`, env-overridable |

**RRF Details:**
- Each event gets a rank in both the BM25 and dense lists (1 = best).
- The RRF score is the sum of reciprocal ranks from both legs.
- With `k=60` and 2 legs, the max possible raw RRF score ≈ `2 * 1/61 ≈ 0.0328`.
- After min-max normalization, the top event always gets score 1.0 and the bottom gets 0.0.

**Things to tune:**
- `RRF_K`: Higher values smooth out rank differences (less volatile). Lower values amplify rank differences. 60 is the standard default from the original RRF paper.
- `RRF_ALPHA`: Currently 0.7 (70% dense, 30% BM25). Lower alpha (e.g. 0.5, 0.3) favors keyword matching. Higher alpha (e.g. 0.8, 0.9) favors semantic matching. Tune based on query type and user satisfaction.
- `RRF_THRESHOLD`: Currently 0.0 (no filtering). Raising it would pre-filter before reranking, reducing cross-encoder compute.
- BM25 is rebuilt per query on the Pinecone subset. IDF statistics differ from a global corpus BM25.
- No query-time BM25 parameter overrides (k1, b are locked to library defaults).

---

## 6. Reranking — Cohere Rerank API (`src/reranker.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | `rerank-v4.0-pro` | Cohere's production reranker. Config key: `COHERE_RERANK_MODEL` |
| **API** | `POST /v2/rerank` | Uses Cohere Python SDK `ClientV2.rerank()` |
| **Input format** | `query` + `documents` list | Query = original (not expanded), Documents = `event.to_document()` per event |
| **Scoring** | `relevance_score` from API | Already normalized to [0, 1] by the API |
| **top_n** | Not set (returns all) | Could be set to limit API response size |
| **max_tokens_per_doc** | `4096` (API default) | Long documents auto-truncated by API |
| **Threshold** | `0.3` (config) but **not applied in current code** | `RERANKER_THRESHOLD` exists in config but `crossencoder_score()` returns ALL results |
| **Top-K** | `100` (config) but **not applied in current code** | `RERANKER_TOP_K` exists in config but is unused |
| **Client caching** | `ClientV2` created once, cached at module level | No Streamlit cache needed (no model download) |

**How it works:**
1. Converts each event to a document string via `event.to_document()`.
2. Calls `client.rerank(model, query, documents)` — single API call.
3. API returns results with `index` (position in original list) and `relevance_score` in [0, 1].
4. Maps results back to Event objects using the index.
5. Returns all events sorted by score descending.

**Alignment with bi-encoder:**
- Both reranker and bi-encoder see the **same query**: expanded (if enabled) or raw (if disabled).
- Reranker sees `to_document()` (includes parameters, detailed definition, key_event).
- Bi-encoder sees `to_index_text()` (same fields, tokenized with stemming for BM25).
- Query expansion is controlled by checkbox in UI — affects both pipelines equally.

**Things to tune:**
- **top_n**: Pass to API to limit results returned (saves response size, not compute — API still scores all docs).
- **max_tokens_per_doc**: Default 4096. Reduce if you want faster inference, increase if documents are very long.
- **Threshold**: The `RERANKER_THRESHOLD=0.3` in config is not enforced. Apply it to filter out low-confidence results.
- **Top-K**: `RERANKER_TOP_K=100` in config is not enforced. Could limit final output.
- **Document format**: `to_document()` is prose. Cohere recommends YAML for structured data — could improve reranking quality.
- **Batching**: API recommends ≤1000 documents per request. If corpus grows, need to batch.

---

## 7. Frontend (`app.py` — Streamlit)

**Layout:**
- Text input for query
- Multi-select for screen name filter (populated from Pinecone via zero-vector query)
- Example queries listed in an expander
- Results displayed as two side-by-side DataFrames

**Pipeline flow on search:**
1. **Query expansion (optional)**: If `enable_expansion` checkbox is checked, `expand_query(query)` adds keywords via Gemini. Otherwise, use raw query.
2. `fetch_all_events(pinecone_index, gemini_client, expanded, screens)` → all events + cosine scores
3. `biencoder_search(events, cosine_scores, expanded)` → BM25+Dense+RRF results
4. `crossencoder_score(events, expanded)` → Cohere Rerank results (same query as bi-encoder)

**Screen name discovery:**
- Queries Pinecone with a zero vector `[0.0] * 1536`, `top_k=10000`
- Extracts unique `screen_name` values from metadata
- Cached for 600 seconds (`ttl=600`)

**Logging:**
- Writes to `logs/latest.log` (overwritten each run, mode `"w"`)
- Also outputs to console
- Level: `DEBUG` for both handlers

---

## 8. End-to-End Data Flow

```
User Query ("acquisition events")
        │
        ▼
┌─────────────────────┐
│  Query Expansion    │  Gemini 3 Pro Preview
│  temp=0.5           │  max_tokens=256
│  → expanded query   │  "acquisition events install, first_open, ..."
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Pinecone Fetch     │  embed expanded query with gemini-embedding-001 (1536d)
│  top_k=10000        │  cosine similarity
│  optional screen    │  returns ALL events + cosine scores
│  filter             │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ BM25   │ │ Dense scores │
│ Okapi  │ │ (from fetch) │
│ rebuilt│ │              │
│ on     │ │              │
│ subset │ │              │
└───┬────┘ └──────┬───────┘
    │              │
    ▼              ▼
┌─────────────────────┐
│  RRF Fusion         │  k=60, equal weight
│  min-max norm [0,1] │  
│  no threshold       │
└────────┬────────────┘
         │
         ▼
  BI-ENCODER RESULTS TABLE
         
         │ (parallel, independent)
         ▼
┌─────────────────────┐
│  Cohere Rerank      │  rerank-v4.0-pro
│  ORIGINAL query     │  input = (query, to_document())
│  scores in [0,1]    │  richer text than bi-encoder sees
│  no threshold       │
└────────┬────────────┘
         │
         ▼
  COHERE RERANK RESULTS TABLE
```

---

## 9. All Configurable Parameters (Quick Reference)

| Parameter | Current Value | Config Key | Env Override | Where Used |
|-----------|--------------|------------|--------------|------------|
| Gemini embed model | `gemini-embedding-001` | `GEMINI_EMBED_MODEL` | No | indexer.py, retriever.py |
| Embedding dimension | `3072` | `GEMINI_EMBED_DIMENSION` | No | indexer.py, app.py |
| Cohere rerank model | `rerank-v4.0-pro` | `COHERE_RERANK_MODEL` | No | reranker.py |
| Gemini model | `gemini-3-pro-preview` | `GEMINI_MODEL` | `GEMINI_MODEL` | query.py |
| Gemini temperature | `0.5` | `GEMINI_TEMPERATURE` | No | query.py |
| Gemini max tokens | `256` | `GEMINI_MAX_TOKENS` | No | query.py |
| Pinecone index name | `events-index-gemini` | `PINECONE_INDEX_NAME` | `PINECONE_INDEX_NAME` | indexer.py, app.py |
| Pinecone cloud | `aws` | `PINECONE_CLOUD` | `PINECONE_CLOUD` | indexer.py |
| Pinecone region | `us-east-1` | `PINECONE_REGION` | `PINECONE_REGION` | indexer.py |
| Pinecone metric | `cosine` | Hardcoded | No | indexer.py |
| Pinecone fetch top_k | `10000` | Hardcoded | No | retriever.py |
| Upsert batch size | `100` | `_UPSERT_BATCH_SIZE` | No | indexer.py |
| BM25 k1 | `1.5` (default) | Library default | No | indexer.py |
| BM25 b | `0.75` (default) | Library default | No | indexer.py |
| BM25 tokenizer | `lower().split()` | Hardcoded | No | indexer.py, retriever.py |
| RRF k | `60` | `RRF_K` | No | retriever.py |
| RRF alpha | `0.7` | `RRF_ALPHA` | No | retriever.py |
| RRF threshold | `0.0` | `RRF_THRESHOLD` | `RRF_THRESHOLD` | config.py (unused in code) |
| Reranker threshold | `0.3` | `RERANKER_THRESHOLD` | `RERANKER_THRESHOLD` | config.py (unused in code) |
| Reranker top-K | `100` | `RERANKER_TOP_K` | No | config.py (unused in code) |
| Screen name cache TTL | `600s` | Hardcoded | No | app.py |
| Retrieval top-N | `100` | `RETRIEVAL_TOP_N` | No | config.py (unused in code) |

---

## 10. Improvement Checklist

### Indexing
- [x] **Tokenization**: Added Porter Stemmer for BM25 (handles inflections like "animations" → "anim"). Further improvements: stopword removal, lemmatization.
- [ ] **Gemini dimension sweep**: Test 768d (fast) vs 1536d (current) vs 3072d (max quality) — all supported natively via MRL.
- [x] **Include parameters in index text**: `to_index_text()` now includes param name, description, and sample values.
- [x] **Idempotency**: Content-hash-based check (sha256 of corpus) — detects changes even if vector count stays the same.
- [x] **Metadata structure**: Refactored to remove redundancy, store parameters separately, add corpus hash + observability fields.

### Query Expansion
- [ ] **Temperature sweep**: Test 0.2, 0.5, 0.8 and measure retrieval quality.
- [ ] **Structured expansion**: Instead of raw keyword dump, have Gemini return JSON with categories (event names, screens, params).
- [ ] **Term weighting**: Weight original query terms higher than expanded terms (e.g., boost factor).

### Retrieval
- [ ] **RRF k sweep**: Test k=10, 30, 60, 100 and measure ranking quality.
- [x] **Leg weighting**: Implemented weighted RRF with `RRF_ALPHA=0.7` (70% dense, 30% BM25). Can be tuned in config.
- [ ] **Apply RRF_THRESHOLD**: Currently unused — wire it up and test.
- [ ] **Apply RETRIEVAL_TOP_N**: Currently unused — wire it up to limit candidates before reranking.
- [ ] **Selective Pinecone top_k**: Instead of 10000, use a reasonable top_k (100-500) and rely on RRF to surface BM25-only hits.

### Reranking (Cohere Rerank API)
- [ ] **Apply RERANKER_THRESHOLD**: Currently unused — wire it up.
- [ ] **Apply RERANKER_TOP_K**: Currently unused — wire it up (or pass `top_n` to the API).
- [ ] **Query input**: Test using expanded query vs original query for reranker input.
- [ ] **Document format**: Try YAML formatting for `to_document()` — Cohere recommends it for structured data.
- [ ] **max_tokens_per_doc**: Test lower values (e.g. 1024, 2048) for speed vs quality tradeoff.

### Architecture
- [ ] **Combined scoring**: Weighted combination of RRF score + cross-encoder score as a final ranking signal.
- [ ] **Evaluation framework**: Build a ground-truth test set of (query → expected events) to measure MRR, NDCG, recall@k.
- [ ] **A/B display**: Show rank differences between bi-encoder and cross-encoder to quickly spot disagreements.
