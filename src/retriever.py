"""Bi-encoder retrieval pipeline: Pinecone fetch → BM25 + dense → RRF fusion.

fetch_all_events()   — fetches all events from Pinecone with optional screen filter
biencoder_search()   — hybrid BM25 + dense retrieval with RRF fusion, returns all scores
"""
from __future__ import annotations

import logging
from google import genai

from src.config import RRF_K, RRF_ALPHA
from src.indexer import embed_query, deserialize_event_from_metadata, build_bm25_index, tokenize
from src.models import Event

logger = logging.getLogger(__name__)


def fetch_all_events(
    pinecone_index: object,
    client: genai.Client,
    query: str,
    screens: list[str] | None = None,
) -> tuple[list[Event], list[float]]:
    """Fetch all matching events from Pinecone using the query embedding.

    Args:
        pinecone_index: Pinecone index handle.
        client:         Gemini client for embedding the query.
        query:          Expanded query string.
        screens:        Optional list of screen names to filter by.

    Returns:
        Tuple of (events, cosine_scores) where each event corresponds
        to the cosine score at the same index.
    """
    logger.info(f"[FETCH] Query: '{query[:100]}'")

    query_vec = embed_query(client, query)
    logger.info(f"[FETCH] Query embedding dimension: {len(query_vec)}")

    # Build Pinecone metadata filter
    pc_filter = None
    if screens:
        pc_filter = {"screen_name": {"$in": screens}}
        logger.info(f"[FETCH] Screen filter: {screens}")

    response = pinecone_index.query(
        vector=query_vec,
        top_k=10000,
        include_metadata=True,
        filter=pc_filter,
    )

    logger.info(f"[FETCH] Pinecone returned {len(response.matches)} matches")

    events: list[Event] = []
    cosine_scores: list[float] = []

    for match in response.matches:
        try:
            event = deserialize_event_from_metadata(match.metadata)
            events.append(event)
            cosine_scores.append(float(match.score))
        except (ValueError, KeyError) as e:
            logger.warning(f"[FETCH] Failed to deserialize {match.id}: {e}")

    logger.info(f"[FETCH] Deserialized {len(events)} events")
    return events, cosine_scores


def biencoder_search(
    events: list[Event],
    cosine_scores: list[float],
    query: str,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Run hybrid BM25 + dense retrieval with weighted RRF fusion.

    Args:
        events:        List of Event objects (from Pinecone fetch).
        cosine_scores: Cosine similarity scores from Pinecone (same order as events).
        query:         Query string for BM25 scoring.
        rrf_k:         RRF smoothing constant.

    Returns:
        List of dicts with keys: event, rrf_score, bm25_score, dense_score.
        Sorted by rrf_score descending. All events included (no threshold).

    RRF weighting: rrf_score = alpha * 1/(k + dense_rank) + (1 - alpha) * 1/(k + bm25_rank)
    where alpha = RRF_ALPHA from config (0.7 = 70% dense, 30% BM25).
    """
    if not events:
        return []

    logger.info(f"[BI-ENCODER] Running hybrid search on {len(events)} events")

    # --- BM25 leg ---
    bm25 = build_bm25_index(events)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    logger.info(f"[BI-ENCODER] BM25 scores: min={min(bm25_scores):.4f}, max={max(bm25_scores):.4f}")

    # --- Rank both legs ---
    dense_ranked_indices = sorted(range(len(cosine_scores)), key=lambda i: cosine_scores[i], reverse=True)
    dense_rank = {idx: rank + 1 for rank, idx in enumerate(dense_ranked_indices)}

    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked_indices)}

    # --- RRF fusion with weighting ---
    results = []
    for i, event in enumerate(events):
        # Weighted RRF: RRF_ALPHA favors dense/semantic, (1-RRF_ALPHA) favors BM25/keyword
        rrf_score = (RRF_ALPHA * (1.0 / (rrf_k + dense_rank[i]))) + \
                    ((1.0 - RRF_ALPHA) * (1.0 / (rrf_k + bm25_rank[i])))
        results.append({
            "event": event,
            "rrf_score": rrf_score,
            "bm25_score": float(bm25_scores[i]),
            "dense_score": cosine_scores[i],
        })

    # Min-max normalize RRF scores to [0, 1]
    raw_rrf = [r["rrf_score"] for r in results]
    min_rrf = min(raw_rrf)
    max_rrf = max(raw_rrf)
    rrf_range = max_rrf - min_rrf

    if rrf_range == 0.0:
        for r in results:
            r["rrf_score"] = 0.5
    else:
        for r in results:
            r["rrf_score"] = (r["rrf_score"] - min_rrf) / rrf_range

    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.info(f"[BI-ENCODER] RRF fusion complete (alpha={RRF_ALPHA}, min-max normalized). Top 3:")
    for r in results[:3]:
        logger.info(f"  {r['event'].event_name}: rrf={r['rrf_score']:.6f} bm25={r['bm25_score']:.4f} dense={r['dense_score']:.4f}")

    return results
