"""Cohere Rerank API for event retrieval scoring.

Scores every candidate event against the query using Cohere's rerank-v4.0-pro,
and returns ALL results sorted by relevance score (already normalized to [0,1] by API).
"""
from __future__ import annotations

import logging
import cohere

from src.config import COHERE_API_KEY, COHERE_RERANK_MODEL
from src.models import Event

logger = logging.getLogger(__name__)

_client: cohere.ClientV2 | None = None


def _get_client() -> cohere.ClientV2:
    """Lazily create and cache the Cohere V2 client."""
    global _client
    if _client is None:
        _client = cohere.ClientV2(api_key=COHERE_API_KEY)
    return _client


def crossencoder_score(events: list[Event], query: str) -> list[dict]:
    """Score all events against the query with Cohere Rerank API.

    Args:
        events: List of Event objects to score.
        query:  Search query string.

    Returns:
        List of dicts with keys: event, score.
        Scores are in [0, 1] as returned by the API.
        Sorted by score descending. All events included.
    """
    if not events:
        return []

    logger.info(f"[RERANKER] Scoring {len(events)} events against query: '{query[:80]}'")

    client = _get_client()
    documents = [e.to_document() for e in events]

    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=documents,
    )

    logger.info(f"[RERANKER] Cohere returned {len(response.results)} results")

    # Build results list — API returns results sorted by relevance already,
    # but we map back to Event objects using the index field.
    results = []
    for r in response.results:
        results.append({
            "event": events[r.index],
            "score": float(r.relevance_score),
        })

    # Already sorted by API but ensure descending order
    results.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"[RERANKER] Top 3 results:")
    for r in results[:3]:
        logger.info(f"  {r['event'].event_name}: {r['score']:.4f}")

    return results
