"""Indexing pipeline: Gemini embeddings → Pinecone + in-memory BM25.

Usage:
    pinecone_index, bm25 = build_indexes(events)

The two indexes share the same corpus (to_index_text() of each event).
"""
from __future__ import annotations

import json
import hashlib
import time
import logging
from google import genai
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

from src.config import (
    GEMINI_API_KEY,
    GEMINI_EMBED_DIMENSION,
    GEMINI_EMBED_MODEL,
    GEMINI_EMBED_TOKEN_LIMIT,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)
from src.models import Event

# Pinecone recommends batches of ≤100 vectors per upsert call
_UPSERT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Tokenization with stemming
# ---------------------------------------------------------------------------

_stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Tokenize text with Porter stemming.

    Process:
    1. Lowercase
    2. Split on whitespace and punctuation
    3. Stem each token
    4. Filter empty strings

    Example: "animations" → "anim", "animation" → "anim" (matches)
    """
    # Simple tokenization: lowercase, split on whitespace, remove punctuation
    tokens = []
    for word in text.lower().split():
        # Remove trailing punctuation (., ;, :, etc)
        word = word.rstrip('.,;:!?()[]{}')
        if word:  # Skip empty tokens
            # Apply stemming
            stemmed = _stemmer.stem(word)
            tokens.append(stemmed)
    return tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def build_bm25_index(events: list[Event]) -> BM25Okapi:
    """Build an in-memory BM25 index from the event corpus.

    Each event's to_index_text() is tokenized with stemming.
    Mirrors exactly the tokenisation used during query time in retriever.py.
    """
    corpus = [tokenize(event.to_index_text()) for event in events]
    return BM25Okapi(corpus)


# ---------------------------------------------------------------------------
# Token limit handling for Gemini embeddings
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough estimate of token count for text.

    Gemini uses approximately 1 token per 4 characters (standard estimate).
    """
    return len(text) // 4


def _truncate_index_text_for_embedding(event: Event) -> str:
    """Truncate event text intelligently for Gemini embedding (2048 token limit).

    Truncation priority (preserve in this order):
    1. event_name + screen_name + event_definition (always keep)
    2. detailed_event_definition (drop first if needed)
    3. parameters (drop last if still over limit)

    Returns the text to embed, logging warnings if truncation occurs.
    """
    # Build full text
    full_text = event.to_index_text()
    token_count = _estimate_tokens(full_text)

    if token_count <= GEMINI_EMBED_TOKEN_LIMIT:
        return full_text

    # Step 1: Drop parameters, keep core definitions
    parts = [event.event_name, event.screen_name, event.event_definition]
    if event.detailed_event_definition:
        parts.append(event.detailed_event_definition)

    core_text = " | ".join(parts)
    token_count = _estimate_tokens(core_text)

    if token_count <= GEMINI_EMBED_TOKEN_LIMIT:
        logger.warning(
            f"[EMBED] Event '{event.event_name}' truncated: dropped parameters "
            f"({_estimate_tokens(full_text)} → {token_count} tokens)"
        )
        return core_text

    # Step 2: Drop detailed_event_definition, keep only name + screen + definition
    minimal_text = " | ".join([event.event_name, event.screen_name, event.event_definition])
    token_count = _estimate_tokens(minimal_text)

    logger.warning(
        f"[EMBED] Event '{event.event_name}' heavily truncated: "
        f"dropped parameters and detailed definition "
        f"({_estimate_tokens(full_text)} → {token_count} tokens)"
    )
    return minimal_text


# ---------------------------------------------------------------------------
# Gemini embedding helpers
# ---------------------------------------------------------------------------

def _make_gemini_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


_GEMINI_BATCH_LIMIT = 100


def _embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with Gemini.

    Returns a list of float vectors (dimension = GEMINI_EMBED_DIMENSION).
    Batches requests to stay within the 100-item-per-batch API limit.
    """
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _GEMINI_BATCH_LIMIT):
        batch = texts[i : i + _GEMINI_BATCH_LIMIT]
        result = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=batch,
            config={"output_dimensionality": GEMINI_EMBED_DIMENSION},
        )
        embeddings.extend(e.values for e in result.embeddings)
    return embeddings


def embed_query(client: genai.Client, query: str) -> list[float]:
    """Embed a single search query string."""
    return _embed_texts(client, [query])[0]


def deserialize_event_from_metadata(metadata: dict) -> Event:
    """Deserialize an Event object from Pinecone metadata."""
    from src.models import Parameter

    # Reconstruct parameters from JSON
    parameters = []
    if "parameters_json" in metadata and metadata["parameters_json"]:
        params_data = json.loads(metadata["parameters_json"])
        parameters = [
            Parameter(
                name=p["name"],
                description=p["description"],
                sample_values=p["sample_values"]
            )
            for p in params_data
        ]

    # Reconstruct Event from top-level metadata fields
    event = Event(
        event_name=metadata.get("event_name", ""),
        event_definition=metadata.get("event_definition", ""),
        screen_name=metadata.get("screen_name", ""),
        key_event=metadata.get("key_event", "No"),
        detailed_event_definition=metadata.get("detailed_definition", ""),
        parameters=parameters,
    )

    return event


# ---------------------------------------------------------------------------
# Corpus hashing for idempotency
# ---------------------------------------------------------------------------

def _compute_corpus_hash(events: list[Event]) -> str:
    """Compute a deterministic hash of the event corpus for idempotency checks.

    Hashes the to_index_text() of each event in order to detect content changes.
    """
    corpus_text = "||".join(e.to_index_text() for e in events)
    return hashlib.sha256(corpus_text.encode()).hexdigest()[:16]


def _get_corpus_hash_from_index(index: object) -> str | None:
    """Fetch the corpus hash from Pinecone index stats description."""
    try:
        stats = index.describe_index_stats()
        # Pinecone stores custom metadata in stats.namespaces if available
        # For now, we check if a sentinel metadata key exists (requires querying a vector)
        # A more robust approach: store in a reserved metadata field on all vectors
        return None  # TODO: implement persistent hash storage
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------

def _get_or_create_pinecone_index(pc: Pinecone) -> object:
    """Return a handle to the Pinecone index, creating it first if absent."""
    existing_names = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_names:
        print(
            f"  Creating Pinecone index '{PINECONE_INDEX_NAME}' "
            f"(dim={GEMINI_EMBED_DIMENSION}, metric=cosine, "
            f"cloud={PINECONE_CLOUD}, region={PINECONE_REGION})..."
        )
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=GEMINI_EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print("  Index created.")
    return pc.Index(PINECONE_INDEX_NAME)


def build_pinecone_index(events: list[Event], client: genai.Client) -> object:
    """Upsert all event embeddings into Pinecone.

    Idempotency: Computes a content hash of the corpus (to_index_text of all events).
    If the index has the same number of vectors AND the hash matches the stored hash,
    skips re-embedding. Otherwise, re-embeds all vectors.

    Token limit handling: Truncates texts intelligently if they exceed 2048 tokens.
    Priority: params first, then detailed_definition, keep event_name/screen/definition.

    Returns the Pinecone index handle.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = _get_or_create_pinecone_index(pc)

    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    current_corpus_hash = _compute_corpus_hash(events)

    # Check if we can skip re-embedding
    if total_vectors == len(events):
        # Try to fetch the stored corpus hash from the first vector's metadata
        try:
            sample_response = index.query(
                vector=[0.0] * GEMINI_EMBED_DIMENSION,
                top_k=1,
                include_metadata=True
            )
            if sample_response.matches:
                stored_hash = sample_response.matches[0].metadata.get("corpus_hash")
                if stored_hash == current_corpus_hash:
                    print(f"  Pinecone index already contains {total_vectors} vectors with matching corpus hash. Skipping re-embed.")
                    return index
        except Exception as e:
            print(f"  Warning: Could not verify corpus hash: {e}. Proceeding with re-embed.")

    print(f"  Embedding {len(events)} events with Gemini {GEMINI_EMBED_MODEL} (dim={GEMINI_EMBED_DIMENSION})...")
    # Apply token limit truncation before embedding
    texts = [_truncate_index_text_for_embedding(event) for event in events]

    # Filter out empty texts - Gemini API rejects empty parts
    event_text_pairs = [(event, text) for event, text in zip(events, texts) if text.strip()]

    if not event_text_pairs:
        logger.error("[INDEXER] All events produced empty text - cannot embed")
        raise ValueError("No valid events to embed (all texts are empty)")

    if len(event_text_pairs) < len(events):
        logger.warning(f"[INDEXER] Filtered out {len(events) - len(event_text_pairs)} events with empty text")

    # Extract events and texts from filtered pairs
    events_to_embed = [pair[0] for pair in event_text_pairs]
    texts_to_embed = [pair[1] for pair in event_text_pairs]

    embeddings = _embed_texts(client, texts_to_embed)

    timestamp = int(time.time())
    vectors = [
        (
            event.doc_id,
            emb,
            {
                "event_name": event.event_name,
                "screen_name": event.screen_name,
                "event_definition": event.event_definition,
                "detailed_definition": event.detailed_event_definition,
                "key_event": event.key_event,
                "parameters_json": json.dumps([
                    {
                        "name": p.name,
                        "description": p.description,
                        "sample_values": p.sample_values
                    }
                    for p in event.parameters
                ]),
                "corpus_hash": current_corpus_hash,
                "embedding_model": GEMINI_EMBED_MODEL,
                "embedding_dim": GEMINI_EMBED_DIMENSION,
                "indexed_at": timestamp,
            },
        )
        for event, emb in zip(events_to_embed, embeddings)
    ]

    for i in range(0, len(vectors), _UPSERT_BATCH_SIZE):
        batch = vectors[i : i + _UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)

    print(f"  Upserted {len(vectors)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'.")
    return index


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def build_indexes(events: list[Event]) -> tuple[genai.Client, object, BM25Okapi]:
    """Build (or connect to) all indexes needed for hybrid search.

    Returns:
        client        — Gemini client (reused for query embedding)
        pinecone_idx  — Pinecone index handle
        bm25          — In-memory BM25Okapi instance
    """
    client = _make_gemini_client()
    pinecone_idx = build_pinecone_index(events, client)
    bm25 = build_bm25_index(events)
    return client, pinecone_idx, bm25
