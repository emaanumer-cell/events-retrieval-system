"""Streamlit app for keyword-based event retrieval with bi-encoder vs cross-encoder comparison."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import streamlit as st
import pandas as pd
from google import genai
from pinecone import Pinecone

from src.config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, GEMINI_EMBED_DIMENSION
from src.query import process_query
from src.retriever import fetch_all_events, biencoder_search
from src.reranker import crossencoder_score

# ---------------------------------------------------------------------------
# Logging — overwrites logs/latest.log each run
# ---------------------------------------------------------------------------
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Clear existing handlers (Streamlit re-runs the script)
if not root_logger.handlers:
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_dir / "latest.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------
if "keywords" not in st.session_state:
    st.session_state.keywords = []


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


@st.cache_data(ttl=600)
def get_screen_names() -> list[str]:
    """Fetch unique screen names by querying Pinecone with a zero vector."""
    idx = get_pinecone_index()
    # Use a zero vector query to pull back all events and extract screen names
    dummy_vec = [0.0] * GEMINI_EMBED_DIMENSION
    response = idx.query(vector=dummy_vec, top_k=10000, include_metadata=True)
    screens = {
        m.metadata["screen_name"]
        for m in response.matches
        if m.metadata and "screen_name" in m.metadata
    }
    return sorted(screens)


# ---------------------------------------------------------------------------
# Example keywords
# ---------------------------------------------------------------------------
EXAMPLE_KEYWORDS = [
    "acquisition",
    "activation",
    "retention",
    "referral",
    "revenue",
    "animation",
    "ad engagement",
]


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Event Retrieval System", layout="wide")
st.title("Event Retrieval System")
# ---------------------------------------------------------------------------
# Keyword Input Section
# ---------------------------------------------------------------------------

def add_keyword_on_enter():
    """Callback to add keyword when Enter is pressed."""
    keyword_value = st.session_state.keyword_input.strip().lower()
    if keyword_value and keyword_value not in st.session_state.keywords:
        st.session_state.keywords.append(keyword_value)
    st.session_state.keyword_input = ""  # Clear input after adding

st.subheader("Add Keywords")
st.text_input(
    "Enter a keyword and press Enter",
    placeholder="e.g., acquisition, revenue, retention, engagement...",
    label_visibility="collapsed",
    key="keyword_input",
    on_change=add_keyword_on_enter
)

# Display keywords as removable tags
if st.session_state.keywords:
    st.write("**Selected Keywords:**")
    cols = st.columns(len(st.session_state.keywords))
    for idx, keyword in enumerate(st.session_state.keywords):
        with cols[idx]:
            if st.button(f"✕ {keyword}", key=f"remove_{idx}", use_container_width=True):
                st.session_state.keywords.pop(idx)
                st.rerun()
else:
    st.info("Type a keyword and press Enter to get started.")

# Example keywords
with st.expander("📚 Example Keywords"):
    cols = st.columns(3)
    for idx, example in enumerate(EXAMPLE_KEYWORDS):
        with cols[idx % 3]:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                if example not in st.session_state.keywords:
                    st.session_state.keywords.append(example)
                st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Filters and Options Section
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    enable_expansion = st.checkbox("Enable query expansion", value=True, help="Use Gemini to expand keywords with related terms")

screens = get_screen_names()
selected_screens = st.multiselect("Filter by screen (optional)", options=screens)

col3, col4 = st.columns(2)
with col3:
    if st.button("Search", use_container_width=True, type="primary"):
        if not st.session_state.keywords:
            st.error("Please add at least one keyword before searching.")
        else:
            st.session_state.search_triggered = True

with col4:
    if st.button("Clear All", use_container_width=True):
        st.session_state.keywords = []
        st.rerun()

search_clicked = st.session_state.get("search_triggered", False)

# ---------------------------------------------------------------------------
# Search Logic
# ---------------------------------------------------------------------------
if search_clicked:
    gemini_client = get_gemini_client()
    pinecone_index = get_pinecone_index()

    # Step 1: Query processing (no POS tagging/lemmatization) + optional expansion
    with st.spinner("Processing keywords: optional expansion..."):
        processed = process_query(st.session_state.keywords, expand=enable_expansion)
    st.info(f"**Processed query:** {processed[:300]}")

    # Step 2: Fetch events from Pinecone
    with st.spinner("Fetching events from Pinecone..."):
        screen_filter = selected_screens if selected_screens else None
        events, cosine_scores = fetch_all_events(pinecone_index, gemini_client, processed, screens=screen_filter)

    if not events:
        st.warning("No events found in Pinecone matching your filters.")
    else:
        st.success(f"Retrieved {len(events)} events from Pinecone")

        # Step 3: Bi-encoder search
        with st.spinner("Running bi-encoder hybrid search (BM25 + dense + RRF)..."):
            bi_results = biencoder_search(events, cosine_scores, processed)

        # Step 4: Cohere Rerank scoring
        with st.spinner("Running Cohere Rerank scoring..."):
            ce_results = crossencoder_score(events, processed)

        # --- Bi-Encoder Results Section ---
        st.subheader("Bi-Encoder Results (BM25 + Dense + RRF)")
        bi_df = pd.DataFrame([
            {
                "RRF Score": round(r['rrf_score'], 4),
                "BM25 Score": round(r['bm25_score'], 4),
                "Dense Score": round(r['dense_score'], 4),
                "Event Name": r["event"].event_name,
                "Screen": r["event"].screen_name,
                "Definition": r["event"].event_definition,
                "Key Event": r["event"].key_event,
            }
            for r in bi_results
        ])
        st.dataframe(bi_df, use_container_width=True, hide_index=True)

        # --- Cohere Rerank Results Section ---
        st.subheader("Cohere Rerank Results (rerank-v4.0-pro)")
        ce_df = pd.DataFrame([
            {
                "Score": round(r['score'], 4),
                "Event Name": r["event"].event_name,
                "Screen": r["event"].screen_name,
                "Definition": r["event"].event_definition,
                "Key Event": r["event"].key_event,
            }
            for r in ce_results
        ])
        st.dataframe(ce_df, use_container_width=True, hide_index=True)

    # Reset search state
    st.session_state.search_triggered = False
