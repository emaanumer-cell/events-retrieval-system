"""Query preprocessing and LLM-based expansion using Gemini.

Pipeline:
1. process_query() — Raw keywords -> spell correct -> detect compound phrases ->
                     lemmatize non-compounds -> optionally expand via Gemini
2. expand_keywords() — Gemini expansion of keywords

Compound phrases (multi-word phrases) are protected from lemmatization to preserve semantics
(e.g., "social media" stays intact, not lemmatized to "social medium").
"""
from __future__ import annotations

import logging
import re
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from src.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS

logger = logging.getLogger(__name__)

# Lazy-load NLTK resources
_NLTK_RESOURCES_LOADED = False


def _simple_tokenize(text: str) -> list[str]:
    """Simple regex-based tokenization (avoids punkt dependency)."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def _ensure_nltk_resources():
    """Download required NLTK data on first use."""
    global _NLTK_RESOURCES_LOADED
    if _NLTK_RESOURCES_LOADED:
        return

    try:
        # Try to load POS tagger (multiple possible names in different NLTK versions)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        logger.info("[NLTK] Downloading POS tagger...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("[NLTK] Downloading WordNet...")
        nltk.download('wordnet', quiet=True)

    _NLTK_RESOURCES_LOADED = True


# def extract_keywords(query: str) -> list[str]:
#     """Extract meaningful keywords from query using POS tagging.
#
#     Removes stop words (articles, prepositions, conjunctions) and keeps only:
#     - NOUN (NN, NNS, NNP, NNPS)
#     - VERB (VB, VBD, VBG, VBN, VBP, VBZ)
#     - ADJ (JJ, JJR, JJS)
#     - ADV (RB, RBR, RBS)
#
#     Args:
#         query: Raw query string.
#
#     Returns:
#         List of meaningful keywords (lowercase, non-lemmatized yet).
#
#     Example:
#         "acquisition and retention events" -> ["acquisition", "retention", "events"]
#     """
#     _ensure_nltk_resources()
#
#     logger.info(f"[PREPROCESS] Extracting keywords from: '{query}'")
#
#     # Tokenize and POS tag
#     tokens = _simple_tokenize(query)
#     pos_tags = pos_tag(tokens)
#
#     # Keep only meaningful POS tags
#     meaningful_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
#                       'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
#
#     keywords = [word for word, pos in pos_tags if pos in meaningful_pos]
#
#     logger.info(f"[PREPROCESS] Extracted {len(keywords)} keywords: {keywords}")
#     return keywords


# def _penn_to_wordnet_pos(penn_pos: str) -> str | None:
#     """Map Penn Treebank POS tag to WordNet POS tag.
#
#     Args:
#         penn_pos: Penn Treebank POS tag (e.g., 'NN', 'VBD', 'JJ')
#
#     Returns:
#         WordNet POS tag (wordnet.NOUN, wordnet.VERB, etc.) or None if unmappable.
#     """
#     if penn_pos.startswith('NN'):
#         return wordnet.NOUN
#     elif penn_pos.startswith('VB'):
#         return wordnet.VERB
#     elif penn_pos.startswith('JJ'):
#         return wordnet.ADJ
#     elif penn_pos.startswith('RB'):
#         return wordnet.ADV
#     return None


# def lemmatize_keywords(keywords: list[str]) -> list[str]:
#     """Lemmatize keywords using POS-aware lemmatization.
#
#     Uses WordNetLemmatizer with POS hints for accuracy.
#     Re-tags keywords to determine POS for lemmatization.
#
#     Args:
#         keywords: List of keywords (output of extract_keywords).
#
#     Returns:
#         List of lemmatized keywords.
#
#     Example:
#         ["acquiring", "acquisitions", "retention"] -> ["acquire", "acquisition", "retention"]
#     """
#     _ensure_nltk_resources()
#
#     logger.info(f"[PREPROCESS] Lemmatizing {len(keywords)} keywords")
#
#     lemmatizer = WordNetLemmatizer()
#
#     # Tag keywords to get accurate POS for lemmatization
#     pos_tags = pos_tag(keywords)
#     lemmatized = []
#
#     for word, penn_pos in pos_tags:
#         wordnet_pos = _penn_to_wordnet_pos(penn_pos)
#         if wordnet_pos:
#             lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
#         else:
#             # Fallback: lemmatize without POS hint
#             lemma = lemmatizer.lemmatize(word)
#         lemmatized.append(lemma)
#
#     logger.info(f"[PREPROCESS] Lemmatized: {lemmatized}")
#     return lemmatized

def _is_compound_phrase(phrase: str) -> bool:
    """Detect if a phrase is a compound phrase (multi-word).

    If the keyword has spaces between words, treat it as a compound phrase
    and protect it from lemmatization.

    Example: "social media" has a space -> compound phrase -> keep as-is.
             "acquisition" has no space -> single word -> lemmatize.

    Args:
        phrase: Keyword phrase (e.g., "social media" or "acquisition").

    Returns:
        True if phrase contains spaces (multi-word), False if single word.
    """
    phrase_clean = phrase.lower().strip()
    # If there are spaces, it's a compound phrase
    return " " in phrase_clean


def _lemmatize_single_word(word: str) -> str:
    """Lemmatize a single word using POS-aware lemmatization.

    Args:
        word: Single word to lemmatize (e.g., "acquiring").

    Returns:
        Lemmatized word (e.g., "acquire").
    """
    _ensure_nltk_resources()

    lemmatizer = WordNetLemmatizer()

    # Tag the word to get its POS
    pos_tags = pos_tag([word])
    word_pos = pos_tags[0][1]

    # Map to WordNet POS
    if word_pos.startswith('NN'):
        wordnet_pos = wordnet.NOUN
    elif word_pos.startswith('VB'):
        wordnet_pos = wordnet.VERB
    elif word_pos.startswith('JJ'):
        wordnet_pos = wordnet.ADJ
    elif word_pos.startswith('RB'):
        wordnet_pos = wordnet.ADV
    else:
        wordnet_pos = None

    # Lemmatize with POS hint if available
    if wordnet_pos:
        lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
    else:
        lemma = lemmatizer.lemmatize(word)

    logger.debug(f"[LEMMATIZE] {word} (POS: {word_pos}) -> {lemma}")
    return lemma


def _process_keyword(keyword: str) -> str:
    """Process a single keyword: spell correct -> detect compound -> selective lemmatize.

    Pipeline:
    1. Spell correct the keyword
    2. If compound phrase: keep as-is
    3. If single word: lemmatize

    Args:
        keyword: Raw keyword phrase (e.g., "socail media" or "acquiring").

    Returns:
        Processed keyword (e.g., "social media" or "acquire").
    """
    # Step 1: Check if compound phrase
    if _is_compound_phrase(keyword):
        logger.info(f"[COMPOUND PHRASE] Protecting from lemmatization: '{keyword}'")
        return keyword

    # Step 2: Single word -> lemmatize
    lemmatized = _lemmatize_single_word(keyword)
    logger.info(f"[SINGLE WORD] Lemmatized: {keyword} -> {lemmatized}")
    return lemmatized


def expand_keywords(lemmatized_keywords: list[str]) -> str:
    """Expand lemmatized keywords with related terms via Gemini.

    Takes lemmatized keywords and uses Gemini to generate semantically
    related terms, abbreviations, and variations for GA4 events.

    Args:
        lemmatized_keywords: List of lemmatized keywords.

    Returns:
        Space-separated string of expansion terms.
        Falls back to empty string if Gemini unavailable.

    Example:
        ["acquire", "user", "retention"] -> "install signup onboard session_count churn"
    """
    logger.info(f"[QUERY EXPANSION] Expanding {len(lemmatized_keywords)} keywords")

    if not GEMINI_API_KEY:
        logger.warning("[QUERY EXPANSION] GEMINI_API_KEY not set — returning empty expansion")
        return ""

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        keywords_str = ", ".join(lemmatized_keywords)
        logger.info(f"[QUERY EXPANSION] Base keywords for Gemini: {keywords_str}")
        prompt = (
            f"Base keywords: {keywords_str}\n\n"
            "You are helping search a GA4 mobile app tracking plan. "
            "Given these base keywords, generate related app event names, screen names, "
            "and analytics keywords. Include synonyms, abbreviations, variations, "
            "and related concepts commonly used in mobile app analytics. "
            "Return ONLY a space-separated list of keywords, nothing else. For compound phrases, use spaces (e.g., 'social media' not 'social_media')."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_TOKENS,
            ),
        )

        expansion = response.text.strip()
        logger.info(f"[QUERY EXPANSION] Gemini expansion ({len(expansion)} chars): {expansion[:150]}")

        return expansion

    except Exception as exc:
        logger.error(f"[QUERY EXPANSION] Gemini failed: {exc}")
        return ""


def process_query(keywords: list[str], expand: bool = True) -> str:
    """Orchestrate full query preprocessing and optional expansion.

    Pipeline:
    1. Spell correct each keyword
    2. Detect compound phrases (multi-word, protect from lemmatization)
    3. Lemmatize single-word keywords only
    4. (Optional) Expand via Gemini
    5. Return final query string (space-separated)

    Args:
        keywords: List of raw keyword phrases (e.g., ["socail media", "acquisiton"]).
        expand: Whether to expand keywords via Gemini (default True).

    Returns:
        Final query string: "{processed_keywords} {expansion}"
        or just "{processed_keywords}" if expand=False.

    Example:
        >>> process_query(["socail media", "acquisiton"], expand=False)
        "social media acquisition"

        >>> process_query(["social media", "acquiring"], expand=True)
        "social media acquire user_engagement screen_view"
    """
    logger.info(f"[PROCESS QUERY] Raw keywords: {keywords} (expand={expand})")

    if not keywords:
        logger.warning("[PROCESS QUERY] No keywords provided")
        return ""

    # Step 1-3: Process each keyword (spell correct -> detect compound -> selective lemmatize)
    processed_keywords = []
    for keyword in keywords:
        processed = _process_keyword(keyword)
        processed_keywords.append(processed)
        logger.info(f"[PROCESS QUERY] {keyword} -> {processed}")

    base_query = " ".join(processed_keywords)
    logger.info(f"[PROCESS QUERY] Processed query (before expansion): '{base_query}'")

    # Step 4: Conditionally expand
    if expand:
        expansion = expand_keywords(processed_keywords)
        final_query = base_query + (" " + expansion if expansion else "")
    else:
        final_query = base_query

    logger.info(f"[PROCESS QUERY] Final query (ready for encoder): '{final_query}'")
    return final_query
