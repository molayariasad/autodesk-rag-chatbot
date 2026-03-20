"""
Web search integration for blended RAG mode.

The assignment requires two modes:
1. Corpus-only: Answer from Autodesk HTML documents only.
2. Blended: Merge corpus results with live web search results.

We use DuckDuckGo search (no API key, no data storage, privacy-friendly).
"""
"""
Web search integration for blended RAG mode.

v2 fix: ddgs library API changed — DDGS().text() signature is now:
    text(keywords: str, ...) — positional, not keyword argument.
Also added a fallback for both old (duckduckgo_search) and new (ddgs) import paths.
"""

from loguru import logger

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False
        logger.warning("Neither ddgs nor duckduckgo_search installed. Web search disabled.")


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo.
    Returns list of dicts with keys: chunk_id, text, metadata, score, source.
    """
    if not HAS_DDGS:
        logger.warning("Web search unavailable (ddgs not installed)")
        return []

    try:
        results = []
        with DDGS() as ddgs:
            # FIX: pass query as positional argument, not keyword
            for r in ddgs.text(f"Autodesk {query}", max_results=max_results, region="us-en"):
                results.append(r)

        hits = []
        for i, r in enumerate(results):
            text = f"Title: {r.get('title', '')}\n{r.get('body', '')}"
            hits.append({
                "chunk_id": f"web_search::{i}",
                "text": text,
                "metadata": {
                    "source_file": r.get("href", "web"),
                    "title": r.get("title", ""),
                    "product_name": "",
                    "page_type": "web_search",
                    "chunk_type": "web",
                    "url": r.get("href", ""),
                },
                "score": 1.0 - (i * 0.1),
                "source": "web_search",
            })

        logger.info(f"Web search returned {len(hits)} results for: {query}")
        return hits

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []