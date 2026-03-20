# src/ingestion/html_parser_baseline.py
"""
Baseline HTML Parser — Experiment 01_baseline.

This is the v1 parser used to establish the baseline RAG Triad scores.
It intentionally preserves the two bugs that were later fixed in next iterations, as well as the lack of product alias normalization. These issues:

  Bug 1 — Extraction order: _remove_boilerplate() runs on the shared soup
  object BEFORE BS4 extraction. BS4 therefore operates on the already-
  stripped DOM and cannot rescue content that trafilatura dropped.

  Bug 2 — trafilatura precision mode: favor_precision=True (the library
  default) aggressively filters "secondary" content — sidebars, comparison
  sections, feature tables. This is exactly where competitor names
  ("vs SolidWorks"), feature contrasts ("unlike Revit"), and version strings
  ("Maya 2025") live, causing recall=0 on those query types.

  Missing — no product alias normalization: internal Autodesk codes
  (ACDIST, F360, ARCHDESK) are stored as-is in chunk metadata. Neither
  BM25 nor embeddings can match "F360" against a user query for "Fusion 360".

These deficiencies are intentional — they establish the lower bound that
the optimized parser (html_parser.py) is measured against.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup, Comment
from loguru import logger
import trafilatura
from markdownify import markdownify as md
import html2text

MIN_TEXT_LENGTH = 100

# Minimal product list — no alias normalization, no internal code mapping.
# Raw meta-tag values (ACDIST, F360) pass through unchanged.
KNOWN_PRODUCTS_BASELINE = sorted([
    "AutoCAD", "AutoCAD LT", "Revit", "Fusion 360", "Maya", "3ds Max",
    "Inventor", "Civil 3D", "Navisworks", "Forma", "MotionBuilder",
    "Construction Cloud",
], key=len, reverse=True)

NAV_SELECTORS = [
    "nav", "header", "footer",
    ".nav", ".navbar", ".header", ".footer",
    ".cookie-banner", ".cookie-consent",
    ".breadcrumb", ".breadcrumbs",
    ".site-header", ".site-footer",
    "#header", "#footer", "#nav",
    "[role='navigation']", "[role='banner']", "[role='contentinfo']",
]

NOISE_TAGS = ["script", "style", "noscript", "iframe", "svg", "canvas", "link"]


@dataclass
class ParsedDocument:
    """Represents a cleaned, parsed HTML document."""
    source_file: str
    source_path: str
    title: str
    content: str
    tables_markdown: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    product_name: Optional[str] = None
    page_type: str = "unknown"
    char_count: int = 0

    def full_text(self) -> str:
        parts = [self.content]
        if self.tables_markdown:
            parts.append("\n\n## Tables\n")
            parts.extend(self.tables_markdown)
        return "\n\n".join(parts)


def _extract_metadata_from_soup(soup: BeautifulSoup) -> dict:
    meta = {}
    for tag in soup.find_all("meta"):
        name = tag.get("name", "").lower()
        content_val = tag.get("content", "")
        if name and content_val:
            meta[name] = content_val
    for tag in soup.find_all("meta", attrs={"property": True}):
        prop = tag.get("property", "").lower()
        content_val = tag.get("content", "")
        if prop and content_val:
            meta[prop] = content_val
    return meta


def _detect_product_name(soup: BeautifulSoup, meta: dict, text: str) -> Optional[str]:
    """
    Baseline product detection — no alias normalization.
    Raw meta-tag values stored as-is (ACDIST, F360, ARCHDESK, etc.).
    """
    # No normalization: raw value from meta tag passes through unchanged
    if "product" in meta and meta["product"]:
        return meta["product"].strip()

    title = ""
    if "og:title" in meta:
        title = meta["og:title"]
    elif soup.title and soup.title.string:
        title = soup.title.string

    scan_text = (title + " " + text[:1500]).lower()
    for product in KNOWN_PRODUCTS_BASELINE:
        if product.lower() in scan_text:
            return product

    return None


def _detect_page_type(soup: BeautifulSoup, meta: dict, text: str) -> str:
    title = (soup.title.string if soup.title and soup.title.string else "").lower()
    if meta.get("topic-type"):
        return "help"
    if any(s in title for s in ["help", "documentation", "user guide", "how to"]):
        return "help"
    if any(s in title for s in ["blog", "news", "what's new"]):
        return "blog"
    if any(s in title for s in ["buy", "price", "overview", "features"]):
        return "product"
    return "unknown"


def _extract_tables_as_markdown(soup: BeautifulSoup) -> list[str]:
    tables_md = []
    for table in soup.find_all("table"):
        try:
            table_md = md(str(table), strip=["img", "a"])
            table_md = re.sub(r"\n{3,}", "\n\n", table_md).strip()
            if len(table_md) > 20:
                tables_md.append(table_md)
        except Exception as e:
            logger.debug(f"Table conversion failed: {e}")
    return tables_md


def _remove_boilerplate(soup: BeautifulSoup) -> BeautifulSoup:
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    for selector in NAV_SELECTORS:
        try:
            for el in soup.select(selector):
                el.decompose()
        except Exception:
            pass
    return soup


def _clean_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = text.split("\n")
    cleaned_lines = [l for l in lines if len(l.strip()) > 1 or l.strip() == ""]
    return "\n".join(cleaned_lines).strip()


def parse_html_file(filepath: Path) -> Optional[ParsedDocument]:
    """
    Baseline single-pass parser.

    Bugs preserved intentionally for baseline measurement:
    1. _remove_boilerplate() runs on the shared soup BEFORE BS4 extraction.
       BS4 therefore sees the stripped DOM, not the original.
    2. trafilatura uses favor_precision=True (library default), which strips
       competitor comparisons, feature sidebars, and secondary content.
    3. No product alias normalization — raw codes stored as-is.
    """
    filepath = Path(filepath)
    logger.debug(f"Parsing (baseline): {filepath.name}")

    try:
        raw_html = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Cannot read {filepath.name}: {e}")
        return None

    if len(raw_html) < 200:
        return None

    soup = BeautifulSoup(raw_html, "lxml")
    meta = _extract_metadata_from_soup(soup)

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif "og:title" in meta:
        title = meta["og:title"].strip()
    else:
        title = filepath.stem

    # Tables extracted before boilerplate removal (same as v1)
    tables_md = _extract_tables_as_markdown(soup)

    # BUG 1: boilerplate removal on the SHARED soup object.
    # After this call, soup is stripped — BS4 below sees the degraded DOM.
    soup = _remove_boilerplate(soup)

    # Layer 1: trafilatura with favor_precision=True (library default).
    # BUG 2: precision mode strips secondary content where competitor names live.
    content = ""
    try:
        traf_result = trafilatura.extract(
            raw_html,
            include_tables=True,
            include_links=False,
            include_images=False,
            include_comments=False,
            favor_precision=True,   # BASELINE: default precision mode
            favor_recall=False,
            output_format="txt",
        )
        if traf_result:
            content = traf_result
    except Exception as e:
        logger.debug(f"  trafilatura failed: {e}")

    # Layer 2: BS4 fallback — but now runs on the already-stripped soup (bug 1)
    if len(content) < MIN_TEXT_LENGTH:
        try:
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", class_=re.compile(r"body|content|main", re.I))
                or soup.find("body")
            )
            if main_content:
                content = main_content.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.debug(f"  BS4 failed: {e}")

    # Layer 3: html2text last resort
    if len(content) < MIN_TEXT_LENGTH:
        try:
            h2t = html2text.HTML2Text()
            h2t.ignore_links = True
            h2t.ignore_images = True
            h2t.body_width = 0
            h2t_result = h2t.handle(raw_html)
            if len(h2t_result) > len(content):
                content = h2t_result
        except Exception:
            pass

    content = _clean_text(content)

    if len(content) < MIN_TEXT_LENGTH:
        logger.info(f"Skipping {filepath.name}: insufficient content ({len(content)} chars)")
        return None

    product = _detect_product_name(soup, meta, content)
    page_type = _detect_page_type(soup, meta, content)
    source_path = f"data/raw/{filepath.name}"

    return ParsedDocument(
        source_file=filepath.name,
        source_path=source_path,
        title=title,
        content=content,
        tables_markdown=tables_md,
        metadata=meta,
        product_name=product,
        page_type=page_type,
        char_count=len(content),
    )


def parse_directory(dir_path: str | Path) -> list[ParsedDocument]:
    dir_path = Path(dir_path)
    html_files = sorted(dir_path.glob("*.html"))
    logger.info(f"Baseline parser: found {len(html_files)} HTML files in {dir_path}")

    documents = []
    skipped = 0
    for fp in html_files:
        doc = parse_html_file(fp)
        if doc:
            documents.append(doc)
        else:
            skipped += 1

    logger.info(f"Baseline parsing complete: {len(documents)} docs, {skipped} skipped")
    return documents