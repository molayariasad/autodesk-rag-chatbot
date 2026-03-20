"""
HTML Parser for Autodesk web pages.

Strategy:
1. Multi-layer extraction: trafilatura (best for article/marketing pages)
   + BeautifulSoup (for structured help docs) + html2text (fallback).
   + We always take the LONGER result, because trafilatura with favor_precision=True aggressively 
     strips secondary content (sidebars,
   comparison sections) where competitor names and feature details often live.
2. Table-aware parsing: HTML tables → Markdown tables so the LLM can reason over them.
3. Noise removal: Strip JS, CSS, nav bars, footers, cookie banners, tracking pixels.
4. Quality filtering: Skip blank pages, JS-only shells, and pages with < MIN_TEXT_LENGTH chars.
5. Metadata extraction: Title, product name, page type, URL hints from filenames.
6. Source path tracking: Store relative file path for clickable UI links.
7. Product normalization: Map Autodesk internal codes (ACDIST, F360, etc. to
   canonical product names so embeddings and BM25 can match user queries.
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

# Minimum meaningful text length (chars) after cleaning
MIN_TEXT_LENGTH = 300

# ============================================================
# Product Name Normalization
# ============================================================
# Maps Autodesk internal meta-tag codes AND common variants to
# canonical product names. This is critical for retrieval: if a
# chunk says "ACDIST" but the user asks about "AutoCAD", neither
# BM25 nor embeddings will match.

PRODUCT_ALIASES: dict[str, str] = {
    # AutoCAD family
    "ACDIST": "AutoCAD",
    "ACD": "AutoCAD",
    "ACAD": "AutoCAD",
    "ACAD_E": "AutoCAD Electrical",
    "ACADELECTRICAL": "AutoCAD Electrical",
    "ARCHDESK": "AutoCAD Architecture",
    "ACAD_ARCH": "AutoCAD Architecture",
    "ACAD_MEP": "AutoCAD MEP",
    "ACADMEP": "AutoCAD MEP",
    "MAP3D": "AutoCAD Map 3D",
    "ACAD_MAP": "AutoCAD Map 3D",
    "PLNT3D": "AutoCAD Plant 3D",
    "ACAD_PLANT": "AutoCAD Plant 3D",
    "ACDLTG": "AutoCAD LT",
    "ACLT": "AutoCAD LT",
    "AUTOCADLT": "AutoCAD LT",
    "AUTOCAD": "AutoCAD",
    # Fusion 360
    "F360": "Fusion 360",
    "FUSION360": "Fusion 360",
    "FUSION": "Fusion 360",
    # Inventor
    "INVNTOR": "Inventor",
    "INV": "Inventor",
    "INVPROSA": "Inventor",
    # Revit
    "RVIT": "Revit",
    "RVT": "Revit",
    "REVIT": "Revit",
    # Maya
    "MAYA": "Maya",
    "MAYALT": "Maya Creative",
    "MAYA_LT": "Maya Creative",
    # 3ds Max
    "3DSMAX": "3ds Max",
    "3DS_MAX": "3ds Max",
    "MAX": "3ds Max",
    # Civil 3D
    "CIV3D": "Civil 3D",
    "CIVIL3D": "Civil 3D",
    "C3D": "Civil 3D",
    # Navisworks
    "NAVSIM": "Navisworks Simulate",
    "NAVMAN": "Navisworks Manage",
    "NAVISWORKS": "Navisworks",
    "NWMAN": "Navisworks Manage",
    "NWSIM": "Navisworks Simulate",
    # Forma
    "FORMA": "Forma",
    # MotionBuilder
    "MOBPRO": "MotionBuilder",
    "MOTIONBUILDER": "MotionBuilder",
    # Infrastructure / InfraWorks
    "IW360P": "InfraWorks",
    "INFRAWORKS": "InfraWorks",
    # Alias
    "ALIAS": "Alias",
    "ALSCPT": "Alias",
    "ALSSRF": "Alias",
    # VRED
    "VRED": "VRED",
    "VREDPRO": "VRED Professional",
    # Vault
    "VAULT": "Vault",
    "VLTM": "Vault",
    # Moldflow
    "MFLDAN": "Moldflow Adviser",
    "MFLI": "Moldflow Insight",
    "MOLDFLOW": "Moldflow",
    # Manufacturing / CAM
    "PWRMILL": "PowerMill",
    "POWERMILL": "PowerMill",
    "PWRSHP": "PowerShape",
    "POWERSHAPE": "PowerShape",
    "FEATURECAM": "FeatureCAM",
    "FCAM": "FeatureCAM",
    "NETFABB": "Netfabb",
    # Construction Cloud
    "BIM360": "BIM 360",
    "BIM_360": "BIM 360",
    "ACC": "Autodesk Construction Cloud",
    "ADSK_BUILD": "Autodesk Build",
    "ADSK_DOCS": "Autodesk Docs",
    # ShotGrid / Flow
    "SGSUB": "ShotGrid",
    "SHOTGRID": "ShotGrid",
    "SHOTGUN": "ShotGrid",
    "FLOW": "Flow Production Tracking",
    # Arnold
    "ARNOLD": "Arnold",
    "ARNOLDRENDER": "Arnold",
    # Flame / Smoke
    "FLAME": "Flame",
    "SMOKE": "Smoke",
    # Recap
    "RECAP": "ReCap",
    "RECAPPRO": "ReCap Pro",
}

# Canonical product list for title/content scanning (longest first for greedy match)
KNOWN_PRODUCTS = sorted([
    "AutoCAD Architecture", "AutoCAD Electrical", "AutoCAD MEP",
    "AutoCAD Plant 3D", "AutoCAD Map 3D", "AutoCAD LT", "AutoCAD",
    "Autodesk Construction Cloud", "Autodesk Build", "Autodesk Docs",
    "Autodesk Takeoff", "BIM Collaborate",
    "Revit", "Fusion 360", "Maya Creative", "Maya", "3ds Max",
    "Inventor", "Civil 3D", "Navisworks Manage", "Navisworks Simulate",
    "Navisworks", "BIM 360", "Forma", "MotionBuilder",
    "Alias", "VRED Professional", "VRED", "Moldflow", "Netfabb",
    "PowerMill", "PowerShape", "FeatureCAM", "InfraWorks",
    "Vault", "ShotGrid", "Flow Production Tracking",
    "Arnold", "Flame", "Smoke", "ReCap Pro", "ReCap",
    "Construction Cloud", "Solidworks",
], key=len, reverse=True)  # Longest first so "AutoCAD Architecture" matches before "AutoCAD"


def _normalize_product_name(raw_name: str) -> str:
    """
    Normalize a product name from meta tags or detection.
    Maps internal codes (ACDIST, F360) to canonical names.
    """
    if not raw_name:
        return ""

    # Try exact match in alias table (case-insensitive, spaces stripped)
    key = raw_name.strip().upper().replace(" ", "").replace("-", "").replace("_", "")
    if key in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[key]

    # Try with underscores preserved (some codes use them)
    key_under = raw_name.strip().upper()
    if key_under in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[key_under]

    # If it looks like a real product name (has lowercase), return as-is
    if any(c.islower() for c in raw_name) and len(raw_name) > 3:
        return raw_name.strip()

    logger.debug(f"Unknown product code: '{raw_name}' — consider adding to PRODUCT_ALIASES")
    return raw_name.strip()


# ============================================================
# Navigation / boilerplate patterns to strip
# ============================================================
NAV_SELECTORS = [
    "nav", "header", "footer",
    ".nav", ".navbar", ".header", ".footer",
    ".cookie-banner", ".cookie-consent",
    ".breadcrumb", ".breadcrumbs",
    ".site-header", ".site-footer",
    "#header", "#footer", "#nav",
    "[role='navigation']", "[role='banner']", "[role='contentinfo']",
    ".social-share", ".share-buttons",
    ".ad-banner", ".advertisement",
    ".skip-to-content", ".skip-link",
]

NOISE_TAGS = ["script", "style", "noscript", "iframe", "svg", "canvas", "link"]


@dataclass
class ParsedDocument:
    """Represents a cleaned, parsed HTML document."""
    source_file: str        # Filename only: "adsk-abc123.html"
    source_path: str        # Relative path for UI links: "data/raw/adsk-abc123.html"
    title: str
    content: str
    tables_markdown: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    product_name: Optional[str] = None
    page_type: str = "unknown"
    char_count: int = 0

    def full_text(self) -> str:
        """Combine content and tables into a single text block."""
        parts = [self.content]
        if self.tables_markdown:
            parts.append("\n\n## Tables\n")
            parts.extend(self.tables_markdown)
        return "\n\n".join(parts)


def _extract_metadata_from_soup(soup: BeautifulSoup) -> dict:
    """Extract metadata from <meta> tags and OpenGraph properties."""
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
    Product name extraction with normalization.
    Priority: meta tag > title scan > content scan.
    """
    # Priority 1: From meta tags (help pages have 'product' meta)
    if "product" in meta:
        normalized = _normalize_product_name(meta["product"])
        if normalized:
            return normalized

    # Priority 2: Title (og:title or <title>)
    title = ""
    if "og:title" in meta:
        title = meta["og:title"]
    elif soup.title and soup.title.string:
        title = soup.title.string

    # Priority 3: Scan title + first 1500 chars for known product names
    scan_text = (title + " " + text[:1500]).lower()

    for product in KNOWN_PRODUCTS:
        if product.lower() in scan_text:
            return product

    return None


def _detect_page_type(soup: BeautifulSoup, meta: dict, text: str) -> str:
    """
    Classify page type using meta tags, title, and content signals.
    Expanded heuristics to minimize "unknown" rate.
    """
    title = (soup.title.string if soup.title and soup.title.string else "").lower()
    text_lower = text[:2000].lower() if text else ""

    # --- Strong meta signals ---
    if meta.get("topic-type") or meta.get("helpsystempath"):
        return "help"

    og_type = meta.get("og:type", "").lower()
    if og_type in ("article", "blog"):
        return "blog"

    # --- Title signals ---
    help_signals = [
        "help", "documentation", "user guide", "reference", "dialog box",
        "command", "tutorial", "how to", "troubleshoot", "about using",
        "about creating", "about editing", "workflow", "utility",
    ]
    if any(s in title for s in help_signals):
        return "help"

    blog_signals = [
        "blog", "news", "stories", "what's new", "what\u2019s new",
        "update", "release note", "announcement", "recap", "roundup",
        "tips and tricks", "behind the scenes",
    ]
    if any(s in title for s in blog_signals):
        return "blog"

    product_signals = [
        "buy", "price", "pricing", "subscribe", "free trial",
        "get prices", "overview", "software |", "| autodesk",
        "features", "what is", "compare",
    ]
    if any(s in title for s in product_signals):
        return "product"

    ecommerce_signals = [
        "solution", "industry", "construction", "manufacturing",
        "architecture", "engineering", "media", "entertainment",
        "customer stor", "case study",
    ]
    if any(s in title for s in ecommerce_signals):
        return "ecommerce"

    # --- Content-based fallback ---
    if any(s in text_lower for s in ["subscribe now", "free trial", "add to cart", "buy now", "per year", "per month"]):
        return "product"
    if any(s in text_lower for s in ["step 1", "step 2", "procedure", "to create a", "to edit a", "command line"]):
        return "help"
    if any(s in text_lower for s in ["posted on", "published", "min read", "by autodesk", "share this"]):
        return "blog"

    return "unknown"


def _extract_tables_as_markdown(soup: BeautifulSoup) -> list[str]:
    """Convert HTML tables into Markdown format for LLM-readable context."""
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
    """Remove navigation, footer, cookie banners, and other boilerplate."""
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
    """
    Post-process extracted text. Also strips common boilerplate phrases
    that survive HTML extraction.
    """
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)

    # Remove leftover boilerplate lines
    boilerplate_patterns = [
        r"^Skip to .*$",
        r"^Cookie Settings$",
        r"^Accept All Cookies$",
        r"^Sign [Ii]n$",
        r"^Log [Ii]n$",
        r"^Search$",
        r"^Menu$",
        r"^Close$",
        r"^Back to top$",
        r"^Share this.*$",
        r"^Follow us.*$",
        r"^\s*PRODUCTS\s*$",
        r"^\s*SUPPORT\s*$",
        r"^\s*RESOURCES\s*$",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = text.split("\n")
    cleaned_lines = [
        line for line in lines
        if len(line.strip()) > 1 or line.strip() == ""
    ]
    text = "\n".join(cleaned_lines).strip()
    return text


# ─── REPLACEMENT for parse_html_file() in html_parser.py ───────────────────

# v2 changes (Issue 2a — Semantic Recall = 0):

def parse_html_file(filepath: Path) -> Optional[ParsedDocument]:
    """
    Parse a single HTML file using a multi-layer extraction strategy.

    Key design: run BOTH trafilatura (recall mode) AND BeautifulSoup on the
    ORIGINAL DOM, then take the longer result. This ensures competitor names,
    feature comparisons, and secondary content survive into the chunk corpus.
    """
    filepath = Path(filepath)
    logger.debug(f"Parsing: {filepath.name}")

    try:
        raw_html = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Cannot read {filepath.name}: {e}")
        return None

    if len(raw_html) < 200:
        logger.info(f"Skipping {filepath.name}: too small ({len(raw_html)} bytes)")
        return None

    # Keep the original soup intact for multi-layer extraction.
    # BUG FIX: previously a single soup object was boilerplate-stripped before
    # BS4 extraction, making the fallback extraction useless.
    soup_original = BeautifulSoup(raw_html, "lxml")

    # --- Extract metadata and tables from the ORIGINAL, unstripped soup ---
    meta = _extract_metadata_from_soup(soup_original)

    title = ""
    if soup_original.title and soup_original.title.string:
        title = soup_original.title.string.strip()
    elif "og:title" in meta:
        title = meta["og:title"].strip()
    else:
        title = filepath.stem

    # Tables are extracted before stripping so comparison tables survive.
    tables_md = _extract_tables_as_markdown(soup_original)

    # === Multi-layer text extraction ===

    traf_content = ""
    bs4_content = ""

    # Layer 1: trafilatura with favor_recall=True.
    # CHANGED: favor_precision=False prevents trafilatura from dropping secondary
    # content (competitor comparisons, feature sidebars, version tables).
    # Trade-off: slightly more noise in chunks; mitigated by reranker downstream.
    try:
        traf_result = trafilatura.extract(
            raw_html,
            include_tables=True,
            include_links=False,
            include_images=False,
            include_comments=False,
            favor_precision=False,   
            favor_recall=True,   
            output_format="txt",
        )
        if traf_result:
            traf_content = traf_result
            logger.debug(f"  trafilatura (recall): {len(traf_content)} chars")
    except Exception as e:
        logger.debug(f"  trafilatura failed: {e}")

    try:
        import copy
        soup_for_bs4 = _remove_boilerplate(copy.copy(soup_original))
        main_content = (
            soup_for_bs4.find("main")
            or soup_for_bs4.find("article")
            or soup_for_bs4.find("div", class_=re.compile(r"body|content|main", re.I))
            or soup_for_bs4.find("div", id=re.compile(r"body|content|main", re.I))
            or soup_for_bs4.find("body")
        )
        if main_content:
            bs4_content = main_content.get_text(separator="\n", strip=True)
            logger.debug(f"  BS4 (original DOM): {len(bs4_content)} chars")
    except Exception as e:
        logger.debug(f"  BS4 failed: {e}")

    # Take the longer result — competitor names survive in the longer version.
    content = traf_content if len(traf_content) >= len(bs4_content) else bs4_content
    logger.debug(
        f"  Chose {'trafilatura' if len(traf_content) >= len(bs4_content) else 'BS4'}: "
        f"{len(content)} chars"
    )

    # Layer 3: html2text fallback if both layers produced too little.
    if len(content) < MIN_TEXT_LENGTH:
        try:
            h2t = html2text.HTML2Text()
            h2t.ignore_links = True
            h2t.ignore_images = True
            h2t.ignore_emphasis = False
            h2t.body_width = 0
            h2t_result = h2t.handle(raw_html)
            if len(h2t_result) > len(content):
                content = h2t_result
                logger.debug(f"  html2text fallback: {len(content)} chars")
        except Exception as e:
            logger.debug(f"  html2text failed: {e}")

    # --- Clean ---
    content = _clean_text(content)

    # --- Quality gate ---
    if len(content) < MIN_TEXT_LENGTH:
        logger.info(
            f"Skipping {filepath.name}: insufficient content "
            f"({len(content)} chars < {MIN_TEXT_LENGTH})"
        )
        return None

    # --- Detect product and page type using original soup ---
    product = _detect_product_name(soup_original, meta, content)
    page_type = _detect_page_type(soup_original, meta, content)

    source_path = f"data/raw/{filepath.name}"

    doc = ParsedDocument(
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

    logger.info(
        f"Parsed {filepath.name}: {len(content)} chars, "
        f"type={page_type}, product={product}, tables={len(tables_md)}"
    )
    return doc


def parse_directory(dir_path: str | Path) -> list[ParsedDocument]:
    """Parse all HTML files in a directory, with quality filtering."""
    dir_path = Path(dir_path)
    html_files = sorted(dir_path.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files in {dir_path}")

    documents = []
    skipped = 0
    unknown_products = 0
    unknown_types = 0

    for fp in html_files:
        doc = parse_html_file(fp)
        if doc:
            documents.append(doc)
            if not doc.product_name:
                unknown_products += 1
            if doc.page_type == "unknown":
                unknown_types += 1
        else:
            skipped += 1

    logger.info(
        f"Parsing complete: {len(documents)} docs ingested, "
        f"{skipped} skipped, {unknown_products} unknown products, "
        f"{unknown_types} unknown page types"
    )
    return documents
