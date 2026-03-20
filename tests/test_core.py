# tests/test_core.py
"""
Unit tests for the Autodesk RAG Chatbot.

Tests core functionality without requiring Ollama or the full pipeline.
Run: uv run pytest tests/ -v
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os


# ============================================================
# Test: HTML Parser
# ============================================================

class TestHTMLParser:
    """Tests for the multi-layer HTML parsing pipeline."""

    def test_parse_empty_file(self, tmp_path):
        """Blank/empty files should return None."""
        from src.ingestion.html_parser import parse_html_file

        empty_file = tmp_path / "empty.html"
        empty_file.write_text("")
        assert parse_html_file(empty_file) is None

    def test_parse_js_only_file(self, tmp_path):
        """JS-only shell pages should return None (< MIN_TEXT_LENGTH)."""
        from src.ingestion.html_parser import parse_html_file

        js_shell = tmp_path / "shell.html"
        js_shell.write_text("""<!doctype html><html><head>
        <title>Help</title></head><body>
        <script>Boot.init(["App"]).then(function(){UIComponent.create("App")})</script>
        </body></html>""")
        result = parse_html_file(js_shell)
        assert result is None

    def test_parse_valid_help_page(self, tmp_path):
        """Valid help documentation pages should be parsed."""
        from src.ingestion.html_parser import parse_html_file

        help_page = tmp_path / "help.html"
        help_page.write_text("""<!DOCTYPE html><html><head>
        <meta name="product" content="AutoCAD">
        <meta name="topic-type" content="concept">
        <title>About Drawing Lines in AutoCAD</title></head>
        <body><div class="body">
        <h1>About Drawing Lines in AutoCAD</h1>
        <p>AutoCAD provides multiple tools for drawing lines. The LINE command
        is the most basic tool. You can specify start and end points to create
        line segments. AutoCAD supports both 2D and 3D line creation with
        precise coordinate input. The POLYLINE command creates connected line
        segments as a single object. Use MLINE for multilines with parallel
        elements and specified widths.</p>
        </div></body></html>""")

        result = parse_html_file(help_page)
        assert result is not None
        assert "AutoCAD" in result.content
        assert result.product_name == "AutoCAD"
        assert result.page_type == "help"
        assert result.title == "About Drawing Lines in AutoCAD"

    def test_parse_product_page(self, tmp_path):
        """Product/marketing pages should extract content and detect product."""
        from src.ingestion.html_parser import parse_html_file

        product_page = tmp_path / "maya.html"
        product_page.write_text("""<!DOCTYPE html><html><head>
        <title>Autodesk Maya Creative Software | Get Prices</title></head>
        <body><main>
        <h1>What is Autodesk Maya Creative?</h1>
        <p>Maya Creative software includes sophisticated animation, modeling,
        and rendering tools to bring your vision to life. Create lifelike
        animations with intuitive tools. Build detailed 3D models with precision.
        Render high-quality images in fewer clicks with Arnold renderer.
        Maya Creative is available with Flex tokens for flexible access to the
        full suite of 3D tools. Animators, modelers, and technical directors
        rely on Maya for film, television, and game production worldwide.
        The software supports Python and MEL scripting for custom workflows.
        Rigging tools allow complex character setups with blend shapes and
        deformers. Dynamics and simulation tools handle cloth, fluids, and
        particle effects for realistic visual effects production.</p>
        </main></body></html>""")

        result = parse_html_file(product_page)
        assert result is not None
        assert "Maya" in result.content or "animation" in result.content
        assert result.product_name in ["Maya", "Maya Creative"]

    def test_table_extraction(self, tmp_path):
        """HTML tables should be converted to Markdown."""
        from src.ingestion.html_parser import parse_html_file

        table_page = tmp_path / "table.html"
        table_page.write_text("""<!DOCTYPE html><html><head>
        <title>Product Comparison</title></head>
        <body><div class="content">
        <p>Compare Autodesk products below. AutoCAD is great for 2D drafting
        while Revit is designed for BIM workflows. Both products support
        collaboration and cloud features.</p>
        <table><tr><th>Feature</th><th>AutoCAD</th><th>Revit</th></tr>
        <tr><td>2D Drafting</td><td>Yes</td><td>Limited</td></tr>
        <tr><td>BIM</td><td>No</td><td>Yes</td></tr>
        <tr><td>3D Modeling</td><td>Yes</td><td>Yes</td></tr></table>
        </div></body></html>""")

        result = parse_html_file(table_page)
        assert result is not None
        assert len(result.tables_markdown) >= 1

    def test_parse_directory(self, tmp_path):
        """parse_directory should process all HTML files in a folder."""
        from src.ingestion.html_parser import parse_directory

        (tmp_path / "valid.html").write_text("""<!DOCTYPE html><html><head>
        <title>Valid Page</title></head><body>
        <p>This is a valid page about Fusion 360 features including 3D modeling,
        CAM manufacturing, and simulation capabilities for modern design and
        engineering workflows in the cloud. Fusion 360 supports collaboration
        between engineers and designers in real time. The software includes
        generative design tools that use AI to explore design options. CAM
        toolpaths can be generated directly from 3D models without file export.
        Simulation tools validate structural integrity before manufacturing.
        Fusion 360 runs on Mac and Windows with cloud storage for all files.
        Teams can share designs and track version history automatically.
        The integrated electronics workspace supports PCB design alongside
        mechanical components for complete product development workflows.</p>
        </body></html>""")
        (tmp_path / "empty.html").write_text("")

        results = parse_directory(tmp_path)
        assert len(results) >= 1
        assert all(r.char_count >= 100 for r in results)

    def test_product_alias_normalization(self, tmp_path):
        """Internal product codes should be normalized to canonical names."""
        from src.ingestion.html_parser import _normalize_product_name

        assert _normalize_product_name("F360")    == "Fusion 360"
        assert _normalize_product_name("ACDIST")  == "AutoCAD"
        assert _normalize_product_name("RVIT")    == "Revit"
        assert _normalize_product_name("ACDLTG")  == "AutoCAD LT"
        # Already canonical names should pass through unchanged
        assert _normalize_product_name("Maya")    == "Maya"


# ============================================================
# Test: Chunker
# ============================================================

class TestChunker:
    """Tests for the optimized document chunking pipeline."""

    def _make_doc(self, content: str, product: str = "AutoCAD",
                  page_type: str = "help", title: str = "Test",
                  tables: list | None = None):
        """Helper — build a ParsedDocument with all required fields."""
        from src.ingestion.html_parser import ParsedDocument
        return ParsedDocument(
            source_file="test.html",
            source_path="data/raw/test.html",   # required in v2
            title=title,
            content=content,
            tables_markdown=tables or [],
            product_name=product,
            page_type=page_type,
            char_count=len(content),
        )

    def test_chunk_document_produces_chunks(self):
        """A parsed document should produce at least one chunk."""
        from src.ingestion.chunker import chunk_document

        doc = self._make_doc("A " * 300)
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
        assert all(c.metadata["source_file"] == "test.html" for c in chunks)
        assert all(c.metadata["product_name"] == "AutoCAD" for c in chunks)

    def test_chunk_includes_metadata_prefix(self):
        """Each chunk should have a metadata prefix for LLM context."""
        from src.ingestion.chunker import chunk_document

        doc = self._make_doc("Maya is a 3D animation software. " * 20,
                              product="Maya", page_type="product",
                              title="Maya Features")
        chunks = chunk_document(doc)
        assert any("Product: Maya" in c.text for c in chunks)

    def test_small_document_kept_whole(self):
        """
        Documents that fit within the chunk budget should not be split.
        Optimized chunker keeps small docs as a single 'full' chunk.
        """
        from src.ingestion.chunker import chunk_document

        # 200 chars — well under the 1200 chunk_size
        doc = self._make_doc("AutoCAD is a 2D drafting tool. " * 6)
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_type"] == "full"

    def test_tables_merged_into_main_content(self):
        """
        Optimized chunker merges tables into the main text (not separate chunks).
        This keeps comparison tables with their surrounding context.
        """
        from src.ingestion.chunker import chunk_document

        doc = self._make_doc(
            content="Main content about AutoCAD. " * 5,
            tables=["| Feature | Value |\n|---|---|\n| 2D | Yes |"],
        )
        chunks = chunk_document(doc)

        # No chunk should have type "table" — tables are merged into text/full
        table_chunks = [c for c in chunks if c.metadata["chunk_type"] == "table"]
        assert len(table_chunks) == 0, (
            "Optimized chunker should merge tables into main content, "
            "not produce separate table chunks"
        )

        # Table content should appear in the main chunk text
        all_text = " ".join(c.text for c in chunks)
        assert "Feature" in all_text or "Value" in all_text

    def test_chunk_id_format(self):
        """Chunk IDs should follow the expected naming convention."""
        from src.ingestion.chunker import chunk_document

        doc = self._make_doc("AutoCAD is a tool. " * 5)
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.chunk_id.startswith("test.html::")


# ============================================================
# Test: BM25 Retriever
# ============================================================

class TestBM25:
    """Tests for BM25 keyword search with bigram tokenizer."""

    def test_bm25_build_and_query(self):
        """BM25 should find relevant chunks by keyword."""
        from src.ingestion.chunker import Chunk
        from src.retrieval.bm25_retriever import BM25Index

        chunks = [
            Chunk("1", "AutoCAD LT is a 2D drafting software",
                  {"source_file": "a.html"}),
            Chunk("2", "Maya is used for 3D animation and rendering",
                  {"source_file": "b.html"}),
            Chunk("3", "Revit is a BIM tool for architecture",
                  {"source_file": "c.html"}),
        ]

        index = BM25Index()
        index.build_index(chunks)

        results = index.query("AutoCAD LT 2D drafting", top_k=2)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "1"

    def test_bm25_empty_query(self):
        """Empty query should return empty results."""
        from src.ingestion.chunker import Chunk
        from src.retrieval.bm25_retriever import BM25Index

        chunks = [Chunk("1", "test content", {"source_file": "a.html"})]
        index = BM25Index()
        index.build_index(chunks)

        results = index.query("", top_k=5)
        assert results == []

    def test_bm25_compound_product_name(self):
        """
        Bigram tokenizer should match compound product names as units.
        This was the BM25 recall=0 bug fix — unigram tokenizer made
        'AutoCAD LT' unsearchable as a compound term.
        """
        from src.ingestion.chunker import Chunk
        from src.retrieval.bm25_retriever import BM25Index

        # Use a realistic corpus size — BM25 IDF scoring behaves oddly
        # with fewer than ~5 documents (all terms appear in 50%+ of docs,
        # driving IDF toward zero and producing empty results).
        chunks = [
            Chunk("1", "AutoCAD LT is for 2D drafting only",
                {"source_file": "a.html"}),
            Chunk("2", "AutoCAD full version supports 3D modeling",
                {"source_file": "b.html"}),
            Chunk("3", "Revit is a BIM tool for architecture and construction",
                {"source_file": "c.html"}),
            Chunk("4", "Maya is used for 3D animation and visual effects",
                {"source_file": "d.html"}),
            Chunk("5", "Fusion 360 combines CAD CAM and CAE in one platform",
                {"source_file": "e.html"}),
        ]
        index = BM25Index()
        index.build_index(chunks)

        # "AutoCAD LT" as a compound should rank chunk 1 higher than chunk 2
        results = index.query("AutoCAD LT", top_k=2)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "1"


# ============================================================
# Test: Prompts
# ============================================================

class TestPrompts:
    """Tests for prompt construction."""

    def test_format_context(self):
        """Context formatting should include document numbers and titles."""
        from src.generation.prompts import format_context

        chunks = [
            {"text": "AutoCAD does 2D drafting",
             "metadata": {"title": "AutoCAD Help", "source_file": "a.html"},
             "source": "vector"},
            {"text": "Maya does animation",
             "metadata": {"title": "Maya Help", "source_file": "b.html"},
             "source": "bm25"},
        ]

        context = format_context(chunks)
        assert "[Document 1]" in context
        assert "[Document 2]" in context
        assert "AutoCAD Help" in context

    def test_build_prompt_corpus_mode(self):
        """Corpus mode should use the grounded system prompt."""
        from src.generation.prompts import build_prompt

        system, user = build_prompt(
            question="What does Maya do?",
            retrieved_chunks=[{
                "text": "Maya is for animation",
                "metadata": {"title": "Maya"},
                "source": "vector",
            }],
            mode="corpus",
        )

        assert "ONLY answer based on the provided context" in system
        assert "What does Maya do?" in user

    def test_build_prompt_with_history(self):
        """Conversation history should be included in the user message."""
        from src.generation.prompts import build_prompt

        history = [
            {"role": "user",      "content": "Tell me about Maya"},
            {"role": "assistant", "content": "Maya is a 3D animation tool."},
        ]

        system, user = build_prompt(
            question="What about its pricing?",
            retrieved_chunks=[{
                "text": "Maya costs $200/yr",
                "metadata": {"title": "Pricing"},
                "source": "vector",
            }],
            chat_history=history,
            mode="corpus",
        )

        assert "Conversation History" in user
        assert "Tell me about Maya" in user


# ============================================================
# Test: Evaluation framework
# ============================================================

class TestEvaluation:
    """
    Tests for the TruLens RAG Triad evaluation framework.

    These tests cover the supporting utilities (question loader, score
    unpacking, CoT extraction) without invoking the judge LLM, which
    requires Ollama and is tested in integration runs via
    scripts/run_trulens_eval.py.
    """

    def test_load_eval_questions_file_not_found(self, tmp_path):
        """load_eval_questions should raise FileNotFoundError for missing files."""
        from src.evaluation.evaluator import load_eval_questions

        with pytest.raises(FileNotFoundError):
            load_eval_questions(path=tmp_path / "nonexistent.json")

    def test_load_eval_questions_valid(self, tmp_path):
        """load_eval_questions should parse a valid JSON question file."""
        from src.evaluation.evaluator import load_eval_questions

        q_file = tmp_path / "questions.json"
        q_file.write_text('[{"question": "What does AutoCAD do?", '
                          '"category": "product_info", "expected_product": "AutoCAD"}]')

        questions = load_eval_questions(path=q_file)
        assert len(questions) == 1
        assert questions[0]["question"] == "What does AutoCAD do?"
        assert questions[0]["category"] == "product_info"

    def test_unpack_score_float(self):
        """Plain float should unpack to (float, None)."""
        from src.evaluation.evaluator import _unpack_score

        score, reason = _unpack_score(0.85)
        assert score == 0.85
        assert reason is None

    def test_unpack_score_tuple(self):
        """CoT tuple (float, str) should unpack to both components."""
        from src.evaluation.evaluator import _unpack_score

        score, reason = _unpack_score((0.9, "The answer is grounded because..."))
        assert score == 0.9
        assert "grounded" in reason

    def test_unpack_score_dict(self):
        """Dict format should unpack score and optional reason."""
        from src.evaluation.evaluator import _unpack_score

        score, reason = _unpack_score({"score": 0.75, "reason": "Partially supported."})
        assert score == 0.75
        assert reason == "Partially supported."

    def test_unpack_score_dict_no_reason(self):
        """Dict without reason key should return None for reason."""
        from src.evaluation.evaluator import _unpack_score

        score, reason = _unpack_score({"score": 0.6})
        assert score == 0.6
        assert reason is None