.PHONY: install ingest serve ui eval test clean docker-up docker-down

# ============================================================
# Autodesk RAG Chatbot — Makefile
# ============================================================

install:
	pip install uv
	uv pip install -e ".[dev]"
	pip install pydantic-settings
	ollama pull mistral:7b-instruct-v0.3-q4_K_M

ingest:
	uv run python scripts/ingest.py --data-dir ./data/raw

serve:
	uv run uvicorn src.api.main:app --reload --port 8000

ui:
	uv run streamlit run ui/app.py

eval:
	uv run python scripts/run_eval.py --mode both

analyze:
	uv run python scripts/analyze_data.py --data-dir ./data/raw

test:
	uv run pytest tests/ -v

clean:
	rm -rf data/chroma_db data/processed __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

docker-up:
	docker-compose up --build -d
	@echo "Streamlit UI: http://localhost:8501"
	@echo "FastAPI docs: http://localhost:8000/docs"

docker-down:
	docker-compose down -v
