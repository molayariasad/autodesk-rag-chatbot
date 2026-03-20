.PHONY: install ingest serve ui eval analyze test clean docker-up docker-down

# ============================================================
# Autodesk RAG Chatbot — Makefile
# Usage: make <target>
# ============================================================

install:
	pip install uv
	uv sync
	ollama pull gemma3:4b
	@echo "✅ Dependencies installed. Use 'uv run' prefix for all commands."

ingest:
	uv run python scripts/ingest.py --data-dir ./data/raw

serve:
	uv run uvicorn src.api.main:app --port 8000

ui:
	uv run streamlit run ui/app.py

eval:
	# Usage: make eval EXP=05_quality_filter
	# Runs both corpus and blended evaluation for the given experiment name.
	uv run python scripts/run_trulens_eval.py \
	    --mode both \
	    --experiment "$(EXP)" \
	    --pipeline optimized

analyze:
	uv run python scripts/analyze_metrics.py
	uv run python scripts/analyze_data.py --data-dir ./data/raw

test:
	uv run pytest tests/ -v

clean:
	rm -rf data/chroma_db data/processed __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

docker-up:
	docker-compose up --build
	@echo "Streamlit UI: http://localhost:8501"
	@echo "FastAPI docs: http://localhost:8000/docs"

docker-down:
	docker-compose down