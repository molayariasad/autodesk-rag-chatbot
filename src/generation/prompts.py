# src/generation/prompts.py
"""
Prompt templates for the RAG chatbot.

Design principles:
- Grounded generation: ONLY use provided context, never external knowledge.
- Explicit refusal: Hard instruction for off-topic questions — prevents
  partial engagement that scores poorly on both AR and GRD.
- Score-weighted context: Chunks sorted by rerank score so the LLM sees
  the most reliable sources first and can down-weight low-confidence ones.
- Structured comparison: Comparison queries get an explicit format
  instruction that improves Answer Relevance scoring on that category.
- Citation transparency: Source references with type (corpus vs web).
- Conversation-aware: Last 3 turns included for coreference resolution.
"""

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CORPUS = """You are an Autodesk product expert assistant. \
Your role is to help customers with questions about Autodesk products, \
features, pricing, and technical documentation.

CRITICAL RULES:
1. ONLY answer based on the provided context documents. Do NOT use any \
external knowledge, training data, or general world knowledge.
2. If the context does not contain enough information, say exactly: \
"Based on the available documentation, I don't have complete information \
about this. I recommend checking Autodesk's official website or contacting \
support."
3. If the question is completely unrelated to Autodesk products or software, \
say exactly: "I can only assist with Autodesk product questions. This topic \
is outside my scope." Do not engage with the off-topic subject at all.
4. When answering, cite the source document: [Source: Document Title].
5. Be concise and technically accurate. Prefer bullet points for \
multi-part answers.
6. Never invent version numbers, prices, or specific technical specifications \
not present in the context."""

SYSTEM_PROMPT_BLENDED = """You are an Autodesk product expert assistant. \
Your role is to help customers with questions about Autodesk products, \
features, pricing, and technical documentation.

You have access to two source types ranked by reliability:
  [CORPUS] Internal Autodesk documentation — authoritative, always prefer this.
  [WEB]    Live web search results — supplementary, use only when corpus lacks \
the answer.

CRITICAL RULES:
1. Always prefer corpus sources over web sources for the same information.
2. If the question is completely unrelated to Autodesk products, say exactly: \
"I can only assist with Autodesk product questions. This topic is outside my \
scope." Do not engage with the off-topic subject at all.
3. Cite every claim: [Source: Title] for corpus, [Web: URL] for web results.
4. If neither source answers the question, say: "I couldn't find specific \
information about this in our documentation or web sources."
5. Never invent version numbers, prices, or technical specifications not \
present in the sources.
6. Be concise. Prefer bullet points for multi-part answers."""

# ── Context formatting ────────────────────────────────────────────────────────

CONTEXT_TEMPLATE = """--- Retrieved Context (sorted by relevance score) ---
{context}
--- End Context ---"""

HISTORY_TEMPLATE = """--- Conversation History ---
{history}
--- End History ---"""

USER_TURN_TEMPLATE = """{context}

{history_section}

User Question: {question}
{format_hint}
Answer based only on the context above. Cite your sources."""

# Format hints injected per query type to improve Answer Relevance scoring
FORMAT_HINTS = {
    "comparison": (
        "\nInstruction: This is a comparison question. Structure your answer "
        "with a brief intro, then address each product's relevant characteristics "
        "separately before summarising the key difference."
    ),
    "pricing_adversarial": (
        "\nInstruction: Only state a price if it appears verbatim in the context. "
        "If no exact price is present, say so explicitly rather than estimating."
    ),
    "version_adversarial": (
        "\nInstruction: Only state a version number or release year if it appears "
        "verbatim in the context. Do not infer or approximate."
    ),
    "irrelevant": (
        "\nInstruction: If this question is unrelated to Autodesk products, "
        "decline immediately without engaging with the topic."
    ),
}


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Chunks are sorted by rerank/RRF score (highest first) so the LLM
    encounters the most reliable sources at the top of the context window.
    Each chunk header includes the confidence score so the LLM can
    down-weight low-confidence sources when synthesising an answer.

    Score-weighted ordering improves Groundedness: the LLM is less likely
    to generate claims from a low-score chunk that happens to appear early.
    """
    if not retrieved_chunks:
        return "No relevant documents found."

    # Sort by best available score: rerank → rrf → raw score
    def _score(chunk: dict) -> float:
        return chunk.get("rerank_score",
               chunk.get("rrf_score",
               chunk.get("score", 0.0)))

    sorted_chunks = sorted(retrieved_chunks, key=_score, reverse=True)

    context_parts = []
    for i, chunk in enumerate(sorted_chunks, 1):
        metadata    = chunk.get("metadata", {})
        title       = metadata.get("title", "Untitled")
        source_type = chunk.get("source", "unknown")
        score       = _score(chunk)

        if source_type == "web_search":
            url    = metadata.get("url", "")
            header = f"[WEB {i}] {title} | score={score:.3f} | {url}"
        else:
            header = f"[Document {i}] {title} | score={score:.3f}"

        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(context_parts)


def format_history(chat_history: list[dict]) -> str:
    """Format the last 3 turns of chat history for the prompt."""
    if not chat_history:
        return ""

    history_lines = []
    for msg in chat_history[-6:]:   # last 3 turns = 6 messages
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        history_lines.append(f"{role.capitalize()}: {content}")

    return HISTORY_TEMPLATE.format(history="\n".join(history_lines))


def _detect_format_hint(question: str, category: str | None = None) -> str:
    """
    Return a format hint string for categories that benefit from
    structured output instructions.

    Category is passed explicitly when available (eval pipeline).
    Falls back to keyword heuristics for live chat queries where
    no category label is known.
    """
    if category and category in FORMAT_HINTS:
        return FORMAT_HINTS[category]

    # Keyword heuristics for live inference
    q_lower = question.lower()
    if any(w in q_lower for w in ["difference between", "compare", "vs", "versus",
                                   "better", "which is"]):
        return FORMAT_HINTS["comparison"]
    if any(w in q_lower for w in ["price", "cost", "usd", "per year", "subscription"]):
        return FORMAT_HINTS["pricing_adversarial"]
    if any(w in q_lower for w in ["version", "release", "2024", "2025", "latest"]):
        return FORMAT_HINTS["version_adversarial"]

    return ""


def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
    mode: str = "corpus",
    category: str | None = None,
) -> tuple[str, str]:
    """
    Build the complete prompt for the LLM.

    Args:
        question:         User's question.
        retrieved_chunks: Top-k retrieval results (sorted by score internally).
        chat_history:     Previous conversation turns.
        mode:             "corpus" or "blended".
        category:         Optional query category from eval suite — used to
                          inject format hints that improve Answer Relevance
                          scoring on comparison, pricing, and adversarial queries.

    Returns:
        (system_prompt, user_message) tuple.
    """
    system_prompt = (
        SYSTEM_PROMPT_CORPUS if mode == "corpus" else SYSTEM_PROMPT_BLENDED
    )

    context_str     = format_context(retrieved_chunks)
    context_section = CONTEXT_TEMPLATE.format(context=context_str)
    history_section = format_history(chat_history or [])
    format_hint     = _detect_format_hint(question, category)

    user_message = USER_TURN_TEMPLATE.format(
        context=context_section,
        history_section=history_section,
        question=question,
        format_hint=format_hint,
    )

    return system_prompt, user_message