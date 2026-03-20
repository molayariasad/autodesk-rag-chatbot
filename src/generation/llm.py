"""
LLM interface supporting Ollama (local) models.

Why Ollama + Mistral 7B:
- Fully open-source, runs locally, no data leaves the machine.
- Mistral 7B Instruct has strong instruction-following and RAG performance.
- Ollama provides a simple REST API compatible with many frameworks.
- In production, would use Azure OpenAI or Anthropic API for quality.
"""

"""
LLM interface supporting Ollama (local) models.

v2 changes (Issue 1 — High Latency):
- stream=True with buffered error handling (response.read() before raise_for_status)
- num_gpu=99: forces all layers onto Apple Metal GPU
- num_thread=8: matches M-series performance core count
- num_predict reads from settings.llm_max_tokens
- keep_alive reads from settings (with getattr fallback for older configs)
- warmup() pre-loads model into GPU memory at startup
"""

import json
import httpx
from loguru import logger

from src.config import settings


class LLMClient:
    """Interface to local Ollama LLM with streaming and Metal GPU support."""

    def __init__(self):
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.keep_alive = getattr(settings, "llm_keep_alive", "60m")

    def _build_payload(self, system_prompt: str, user_message: str, temperature: float) -> dict:
        return {
            "model": self.model,
            "keep_alive": self.keep_alive,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": self.max_tokens,
                "num_gpu": 99,
                "num_thread": 8,
            },
        }

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Streams tokens from Ollama and assembles them into a complete string.
        Calls response.read() before raise_for_status() so that error body
        text is accessible without hitting httpx.ResponseNotRead.
        """
        temperature = temperature or self.temperature
        payload = self._build_payload(system_prompt, user_message, temperature)

        try:
            with httpx.Client(timeout=120.0) as client:
                tokens: list[str] = []

                with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    # Buffer the body before raise_for_status() so that
                    # e.response.text is readable in the except block.
                    if response.status_code >= 400:
                        response.read()
                        response.raise_for_status()

                    for raw_line in response.iter_lines():
                        if not raw_line:
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            logger.debug(f"Non-JSON line from Ollama: {raw_line!r}")
                            continue

                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            tokens.append(token)

                        if chunk.get("done", False):
                            break

                answer = "".join(tokens)
                logger.debug(
                    f"LLM generated {len(answer)} chars "
                    f"(model={self.model}, temp={temperature})"
                )
                return answer

        except httpx.ConnectError:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
            return (
                "I'm sorry, the language model service is currently unavailable. "
                "Please ensure Ollama is running with: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text
            logger.error(f"Ollama HTTP error: {status} — {body}")
            if status == 404:
                return (
                    f"Model endpoint not found (HTTP 404). "
                    f"Check that '{self.model}' is installed "
                    f"(`ollama pull {self.model}`) and that your Ollama version "
                    f"supports /api/chat (`ollama --version`, "
                    f"update with `brew upgrade ollama`)."
                )
            return (
                f"The language model returned an error (HTTP {status}). "
                f"Details: {body[:200]}"
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"An error occurred while generating the response: {str(e)}"

    def warmup(self) -> bool:
        """
        Pre-load the model into GPU memory by sending a trivial request.
        Returns True if warmup succeeded, False otherwise.
        """
        logger.info(f"Warming up LLM model '{self.model}' into GPU memory...")
        try:
            response = self.generate(
                system_prompt="You are a helpful assistant.",
                user_message="Reply with the single word: ready",
            )
            # Check for any known error signal phrases returned by generate()
            # when Ollama returns a non-2xx response.
            error_signals = [
                "unavailable",
                "not found",
                "http 404",
                "http 5",
                "an error occurred",
            ]
            success = bool(response) and not any(
                signal in response.lower() for signal in error_signals
            )
            if success:
                logger.info(f"LLM warmup complete. Model '{self.model}' is loaded.")
            else:
                logger.warning(
                    f"LLM warmup failed — model may not be installed. "
                    f"Run: ollama pull {self.model}\n"
                    f"Response was: {response!r}"
                )
            return success
        except Exception as e:
            logger.warning(f"LLM warmup failed (non-fatal): {e}")
            return False

    def is_available(self) -> bool:
        """Check if the Ollama service is reachable."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                return [m["name"] for m in models]
        except Exception:
            return []