"""Small helpers shared by the LLM-driven agents (planner, supervisor advisor, tool loop)."""
import concurrent.futures
import json
import re
from typing import Any, Callable, Dict, List, Optional

from google.genai import types

from app.core.config import settings
from app.core.logger import logger


def model_chain() -> List[str]:
    """Primary model followed by configured fallbacks (no duplicates)."""
    chain = [settings.gemini_model]
    for m in settings.fallback_models:
        if m not in chain:
            chain.append(m)
    return chain


def call_with_timeout(fn: Callable, timeout: float):
    """Run a blocking SDK call with a hard timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn).result(timeout=timeout)


def extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of model text (handles markdown fences and stray prose)."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return json.loads(cleaned[start:end + 1])


def record_usage(req_logger: Any, response: Any) -> None:
    """Add token usage from a Gemini response to the request logger, if present."""
    usage = getattr(response, "usage_metadata", None)
    if req_logger is None or usage is None:
        return
    try:
        req_logger.record_token_usage(
            int(getattr(usage, "prompt_token_count", 0) or 0),
            int(getattr(usage, "candidates_token_count", 0) or 0),
        )
    except Exception as e:  # observability must never break the pipeline
        logger.debug(f"Token usage recording skipped: {e}")


def generate_json(client: Any, prompt: str, req_logger: Any = None, temperature: float = 0.2) -> Dict[str, Any]:
    """One structured-JSON LLM call. One attempt per model in the fallback chain, short timeout.

    Raises the last error if every model fails, so callers can degrade to deterministic logic.
    """
    last_err: Optional[Exception] = None
    for model_id in model_chain():
        try:
            resp = call_with_timeout(
                lambda m=model_id: client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                    ),
                ),
                settings.agent_step_timeout,
            )
            record_usage(req_logger, resp)
            return extract_json(resp.text)
        except Exception as e:
            last_err = e
            logger.warning(f"[llm_utils] model '{model_id}' failed: {e}")
    raise last_err or RuntimeError("No models available")
