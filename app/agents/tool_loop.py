"""
Gemini function-calling loop (ReAct-style) with a full trace.

The model decides which tool to call and when to stop. We run the loop manually
(automatic function calling disabled) so that every step is observable, capped,
and timed. The loop is client-injectable, which makes it unit-testable offline.
"""
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.genai import types

from app.agents.llm_utils import call_with_timeout, model_chain, record_usage
from app.agents.tools import summarize_result
from app.core.config import settings
from app.core.logger import logger


class ToolLoopError(Exception):
    """Raised when the loop cannot produce a final answer."""


def run_tool_loop(
    client: Any,
    prompt: str,
    tools: List[Callable],
    req_logger: Any = None,
    max_steps: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the agent until it answers without calling a tool.

    Returns (final_text, trace). Trace entries:
        {"step": 1, "type": "tool_call", "tool": "check_seo", "args": {...},
         "result_preview": "...", "duration_seconds": 0.01}
        {"step": 4, "type": "final", "tool": None, ...}
    """
    max_steps = max_steps or settings.agent_max_tool_steps
    registry = {fn.__name__: fn for fn in tools}
    temp = settings.gemini_temperature if temperature is None else temperature

    last_err: Optional[Exception] = None
    for model_id in model_chain():
        trace: List[Dict[str, Any]] = []
        try:
            config = types.GenerateContentConfig(
                temperature=temp,
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            contents: List[Any] = [types.Content(role="user", parts=[types.Part(text=prompt)])]

            for step in range(1, max_steps + 1):
                is_last_allowed = step == max_steps
                if is_last_allowed:
                    # Out of budget: force a final answer instead of another tool call.
                    contents.append(types.Content(role="user", parts=[types.Part(
                        text="Tool budget exhausted. Return your best final JSON now, with no further tool calls."
                    )]))

                t0 = time.time()
                response = call_with_timeout(
                    lambda: client.models.generate_content(model=model_id, contents=contents, config=config),
                    settings.agent_step_timeout,
                )
                record_usage(req_logger, response)
                if req_logger is not None:
                    req_logger.gemini_duration += time.time() - t0

                candidate = response.candidates[0]
                parts = list(candidate.content.parts or [])
                calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

                if not calls or is_last_allowed:
                    text = getattr(response, "text", None) or "".join(
                        p.text for p in parts if getattr(p, "text", None)
                    )
                    if not text:
                        raise ToolLoopError("Model returned no final text.")
                    trace.append({"step": step, "type": "final", "tool": None,
                                  "args": {}, "result_preview": "Final draft submitted.",
                                  "duration_seconds": round(time.time() - t0, 3)})
                    return text, trace

                # Keep the model turn verbatim (preserves thought signatures), then answer each call.
                contents.append(candidate.content)
                reply_parts = []
                for call in calls:
                    name = call.name
                    args = dict(call.args or {})
                    c0 = time.time()
                    try:
                        if name not in registry:
                            raise KeyError(f"Unknown tool '{name}'")
                        result = registry[name](**args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                        logger.warning(f"[ToolLoop] tool '{name}' failed: {e}")
                    trace.append({
                        "step": step, "type": "tool_call", "tool": name,
                        "args": {k: (str(v)[:120] + "…" if len(str(v)) > 120 else v) for k, v in args.items()},
                        "result_preview": summarize_result(result),
                        "duration_seconds": round(time.time() - c0, 4),
                    })
                    reply_parts.append(types.Part.from_function_response(name=name, response={"result": result}))
                contents.append(types.Content(role="user", parts=reply_parts))

            raise ToolLoopError("Loop ended without a final answer.")
        except Exception as e:
            last_err = e
            logger.warning(f"[ToolLoop] model '{model_id}' failed: {e}")

    raise ToolLoopError(f"All models failed in tool loop: {last_err}")
