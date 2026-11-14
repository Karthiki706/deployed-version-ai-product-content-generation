"""
Agentic Writer: drafts product copy, then *chooses* which verification tools to run and
revises its own draft before submitting. If the agent loop fails for any reason, it falls
back to the classic single-shot generator so the pipeline never loses availability.
"""
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.agents.llm_utils import extract_json
from app.agents.tool_loop import run_tool_loop
from app.agents.tools import ToolContext, build_tools
from app.core.config import settings
from app.core.logger import logger
from app.engines.generator import GeminiContentGenerator
from app.orchestrator.graph_state import GraphState

AGENT_PROTOCOL = """

=========================================
AGENT PROTOCOL (you have tools — use them)
=========================================
You are an autonomous product-copy agent. Work in this order:
1. Call get_verified_facts first. Only those facts may appear in the copy. Never mention FORBIDDEN attributes.
2. Draft the full copy internally.
3. Call check_hallucination, check_brand_rules and check_seo with your draft as a JSON string.
4. If any check fails, fix exactly what it reports and re-check only the failing tools.
5. Stop after the checks pass, or after 3 fix rounds, and submit your best draft.
Your FINAL message must be ONLY the raw JSON object matching the required schema — no prose, no markdown fences.
"""


class AgenticGenerationAgent(BaseAgent):
    """Tool-using replacement for the single-shot GenerationAgent (same node name in the graph)."""

    def __init__(self):
        super().__init__("Agentic Writer")

    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        req_logger = state.get("request_logger")
        prior_trace = list(state.get("agent_trace") or [])
        attempt = state.get("retry_count", 0) + 1

        if not settings.agentic_mode:
            return self._classic(state, req_logger, prior_trace)

        try:
            from app.core.gemini_service import GeminiService
            client = GeminiService.get_client()

            ctx = ToolContext(
                product=state["normalized_product"],
                category=state.get("subcategory") or state["category"],
                validation_report=state.get("validation_report") or {},
                accepted_attributes=state.get("accepted_attributes") or [],
                rejected_attributes=state.get("rejected_attributes") or [],
            )
            final_text, trace = run_tool_loop(
                client,
                state["prompt"] + AGENT_PROTOCOL,
                build_tools(ctx),
                req_logger=req_logger,
            )
            extract_json(final_text)  # fail fast so we can fall back; schema check happens downstream

            for entry in trace:
                entry["attempt"] = attempt
                if req_logger:
                    label = f"Tool: {entry['tool']}" if entry["tool"] else "Writer submitted draft"
                    req_logger._record_timeline_event(label, "AGENT_STEP", entry["result_preview"][:160])
            logger.info(f"[AgenticWriter] attempt {attempt}: {sum(1 for t in trace if t['tool'])} tool calls")

            return {
                "generated_content_raw": final_text,
                "agent_trace": prior_trace + trace,
                "prompt_tokens": getattr(req_logger, "prompt_tokens", 0),
                "completion_tokens": getattr(req_logger, "completion_tokens", 0),
                "estimated_cost": getattr(req_logger, "estimated_cost_usd", 0.0),
            }
        except Exception as e:
            logger.warning(f"[AgenticWriter] agent loop failed ({e}); falling back to classic generation.")
            if req_logger:
                req_logger.add_warning(f"Agentic writer failed, used classic generation: {e}")
            return self._classic(state, req_logger, prior_trace, note=f"fallback: {e}")

    def _classic(self, state: GraphState, req_logger: Any, prior_trace: list, note: str = "") -> Dict[str, Any]:
        raw = GeminiContentGenerator().generate(state["prompt"], req_logger=req_logger)
        trace = prior_trace + [{
            "step": 1, "type": "classic_generation", "tool": None, "args": {},
            "result_preview": note or "Agentic mode off: single-shot generation.",
            "duration_seconds": 0.0, "attempt": state.get("retry_count", 0) + 1,
        }]
        return {
            "generated_content_raw": raw,
            "agent_trace": trace,
            "prompt_tokens": getattr(req_logger, "prompt_tokens", 0),
            "completion_tokens": getattr(req_logger, "completion_tokens", 0),
            "estimated_cost": getattr(req_logger, "estimated_cost_usd", 0.0),
        }
