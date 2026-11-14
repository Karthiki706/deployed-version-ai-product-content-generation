import json
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from app.agents.base import BaseAgent
from app.agents.llm_utils import generate_json
from app.core.config import settings
from app.core.logger import logger
from app.engines.planner import PlannerEngine
from app.orchestrator.graph_state import GraphState

PLAN_KEYS = [
    "writing_strategy", "selected_brand_style", "seo_keyword_strategy",
    "bullet_style", "content_structure",
]


class PlannerAgent(BaseAgent):
    """LLM planner: reads the product and decides the copywriting strategy.

    On retries it replans using the audit weaknesses and the supervisor's directive instead of
    just appending a string. Falls back to the deterministic PlannerEngine on any failure.
    """

    def __init__(self):
        super().__init__("Planner Agent")

    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        product = state["normalized_product"]
        retry_count = state.get("retry_count", 0)

        # Deterministic baseline: also provides the fixed fields and the safety net.
        plan = PlannerEngine.plan(product)
        plan["planner_source"] = "deterministic"
        plan["planner_risks"] = []

        if settings.agentic_mode and settings.llm_planner_enabled:
            try:
                plan.update(self._llm_plan(state, product, plan))
                plan["planner_source"] = "llm"
            except Exception as e:
                logger.warning(f"[Planner] LLM planning failed ({e}); using deterministic plan.")
                req_logger = state.get("request_logger")
                if req_logger:
                    req_logger.add_warning(f"LLM planner failed, used deterministic plan: {e}")

        if retry_count > 0 and plan["planner_source"] == "deterministic":
            plan["writing_strategy"] += f" Focus heavily on correcting previous issues (Attempt #{retry_count + 1})."
        return plan

    def _llm_plan(self, state: GraphState, product, baseline: Dict[str, Any]) -> Dict[str, Any]:
        from app.core.gemini_service import GeminiService
        client = GeminiService.get_client()

        context = {
            "product_name": product.product_name,
            "category": state.get("subcategory") or state.get("category"),
            "brand": product.brand,
            "material": product.material,
            "features": state.get("accepted_attributes") or product.features,
            "rejected_attributes_do_not_use": state.get("rejected_attributes", []),
            "target_audience": product.target_audience,
            "seo_keywords": product.seo_keywords,
            "brand_voice": product.brand_voice,
            "tone": product.tone,
            "target_words": product.description_length,
        }
        retry_block = ""
        if state.get("retry_count", 0) > 0:
            retry_block = (
                "\nThis is a RETRY. Previous audit weaknesses:\n"
                + json.dumps(state.get("weaknesses", []), indent=1)
                + f"\nSupervisor directive: {state.get('supervisor_focus', 'none')}\n"
                "Change the strategy so these cannot recur."
            )

        prompt = (
            "You are the planning agent for an e-commerce copywriting pipeline. "
            "Decide the writing plan for this product. Do NOT write the copy.\n"
            f"Product context:\n{json.dumps(context, indent=1)}\n{retry_block}\n"
            "Respond with JSON only, with these keys:\n"
            '- "writing_strategy": 1-2 sentences on angle and emphasis\n'
            '- "selected_brand_style": short description of voice\n'
            '- "seo_keyword_strategy": where each target keyword should appear\n'
            '- "bullet_style": how bullets should be written\n'
            '- "content_structure": ordered sections, joined with " -> "\n'
            '- "planner_risks": list of up to 4 specific hallucination or compliance risks for THIS product '
            "(e.g. claims that are tempting but unsupported)\n"
        )
        data = generate_json(client, prompt, state.get("request_logger"))
        out = {k: str(data[k]) for k in PLAN_KEYS if data.get(k)}
        if not out:
            raise ValueError("Planner returned no usable fields")
        risks = data.get("planner_risks") or []
        out["planner_risks"] = [str(r) for r in risks][:4] if isinstance(risks, list) else []
        return out
