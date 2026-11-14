import json
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from app.core.config import settings
from app.agents.base import BaseAgent
from app.orchestrator.graph_state import GraphState
from app.agents.llm_utils import generate_json
from app.core.logger import logger

class SupervisorAgent(BaseAgent):
    """The central decision-making Supervisor Agent governing loops and best attempts."""
    
    def __init__(self):
        super().__init__("Supervisor Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        json_error = state.get("json_error")
        timeline = state.get("timeline", [])
        decision_log = state.get("decision_log") or []
        # Create a copy of decision log to avoid mutating reference directly
        decision_log = list(decision_log)
        retry_count = state.get("retry_count", 0)
        req_logger = state.get("request_logger")
        
        # 1. Handle JSON syntax parse failures
        if json_error:
            decision_log.append(f"Attempt {retry_count + 1}: JSON Validation failed: {json_error}")
            
            # Check pattern of successive errors
            attempts_history = state.get("attempts_history") or []
            attempts_history = list(attempts_history)
            attempts_history.append({"type": "JSON_ERROR", "error": json_error, "retry_count": retry_count})
            
            if retry_count + 1 < settings.max_retries:
                next_step = "CORRECT"
                retry_reason = f"JSON Validation failed: {json_error}"
                new_prompt = (
                    f"{state['generation_prompt']}\n\nCRITICAL FIX REQUIRED: "
                    f"Your previous output was not valid JSON: {json_error}. "
                    "Return strictly valid JSON."
                )
                
                if req_logger:
                    req_logger.increment_retries(reason=retry_reason)
                    req_logger.correction_loop_count += 1
                    
                return {
                    "next_step": next_step,
                    "retry_count": retry_count + 1,
                    "retry_reason": retry_reason,
                    "prompt": new_prompt,
                    "decision_log": decision_log,
                    "attempts_history": attempts_history,
                }
            else:
                decision_log.append(f"Attempt {retry_count + 1}: Retries exhausted on JSON Error.")
                return {
                    "next_step": "END",
                    "decision_log": decision_log,
                    "attempts_history": attempts_history,
                }

        # 2. Evaluate scores if parsed successfully
        overall_quality = state.get("overall_quality", 0.0)
        safety_score = state.get("safety_score", 100.0)
        
        q_model = state.get("quality_report")
        hallucination_score = q_model.hallucination_score if q_model else 100.0
        
        h_model = state["hallucination_report"]
        s_model = state["seo_report"]
        b_model = state["brand_report"]
        
        h_rep = h_model.model_dump() if h_model else {}
        s_rep = s_model.model_dump() if s_model else {}
        b_rep = b_model.model_dump() if b_model else {}
        
        section_d = h_rep.get("section_d") or {}
        ai_claim_count = section_d.get("ai_hallucination_count", len(h_rep.get("hallucinated_fields", [])))
        
        has_hallucination = hallucination_score < settings.hallucination_threshold or ai_claim_count > 0 or bool(h_rep.get("hallucinated_fields"))
        has_missing_keywords = bool(s_rep.get("missing_keywords"))
        has_brand_restricted = bool(b_rep.get("violations"))
        
        need_regeneration = has_hallucination or has_missing_keywords or has_brand_restricted
        
        # 3. Track best attempt (highest overall quality score)
        current_attempt_data = {
            "generated_content": state["generated_content"],
            "hallucination_report": state["hallucination_report"],
            "seo_report": state["seo_report"],
            "brand_report": state["brand_report"],
            "quality_report": state["quality_report"],
            "overall_quality": overall_quality,
            "safety_score": safety_score,
            "comparison_report": state.get("comparison_report", []),
            "risk_level": state.get("risk_level", "Low"),
            "generation_confidence": state.get("generation_confidence", 1.0),
            "weaknesses": list(state.get("weaknesses", [])),
        }
        
        attempts_history = state.get("attempts_history") or []
        attempts_history = list(attempts_history)
        attempts_history.append(current_attempt_data)
        
        best_quality_score = state.get("best_quality_score", 0.0)
        best_attempt = state.get("best_attempt")
        
        if best_attempt is None or overall_quality > best_quality_score:
            best_quality_score = overall_quality
            best_attempt = current_attempt_data
            
        decision_log.append(
            f"Attempt {retry_count + 1}: Quality Score = {overall_quality:.2f} | "
            f"Safety Score = {safety_score:.2f} | Needs Correction = {need_regeneration}"
        )
        
        # 4. Check if gates are met
        if not need_regeneration:
            decision_log.append(f"Attempt {retry_count + 1}: Quality criteria satisfied. Terminating loop.")
            return {
                "next_step": "END",
                "best_attempt": best_attempt,
                "best_quality_score": best_quality_score,
                "decision_log": decision_log,
                "manual_review_required": False,
                "attempts_history": attempts_history,
            }
            
        # 5. Handle failed quality checks and schedule correction prompt
        weaknesses = state.get("weaknesses", [])
        retry_reason_str = state.get("reflection_summary", "Audit violations detected.")
        
        # Identify persistent failure patterns (e.g., if a weakness is repeated)
        if retry_count > 0:
            previous_errors = attempts_history[-2].get("weaknesses", []) if len(attempts_history) > 1 else []
            overlapping_errors = [w for w in weaknesses if w in previous_errors]
            if overlapping_errors:
                decision_log.append(f"Attempt {retry_count + 1}: Persistent failure patterns detected: {overlapping_errors}")
                retry_reason_str += f"\n⛔ PERSISTENT FAILURES (Focus heavily on correcting these): {overlapping_errors}"

        if retry_count + 1 < settings.max_retries:
            # LLM advisor chooses the STRATEGY; deterministic gates above still decide pass/fail.
            action, focus, reason = self._advise(
                state, weaknesses, attempts_history, decision_log, retry_count
            )
            if action == "escalate":
                decision_log.append(
                    f"Attempt {retry_count + 1}: Supervisor (LLM) escalated to human review: {reason}"
                )
                return self._finish_with_best(
                    best_attempt, best_quality_score, decision_log, attempts_history,
                    extra={"supervisor_action": "escalate", "supervisor_focus": focus},
                )

            decision_log.append(
                f"Attempt {retry_count + 1}: Supervisor (LLM) -> regenerate | focus: {focus or 'n/a'} | {reason}"
            )
            if focus:
                retry_reason_str += f"\n🎯 SUPERVISOR FOCUS: {focus}"

            if req_logger:
                req_logger.increment_retries(reason=retry_reason_str)
                req_logger.correction_loop_count += 1
                req_logger._record_timeline_event("Supervisor decision", "AGENT_STEP", f"{action}: {focus or reason}"[:160])

            return {
                "next_step": "CORRECT",
                "retry_count": retry_count + 1,
                "retry_reason": retry_reason_str,
                "best_attempt": best_attempt,
                "best_quality_score": best_quality_score,
                "decision_log": decision_log,
                "attempts_history": attempts_history,
                "supervisor_action": action,
                "supervisor_focus": focus,
            }
        else:
            decision_log.append(f"Attempt {retry_count + 1}: Retries exhausted. Reverting to highest-quality attempt.")
            return self._finish_with_best(best_attempt, best_quality_score, decision_log, attempts_history)

    # ------------------------------------------------------------------
    @staticmethod
    def _finish_with_best(best_attempt, best_quality_score, decision_log, attempts_history, extra=None) -> Dict[str, Any]:
        """Restore the highest-quality attempt and end the run flagged for manual review."""
        out = {
            "next_step": "END",
            "generated_content": best_attempt["generated_content"],
            "hallucination_report": best_attempt["hallucination_report"],
            "seo_report": best_attempt["seo_report"],
            "brand_report": best_attempt["brand_report"],
            "quality_report": best_attempt["quality_report"],
            "overall_quality": best_attempt["overall_quality"],
            "safety_score": best_attempt["safety_score"],
            "comparison_report": best_attempt["comparison_report"],
            "risk_level": best_attempt["risk_level"],
            "generation_confidence": best_attempt["generation_confidence"],
            "best_attempt": best_attempt,
            "best_quality_score": best_quality_score,
            "decision_log": decision_log,
            "manual_review_required": True,
            "attempts_history": attempts_history,
        }
        if extra:
            out.update(extra)
        return out

    def _advise(self, state, weaknesses, attempts_history, decision_log, retry_count):
        """Ask the LLM how to proceed. Guardrails keep it safe; any failure => plain regenerate.

        Allowed actions: "regenerate" (with a focused directive) or "escalate" (human review).
        The LLM can never accept a failing draft, and may escalate only after >= 1 correction try.
        """
        default = ("regenerate", "", "Deterministic default (advisor unavailable).")
        if not (settings.agentic_mode and settings.llm_supervisor_enabled):
            return default
        try:
            from app.core.gemini_service import GeminiService
            client = GeminiService.get_client()
            history = [
                {"attempt": i + 1, "overall_quality": a.get("overall_quality"),
                 "weaknesses": a.get("weaknesses", [])[:5]}
                for i, a in enumerate(attempts_history) if isinstance(a, dict) and "overall_quality" in a
            ]
            prompt = (
                "You supervise an AI product-copy pipeline. The latest draft failed compliance audits.\n"
                f"Current weaknesses: {json.dumps(weaknesses[:8])}\n"
                f"Attempt history: {json.dumps(history)}\n"
                f"Retries used: {retry_count + 1} of {settings.max_retries}.\n"
                "Choose ONE action:\n"
                '- "regenerate": try again; give a short, specific "focus" directive (max 40 words) that '
                "targets the root cause, not a restatement of the errors.\n"
                '- "escalate": the same failure keeps recurring and more retries are unlikely to help; '
                "send to a human reviewer.\n"
                'Respond with JSON only: {"action": "...", "focus": "...", "reason": "<=20 words"}'
            )
            data = generate_json(client, prompt, state.get("request_logger"))
            action = str(data.get("action", "regenerate")).lower().strip()
            focus = str(data.get("focus", ""))[:400]
            reason = str(data.get("reason", ""))[:200]
            if action not in ("regenerate", "escalate"):
                action = "regenerate"
            if action == "escalate" and retry_count < 1:
                action, reason = "regenerate", "Guardrail: must try at least one correction before escalating."
            return action, focus, reason
        except Exception as e:
            logger.warning(f"[Supervisor] LLM advisor failed ({e}); defaulting to regenerate.")
            return default
