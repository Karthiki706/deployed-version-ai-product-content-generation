import os
import json
import time
from typing import Dict, Any, Optional, List

from app.core.config import settings
from app.core.logger import logger
from app.core.request_logger import RequestLogger

from app.models.product import Product
from app.models.response import GeneratedContent, PipelineResponse

from app.orchestrator.graph import compiled_graph


class AIOrchestrator:
    """The master orchestrator using LangGraph multi-agent execution pipeline."""

    def run_pipeline(
        self,
        product: Product,
        api_key_override: Optional[str] = None,
    ) -> PipelineResponse:

        # Instantiate unified request logger for observability timeline and cost metrics
        req_logger = RequestLogger(
            model_used=settings.gemini_model,
            prompt_version="v1",
        )
        logger.info(f"--- [Pipeline Started via LangGraph] Request ID: {req_logger.request_id} ---")

        try:
            # 1. Initialize GraphState dictionary parameters
            initial_state = {
                "request_id": req_logger.request_id,
                "product": product,
                "request_logger": req_logger,
                "retry_count": 0,
                "best_quality_score": 0.0,
                "decision_log": [],
                "agent_trace": [],
                "timeline": [],
            }

            # 2. Invoke the compiled LangGraph with configurations (API overrides and custom recursion limit)
            final_state = compiled_graph.invoke(
                initial_state,
                config={
                    "configurable": {"api_key_override": api_key_override},
                    "recursion_limit": 50
                }
            )

            # 3. Extract final states
            final_content = final_state.get("generated_content")
            h_model = final_state.get("hallucination_report")
            s_model = final_state.get("seo_report")
            b_model = final_state.get("brand_report")
            q_model = final_state.get("quality_report")
            validation_rep = final_state.get("validation_report", {})
            prompt_version = final_state.get("prompt_version", "v1")
            semantic_status = final_state.get("semantic_status", "PASS")

            # Convert Pydantic models back to standard dictionaries for report serialization
            h_rep = h_model.model_dump() if h_model else {}
            s_rep = s_model.model_dump() if s_model else {}
            b_rep = b_model.model_dump() if b_model else {}
            q_rep = q_model.model_dump() if q_model else {}

            # Ensure prompt version is set in all audit report contexts
            for rep in [h_rep, s_rep, b_rep, q_rep]:
                if rep:
                    rep["prompt_version"] = prompt_version

            # 4. Finalize req_logger to persist execution logs
            execution_log = req_logger.finalize()

            # 5. Persist JSON reports toOutputs directory (unique + static filenames)
            self._save_reports(
                req_logger.request_id,
                final_content.model_dump() if final_content else {},
                h_rep, s_rep, b_rep, q_rep, validation_rep,
            )

            # 6. Assemble API Pydantic Response payload matching the pipeline contract
            return PipelineResponse(
                request_id=req_logger.request_id,
                content=final_content,
                scores={
                    "overall":      q_rep.get("overall_quality", 0.0),
                    "hallucination": q_rep.get("hallucination_score", 0.0),
                    "seo":          q_rep.get("seo_score", 0.0),
                    "brand":        q_rep.get("brand_score", 0.0),
                },
                reports={
                    "hallucination": h_rep,
                    "seo":          s_rep,
                    "brand":        b_rep,
                    "quality":      q_rep,
                    "validation":   validation_rep,
                },
                metadata={
                    "execution_time_seconds":     execution_log.get("execution_time_seconds", 0.0),
                    "gemini_duration_seconds":    execution_log.get("gemini_duration_seconds", 0.0),
                    "validation_latency_seconds": execution_log.get("validation_latency_seconds", 0.0),
                    "retry_count":                execution_log.get("retry_count", 0),
                    "correction_loop_count":      execution_log.get("correction_loop_count", 0),
                    "prompt_tokens":              execution_log.get("prompt_tokens", 0),
                    "completion_tokens":          execution_log.get("completion_tokens", 0),
                    "estimated_cost_usd":         execution_log.get("estimated_cost_usd", 0.0),
                    "model_used":                 execution_log.get("model_used", ""),
                    "fallback_model_used":        execution_log.get("fallback_model_used"),
                    "prompt_version":             prompt_version,
                    "semantic_status":            semantic_status,
                    "risk_level":                 q_rep.get("risk_level", "Low"),
                    "decision_timeline":          execution_log.get("decision_timeline", []),
                    "agentic_mode":               settings.agentic_mode,
                    "agent_trace":                final_state.get("agent_trace", []),
                    "planner_source":             final_state.get("planner_source", "deterministic"),
                    "supervisor_decisions":       [d for d in final_state.get("decision_log", []) if "Supervisor (LLM)" in d],
                    "manual_review_required":     bool(final_state.get("manual_review_required", False)),
                },
            )

        except Exception as e:
            req_logger.add_error(f"Pipeline crashed with error: {str(e)}")
            req_logger.finalize()
            raise e

    def _save_reports(
        self,
        request_id: str,
        content: dict,
        h: dict,
        s: dict,
        b: dict,
        q: dict,
        v: dict,
    ):
        """Write all reports to outputs/ folder for Streamlit display check."""
        report_mappings = {
            "generated_content":    content,
            "hallucination_report": h,
            "seo_report":           s,
            "brand_report":         b,
            "quality_report":       q,
            "validation_report":    v,
        }
        for base_name, report_data in report_mappings.items():
            unique_path = os.path.join(settings.outputs_dir, f"{base_name}_{request_id}.json")
            static_path = os.path.join(settings.outputs_dir, f"{base_name}.json")
            for path in [unique_path, static_path]:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(report_data, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to write report file {path}: {str(e)}")
