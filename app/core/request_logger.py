import uuid
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.core.logger import logger


class RequestLogger:
    """
    Tracks performance and operational metadata for a single pipeline run.

    New fields (v2):
      validation_latency_seconds  — time taken by the semantic validator Gemini call
      prompt_tokens               — total prompt tokens across all Gemini calls
      completion_tokens           — total completion tokens across all Gemini calls
      estimated_cost_usd          — rough cost estimate based on token counts
      retry_reasons               — human-readable reasons for each retry
      decision_timeline           — ordered list of AI decisions with timestamps
    """

    # Gemini 1.5 Flash pricing estimate (USD per 1M tokens)
    _PROMPT_TOKEN_COST_PER_M  = 0.075
    _COMPLETION_TOKEN_COST_PER_M = 0.30

    def __init__(self, model_used: str = "", prompt_version: str = ""):
        self.request_id            = str(uuid.uuid4())
        self.model_used            = model_used
        self.fallback_model_used   = None
        self.prompt_version        = prompt_version
        self.start_time            = time.time()
        self.retry_count           = 0
        self.gemini_duration       = 0.0   # cumulative Gemini call duration (seconds)
        self.correction_loop_count = 0

        # New enterprise monitoring fields
        self.validation_latency_seconds: float = 0.0
        self.prompt_tokens: int     = 0
        self.completion_tokens: int = 0
        self.estimated_cost_usd: float = 0.0
        self.retry_reasons: List[str] = []
        self.decision_timeline: List[Dict[str, Any]] = []

        # Legacy
        self.tokens_used   = None
        self.estimated_cost = None

        self.stages:   List[Dict[str, Any]] = []
        self.errors:   List[str] = []
        self.warnings: List[str] = []
        self.metadata: Dict[str, Any] = {}

    # ── Live progress file ────────────────────────────────────────────────

    def _write_live_progress(self):
        try:
            progress_data = {
                "request_id": self.request_id,
                "stages":     self.stages,
                "errors":     self.errors,
                "warnings":   self.warnings,
                "timestamp":  time.time(),
                "decision_timeline": self.decision_timeline,
            }
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            outputs_dir = os.path.join(project_root, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            filepath = os.path.join(outputs_dir, "live_progress.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write live progress: {e}")

    # ── Stage tracking ────────────────────────────────────────────────────

    def start_stage(self, stage_name: str):
        logger.info(f"[{self.request_id}] Starting stage: {stage_name}")
        self.stages.append({
            "stage":      stage_name,
            "start_time": time.time(),
            "status":     "IN_PROGRESS",
            "end_time":   None,
            "duration":   None,
        })
        self._record_timeline_event(stage_name, "STARTED")
        self._write_live_progress()

    def end_stage(self, stage_name: str, status: str = "SUCCESS", details: Any = None):
        end_time = time.time()
        for stage in self.stages:
            if stage["stage"] == stage_name and stage["end_time"] is None:
                stage["status"]   = status
                stage["end_time"] = end_time
                stage["duration"] = round(end_time - stage["start_time"], 4)
                if details:
                    stage["details"] = details
                logger.info(
                    f"[{self.request_id}] Completed stage: {stage_name} "
                    f"in {stage['duration']:.4f}s | status={status}"
                )
                break
        self._record_timeline_event(stage_name, status)
        self._write_live_progress()

    # ── Timeline ──────────────────────────────────────────────────────────

    def _record_timeline_event(self, label: str, status: str, detail: str = ""):
        self.decision_timeline.append({
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
            "label":     label,
            "status":    status,
            "detail":    detail,
        })

    def record_validation_decision(
        self,
        detected_category: str,
        accepted: List[str],
        rejected: List[str],
        warned: List[str],
        latency: float,
    ):
        """Record semantic validation decisions into the timeline."""
        self.validation_latency_seconds = latency
        if rejected:
            for attr in rejected:
                self._record_timeline_event(f"Rejected: {attr}", "REJECTED")
        if warned:
            for w in warned:
                attr = w if isinstance(w, str) else w.get("attribute", "")
                self._record_timeline_event(f"Warning: {attr}", "WARNING")
        self._record_timeline_event(
            f"Category: {detected_category}",
            "CLASSIFIED",
            f"Accepted={len(accepted)} | Rejected={len(rejected)} | Warned={len(warned)}",
        )

    # ── Token + cost tracking ─────────────────────────────────────────────

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens     += prompt_tokens
        self.completion_tokens += completion_tokens
        self._recalculate_cost()

    def _recalculate_cost(self):
        prompt_cost     = (self.prompt_tokens     / 1_000_000) * self._PROMPT_TOKEN_COST_PER_M
        completion_cost = (self.completion_tokens / 1_000_000) * self._COMPLETION_TOKEN_COST_PER_M
        self.estimated_cost_usd = round(prompt_cost + completion_cost, 6)

    # ── Misc ──────────────────────────────────────────────────────────────

    def add_error(self, err_msg: str):
        logger.error(f"[{self.request_id}] Error: {err_msg}")
        self.errors.append(err_msg)

    def add_warning(self, warn_msg: str):
        logger.warning(f"[{self.request_id}] Warning: {warn_msg}")
        self.warnings.append(warn_msg)

    def increment_retries(self, reason: str = ""):
        self.retry_count += 1
        if reason:
            self.retry_reasons.append(reason)
        logger.info(
            f"[{self.request_id}] Retry #{self.retry_count}"
            + (f" — {reason}" if reason else "")
        )

    # ── Finalize ──────────────────────────────────────────────────────────

    def finalize(self) -> Dict[str, Any]:
        end_time       = time.time()
        total_duration = round(end_time - self.start_time, 4)

        log_report = {
            "request_id":                  self.request_id,
            "model_used":                  self.model_used,
            "fallback_model_used":         self.fallback_model_used,
            "prompt_version":              self.prompt_version,
            "start_time":                  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.start_time)),
            "execution_time_seconds":      total_duration,
            "gemini_duration_seconds":     round(self.gemini_duration, 4),
            "validation_latency_seconds":  round(self.validation_latency_seconds, 4),
            "retry_count":                 self.retry_count,
            "retry_reasons":               self.retry_reasons,
            "correction_loop_count":       self.correction_loop_count,
            "prompt_tokens":               self.prompt_tokens,
            "completion_tokens":           self.completion_tokens,
            "estimated_cost_usd":          self.estimated_cost_usd,
            "stages":                      self.stages,
            "decision_timeline":           self.decision_timeline,
            "errors":                      self.errors,
            "warnings":                    self.warnings,
            "metadata":                    self.metadata,
            # Legacy aliases
            "tokens_used":                 self.tokens_used,
            "estimated_cost":              self.estimated_cost,
        }

        filepath = os.path.join(settings.outputs_dir, f"execution_log_{self.request_id}.json")
        static_filepath = os.path.join(settings.outputs_dir, "execution_log.json")
        for path in [filepath, static_filepath]:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(log_report, f, indent=2)
            except Exception as e:
                logger.error(f"[{self.request_id}] Failed to write log to {path}: {e}")

        return log_report
