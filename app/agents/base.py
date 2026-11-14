import time
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from app.orchestrator.graph_state import GraphState
from app.core.request_logger import RequestLogger
from app.core.logger import logger

def sanitize_serializable(obj: Any) -> Any:
    """Helper to convert Pydantic models and other complex structures to serializable forms."""
    if isinstance(obj, dict):
        return {k: sanitize_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_serializable(x) for x in obj]
    elif hasattr(obj, "model_dump"):
        try:
            return sanitize_serializable(obj.model_dump())
        except Exception:
            pass
    elif hasattr(obj, "dict"):
        try:
            return sanitize_serializable(obj.dict())
        except Exception:
            pass
    return obj

class BaseAgent:
    """Base class for all agents. Handles observability, timeline logging, and duration tracking."""
    
    def __init__(self, name: str):
        self.name = name
        
    def __call__(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """Callable interface matching LangGraph node expectation. Traces start/end durations."""
        req_logger: Optional[RequestLogger] = state.get("request_id") and state.get("request_logger")
        attempt_idx = state.get("retry_count", 0) + 1
        stage_label = f"{self.name} (Attempt {attempt_idx})" if attempt_idx > 1 else self.name
        
        if req_logger:
            req_logger.start_stage(stage_label)
            
        start_time = time.time()
        try:
            # Execute actual agent code
            updates = self.execute(state, config)
            duration = time.time() - start_time
            
            if req_logger:
                sanitized_details = sanitize_serializable(updates)
                req_logger.end_stage(stage_label, status="SUCCESS", details=sanitized_details)
                
            if "timeline" not in state or state["timeline"] is None:
                state["timeline"] = []
                
            state["timeline"].append({
                "agent": self.name,
                "status": "COMPLETED",
                "duration_seconds": round(duration, 4),
                "timestamp": time.time(),
            })
            return updates
        except Exception as e:
            duration = time.time() - start_time
            if req_logger:
                req_logger.add_error(f"Error in agent {self.name}: {str(e)}")
                req_logger.end_stage(stage_label, status="FAILED")
                
            if "timeline" not in state or state["timeline"] is None:
                state["timeline"] = []
                
            state["timeline"].append({
                "agent": self.name,
                "status": "FAILED",
                "error": str(e),
                "duration_seconds": round(duration, 4),
                "timestamp": time.time(),
            })
            raise e
            
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        """Logic execution. Must be overridden by subclasses."""
        raise NotImplementedError
