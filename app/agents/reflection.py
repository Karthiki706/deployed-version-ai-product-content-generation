from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from app.agents.base import BaseAgent
from app.engines.reflection import ReflectionEngine
from app.orchestrator.graph_state import GraphState

class ReflectionAgent(BaseAgent):
    """Reflection Agent summarizing weaknesses from compliance check violations."""
    
    def __init__(self):
        super().__init__("Reflection Agent")
        
    def execute(self, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        h_model = state.get("hallucination_report")
        s_model = state.get("seo_report")
        b_model = state.get("brand_report")
        json_error = state.get("json_error")
        
        reflection_data = ReflectionEngine.reflect(h_model, s_model, b_model, json_error)
        return reflection_data
