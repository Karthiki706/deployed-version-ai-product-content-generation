from langgraph.graph import StateGraph, END
from app.orchestrator.graph_state import GraphState
from app.orchestrator.registry import agent_registry
from app.orchestrator.routers import supervisor_router
import app.orchestrator.nodes  # Triggers node registration inside nodes.py

class GraphBuilder:
    """Builder class to construct and compile the LangGraph workflow."""
    
    @staticmethod
    def build_graph() -> StateGraph:
        # Initialize graph with GraphState TypedDict
        workflow = StateGraph(GraphState)
        
        # 1. Add all registered agents/nodes using registry metadata
        for name, meta in agent_registry.get_all().items():
            workflow.add_node(name, meta["agent"])
            
        # 2. Set entrance node
        workflow.set_entry_point("normalization_node")
        
        # 3. Define sequential pipeline up to JSON Validation
        workflow.add_edge("normalization_node", "classification_node")
        workflow.add_edge("classification_node", "validation_node")
        workflow.add_edge("validation_node", "planner_agent_node")
        workflow.add_edge("planner_agent_node", "enrichment_node")
        workflow.add_edge("enrichment_node", "prompt_builder_node")
        workflow.add_edge("prompt_builder_node", "generation_node")
        workflow.add_edge("generation_node", "json_validation_node")
        
        # 4. Fork JSON Validation output into Parallel Compliance Auditing branches
        workflow.add_edge("json_validation_node", "hallucination_node")
        workflow.add_edge("json_validation_node", "seo_node")
        workflow.add_edge("json_validation_node", "brand_node")
        
        # 5. Merge parallel execution paths into Quality Scoring node
        workflow.add_edge("hallucination_node", "quality_scoring_node")
        workflow.add_edge("seo_node", "quality_scoring_node")
        workflow.add_edge("brand_node", "quality_scoring_node")
        
        # 6. Quality Scoring feeds into Reflection Agent and then to Supervisor Node
        workflow.add_edge("quality_scoring_node", "reflection_agent_node")
        workflow.add_edge("reflection_agent_node", "supervisor_agent_node")
        
        # 7. Configure Supervisor Router conditional gates
        workflow.add_conditional_edges(
            "supervisor_agent_node",
            supervisor_router,
            {
                "CORRECT": "correction_node",
                "END": END
            }
        )
        
        # 8. Loop back Correction Node to Planner Agent Node (Self-Correction retry path)
        workflow.add_edge("correction_node", "planner_agent_node")
        
        return workflow
