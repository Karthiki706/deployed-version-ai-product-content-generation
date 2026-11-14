from app.orchestrator.graph_state import GraphState

def supervisor_router(state: GraphState) -> str:
    """Routing function returning the next step decided by the Supervisor Agent node."""
    next_step = state.get("next_step", "END")
    if next_step not in ["CORRECT", "END"]:
        return "END"
    return next_step
