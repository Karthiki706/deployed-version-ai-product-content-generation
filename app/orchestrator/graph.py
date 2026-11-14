from app.orchestrator.builder import GraphBuilder

# Compile the compiled runnable workflow graph once
compiled_graph = GraphBuilder.build_graph().compile()
