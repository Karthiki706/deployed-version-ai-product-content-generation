from typing import Dict, Callable, Any, Optional

class AgentRegistry:
    """A pluggable registry for LangGraph workflow nodes/agents."""
    
    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        
    def register(self, name: str, agent_instance: 'BaseAgent', node_type: str = "sequential", depends_on: Optional[str] = None):
        """Registers a BaseAgent instance into the workflow registry metadata database."""
        self._agents[name] = {
            "agent": agent_instance,
            "type": node_type,
            "depends_on": depends_on
        }
        
    def get_node(self, name: str) -> Dict[str, Any]:
        """Retrieve the registered agent node function by name."""
        if name not in self._agents:
            raise KeyError(f"Agent node '{name}' is not registered in the AgentRegistry.")
        return self._agents[name]
        
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve all registered node functions."""
        return self._agents

# Global registry instance
agent_registry = AgentRegistry()
