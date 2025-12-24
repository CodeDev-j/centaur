from typing import List, Annotated, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    The Short-Term Memory of the Brain.
    Tracks the conversation history and the plan.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    context: Dict[str, Any]  # Stores retrieved documents
    plan: List[str]          # The steps the agent intends to take
    final_answer: str        # The answer to display to the user