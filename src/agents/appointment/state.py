import operator
from typing import TypedDict, Optional, Annotated, List, Any
from langchain_core.messages import BaseMessage

class OrchestratorState(TypedDict):
    user_input: str

    # one or more selected destinations, in execution order
    route: List[str]

    # each node appends its own message here; the reducer merges
    # writes from parallel branches instead of one overwriting another
    msg: Annotated[List[str], operator.add]

    # structured output from the appointment node, when it runs
    appointment_result: Optional[dict]
    
    response: Optional[str]

    # --- Added state fields required for multi-turn conversational memory ---
    messages: List[BaseMessage]        # Captures active tool invocation messages
    chat_history: List[BaseMessage]    # Context window memory for the router
    active_agent: Optional[str]        # Keeps state on which agent is mid-process