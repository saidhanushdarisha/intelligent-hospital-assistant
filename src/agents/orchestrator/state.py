import operator
from typing import TypedDict, Optional, Annotated, List, Any
from langchain_core.messages import BaseMessage


class OrchestratorState(TypedDict):
    user_input: str

    # one or more selected destinations, in execution order
    route: List[str]

    # each node appends its own message here (this is what app.py prints)
    msg: Annotated[List[str], operator.add]

    response: Optional[str]

    # Tracks which node is currently in control of a multi-turn conversation
    active_agent: Optional[str]

    # The appointment agent's own tool-calling transcript (system/human/ai/
    # tool messages for the booking ReAct loop). Scoped to that agent only.
    messages: Annotated[List[BaseMessage], operator.add]

    # NEW: general short-term conversational memory for the whole
    # orchestrator. The router reads this to understand follow-ups and
    # topic continuity across turns, and every node that produces a
    # user-facing reply appends its own turn here too — so later turns,
    # even handled by a different node, have context.
    chat_history: Annotated[List[BaseMessage], operator.add]

    # Role of the current user (e.g., 'patient' or 'staff')
    user_role: Optional[str]