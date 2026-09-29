import operator
from typing import TypedDict, Optional, Annotated, List, Any


class OrchestratorState(TypedDict):
    user_input: str

    # one or more selected destinations, in execution order
    route: List[str]

    # each node appends its own message here; the reducer merges
    # writes from parallel branches instead of one overwriting another
    msg: Annotated[List[str], operator.add]

    # structured output from the high_risk_patients node, when it runs
    high_risk_patients_result: Optional[dict]

    response: Optional[str]