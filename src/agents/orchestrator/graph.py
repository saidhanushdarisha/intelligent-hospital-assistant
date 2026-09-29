import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import OrchestratorState

from .router import (
    router_node,
    general_node,
    readmission_risk_node,
    high_risk_patients_node,
    health_guidelines_node,
    treatment_procedures_node,
    triage_node,
    appointment_node,
    policy_search_node,
    unknown_node,
)

builder = StateGraph(OrchestratorState)

builder.add_node("router", router_node)
builder.add_node("general", general_node)
builder.add_node("readmission_risk", readmission_risk_node)
builder.add_node("high_risk_patients", high_risk_patients_node)
builder.add_node("health_guidelines", health_guidelines_node)
builder.add_node("treatment_procedures", treatment_procedures_node)
builder.add_node("triage", triage_node)
builder.add_node("appointment", appointment_node)
builder.add_node("policy_search", policy_search_node)
builder.add_node("unknown", unknown_node)

# --- Every turn goes through the router, always. ---
# Previously this bypassed the router straight to "appointment"/"triage"
# whenever active_agent was set, so an unrelated question asked
# mid-booking would get force-fed into the appointment agent. Now the
# router itself is given the active agent + its recent context and
# decides whether to continue that flow or release it — see the
# "Active-agent continuation rule" in ROUTER_PROMPT and the context
# building in router_node.
builder.add_edge(START, "router")
# -------------------------------

def routing_function(state: OrchestratorState):
    return state["route"]

builder.add_conditional_edges(
    "router",
    routing_function,
    {
        "general": "general",
        "readmission_risk": "readmission_risk",
        "high_risk_patients": "high_risk_patients",
        "health_guidelines": "health_guidelines",
        "treatment_procedures": "treatment_procedures",
        "triage": "triage",
        "appointment": "appointment",
        "policy_search": "policy_search",
        "unknown": "unknown",
    },
)

builder.add_edge("general", END)
builder.add_edge("readmission_risk", END)
builder.add_edge("high_risk_patients", END)
builder.add_edge("health_guidelines", END)
builder.add_edge("treatment_procedures", END)
builder.add_edge("triage", END)
builder.add_edge("appointment", END)
builder.add_edge("policy_search", END)
builder.add_edge("unknown", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)