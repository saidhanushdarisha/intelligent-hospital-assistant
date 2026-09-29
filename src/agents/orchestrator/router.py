import json
import numpy as np
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from agents.appointment.tools import APPOINTMENT_TOOLS
from agents.readmission_risk.tools.tools import READMISSION_RISK_TOOLS
from agents.high_risk_patients.tools.tools import HIGH_RISK_PATIENTS_TOOLS

# Import LLMs
from agents.appointment.models import appointment_llm
from agents.readmission_risk.models import readmission_risk_llm
from agents.high_risk_patients.models import high_risk_patients_llm
from .models import router_llm

# Policy/RAG stores are loaded lazily. This keeps the Patient Portal fast to open
# because embedding models and vector indexes are not initialized until the
# corresponding service is actually used.
from agents.policy_search.models import policy_search_llm
from functools import lru_cache

from .prompts import (
    ROUTER_PROMPT,
    TOOL_PRIORITY,
    GENERAL_PROMPT,
    READMISSION_RISK_PROMPT,
    HIGH_RISK_PATIENTS_PROMPT,
    TREATMENT_PROCEDURES_PROMPT,
    POLICY_SEARCH_PROMPT,
)
from .state import OrchestratorState

@lru_cache(maxsize=1)
def get_policy_db():
    from agents.policy_search.vector_store import create_db
    try:
        return create_db()
    except Exception as e:
        print(f"Warning: Could not initialize policy_db: {e}")
        return None


@lru_cache(maxsize=1)
def get_bge_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


@lru_cache(maxsize=1)
def get_health_guidelines_db():
    from langchain_community.vectorstores import FAISS
    try:
        return FAISS.load_local(
            "src/agents/health_guidelines/data/vector_store",
            get_bge_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Warning: Could not initialize health_guidelines_db: {e}")
        return None


@lru_cache(maxsize=1)
def get_treatment_procedures_db():
    from langchain_community.vectorstores import FAISS
    try:
        return FAISS.load_local(
            "src/agents/treatment_procedures/data/vector_store",
            get_bge_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Warning: Could not initialize treatment_procedures_db: {e}")
        return None


# Bind the tools to the LLMs
appointment_tools = appointment_llm.bind_tools(APPOINTMENT_TOOLS)
readmission_risk_tools = readmission_risk_llm.bind_tools(READMISSION_RISK_TOOLS)
high_risk_patients_tools_llm = high_risk_patients_llm.bind_tools(HIGH_RISK_PATIENTS_TOOLS)

VALID_TOOLS = {
    "general",
    "readmission_risk",
    "high_risk_patients",
    "health_guidelines",
    "treatment_procedures",
    "triage",
    "appointment",
    "policy_search",
    "unknown",
}

MAX_HISTORY_PAIRS = 4
MAX_AGENT_TRANSCRIPT_PAIRS = 3


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _format_messages(messages: list, max_pairs: int) -> str:
    if not messages:
        return ""
    lines = []
    for m in messages[-(max_pairs * 2):]:
        role = getattr(m, "type", m.__class__.__name__)
        content = getattr(m, "content", "")
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _trim_agent_history(history: list, max_recent: int = 10) -> list:
    """
    Safely trims the message history to prevent 413 Payload Too Large errors.
    Ensures we don't orphan ToolMessages from their calling AIMessages by 
    always starting the trimmed history on a HumanMessage.
    """
    if not history or len(history) <= 1:
        return history
        
    sys_msg = history[0]
    if len(history) <= max_recent + 1:
        return history
        
    recent_msgs = history[-max_recent:]
    
    while recent_msgs:
        role = getattr(recent_msgs[0], "type", recent_msgs[0].__class__.__name__).lower()
        if "human" in role:
            break
        recent_msgs.pop(0)
        
    return [sys_msg] + recent_msgs


def parse_router_output(raw_output: str) -> list[str]:
    cleaned_raw = _strip_code_fences(raw_output)
    try:
        tools = json.loads(cleaned_raw)
    except json.JSONDecodeError:
        return ["unknown"]

    if not isinstance(tools, list) or not tools:
        return ["unknown"]

    seen = set()
    cleaned = []
    for t in tools:
        if isinstance(t, str):
            t_norm = t.strip().lower()
            if t_norm in VALID_TOOLS and t_norm not in seen:
                cleaned.append(t_norm)
                seen.add(t_norm)

    if not cleaned:
        return ["unknown"]
    if "unknown" in cleaned and len(cleaned) > 1:
        cleaned = [t for t in cleaned if t != "unknown"]
    if "general" in cleaned and len(cleaned) > 1:
        cleaned = [t for t in cleaned if t != "general"]
    if len(cleaned) > 3:
        cleaned = sorted(cleaned, key=lambda t: TOOL_PRIORITY.get(t, 999))[:3]

    return sorted(cleaned, key=lambda t: TOOL_PRIORITY.get(t, 999))


def router_node(state: OrchestratorState):
    user_query = state["user_input"]
    active_agent = state.get("active_agent")
    chat_history = state.get("chat_history", [])
    context_blocks = []

    history_snippet = _format_messages(chat_history, MAX_HISTORY_PAIRS)
    if history_snippet:
        context_blocks.append(f"RECENT CONVERSATION:\n{history_snippet}")

    if active_agent:
        agent_snippet = _format_messages(state.get("messages", []), MAX_AGENT_TRANSCRIPT_PAIRS)
        active_block = f"ACTIVE_AGENT: {active_agent}"
        if agent_snippet:
            active_block += f"\nThat agent's recent exchange with the user:\n{agent_snippet}"
        context_blocks.append(active_block)

    context_text = "\n\n".join(context_blocks)
    human_content = f"{context_text}\n\nCURRENT_MESSAGE: {user_query}" if context_text else user_query

    messages = [("system", ROUTER_PROMPT), ("human", human_content)]
    raw_output = router_llm.invoke(messages).content
    decision = parse_router_output(raw_output)

    # Patient-role restriction rule
    user_role = state.get("user_role")
    if user_role == "patient":
        patient_allowed = {"general", "health_guidelines", "triage", "appointment", "policy_search", "unknown"}
        # If the LLM tried to route to a staff-only feature, replace it with 'unknown'
        filtered_decision = [r for r in decision if r in patient_allowed]
        if len(filtered_decision) < len(decision) or not filtered_decision:
            decision = ["unknown"]
        else:
            decision = filtered_decision

    multi_turn_agents = {"appointment", "readmission_risk", "high_risk_patients"}
    new_active_agent = decision[0] if len(decision) == 1 and decision[0] in multi_turn_agents else None

    return {
        "route": decision,
        "active_agent": new_active_agent,
        "chat_history": [HumanMessage(content=user_query)],
    }


def sanitize_tool_args(obj):
    """Converts numpy data types to native Python types to avoid JSON errors."""
    if isinstance(obj, dict):
        return {k: sanitize_tool_args(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_tool_args(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return obj


def appointment_node(state: OrchestratorState):
    user_query = state["user_input"]
    history = state.get("messages", [])

    if not history:
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        sys_msg = SystemMessage(content=f"""
        You are a highly accurate hospital appointment booking agent. 
        Current time: {current_time}.
        Rules:
        1. ALWAYS use `check_doctor_availability` before confirming a booking.
        2. If the user asks which doctors or slots are available without giving a date, use `search_doctors_directory` or `find_next_available_slots`.
        3. Gather the patient name, department, date, and time slot before booking.
        3. If you have finished the user's booking/cancellation request and they are satisfied, end your response with the exact string: "[TASK_COMPLETE]".
        """)
        history = [sys_msg]
    else:
        history = _trim_agent_history(history, max_recent=10)

    history.append(HumanMessage(content=user_query))
    response = appointment_tools.invoke(history)
    history.append(response)

    while response.tool_calls:
        for tool_call in response.tool_calls:
            selected_tool = next((t for t in APPOINTMENT_TOOLS if t.name == tool_call["name"]), None)
            if selected_tool:
                try:
                    tool_output = selected_tool.invoke(tool_call["args"])
                except Exception as e:
                    tool_output = f"Error: {str(e)}"
            else:
                tool_output = "Error: Tool not found."
            history.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

        response = appointment_tools.invoke(history)
        history.append(response)

    final_text = response.content
    active_agent = "appointment" 

    if "[TASK_COMPLETE]" in final_text:
        final_text = final_text.replace("[TASK_COMPLETE]", "").strip()
        active_agent = None  

    return {
        "msg": [f"[Appointment] {final_text}"],
        "messages": history, 
        "active_agent": active_agent,
        "chat_history": [AIMessage(content=final_text)],
    }


def readmission_risk_node(state: OrchestratorState):
    user_query = state["user_input"]
    history = state.get("messages", [])

    if not history:
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        sys_msg = SystemMessage(
            content=f"""
You are a hospital readmission risk prediction agent.
Current time: {current_time}

Your responsibilities:
1. Collect all required patient details (Age, Gender, Condition, Procedure, Cost, Length_of_Stay, Outcome, Satisfaction).
2. ALWAYS use the predict_readmission tool when patient information is available.
3. Explain: Whether the patient is high/low risk, Confidence score, Important factors from the input.
4. If prediction is completed and the user is satisfied, end your response with: [TASK_COMPLETE]
"""
        )
        history = [sys_msg]
    else:
        history = _trim_agent_history(history, max_recent=10)

    history.append(HumanMessage(content=user_query))
    response = readmission_risk_tools.invoke(history)
    history.append(response)

    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool_call["args"] = sanitize_tool_args(tool_call["args"])

        for tool_call in response.tool_calls:
            selected_tool = next((tool for tool in READMISSION_RISK_TOOLS if tool.name == tool_call["name"]), None)
            if selected_tool:
                try:
                    tool_output = selected_tool.invoke(tool_call["args"])
                except Exception as e:
                    print(f"\n[TOOL EXCEPTION]: {type(e).__name__} - {e}\n")
                    tool_output = {"error": str(e)}
            else:
                tool_output = {"error": "Tool not found"}

            history.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

        response = readmission_risk_llm.invoke(history)
        history.append(response)

    final_text = response.content or ""
    active_agent = "readmission_risk"

    if "[TASK_COMPLETE]" in final_text:
        final_text = final_text.replace("[TASK_COMPLETE]", "").strip()
        active_agent = None

    return {
        "msg": [f"[Readmission Risk] {final_text}"],
        "messages": history,
        "active_agent": active_agent,
        "readmission_risk_result": {"response": final_text},
        "chat_history": [AIMessage(content=final_text)]
    }


def high_risk_patients_node(state: OrchestratorState):
    user_query = state["user_input"]
    history = state.get("messages", [])

    if not history:
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        sys_msg = SystemMessage(
            content=f"""
You are a hospital high-risk patient monitoring agent.
Current time: {current_time}

Your responsibility:
1. Identify whether a patient is currently at high risk.
2. Collect patient details (Respiratory Rate, Oxygen Saturation, O2 Scale, Systolic BP, Heart Rate, Temperature, Consciousness level, On oxygen).
3. ALWAYS use the predict_highrisk tool when enough information is available.
4. After prediction explain: Risk Level, Confidence score, Why the patient may require attention.
5. Do not diagnose diseases. Only report risk prediction from the ML model.
6. When the user request is completed and the user is satisfied, end your response with: [TASK_COMPLETE]
"""
        )
        history = [sys_msg]
    else:
        history = _trim_agent_history(history, max_recent=10)

    history.append(HumanMessage(content=user_query))
    response = high_risk_patients_tools_llm.invoke(history)
    history.append(response)

    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool_call["args"] = sanitize_tool_args(tool_call["args"])

        for tool_call in response.tool_calls:
            selected_tool = next((tool for tool in HIGH_RISK_PATIENTS_TOOLS if tool.name == tool_call["name"]), None)
            if selected_tool:
                try:
                    tool_output = selected_tool.invoke(tool_call["args"])
                except Exception as e:
                    print(f"\n[TOOL EXCEPTION]: {type(e).__name__} - {e}\n")
                    tool_output = {"error": str(e)}
            else:
                tool_output = {"error": "Tool not found"}

            history.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

        response = high_risk_patients_llm.invoke(history)
        history.append(response)

    final_text = response.content or ""
    active_agent = "high_risk_patients"

    if "[TASK_COMPLETE]" in final_text:
        final_text = final_text.replace("[TASK_COMPLETE]", "").strip()
        active_agent = None

    return {
        "msg": [f"[High Risk Patients] {final_text}"],
        "messages": history,
        "active_agent": active_agent,
        "high_risk_patients_result": {"response": final_text},
        "chat_history": [AIMessage(content=final_text)]
    }


def health_guidelines_node(state: OrchestratorState):
    user_query = state["user_input"]
    
    health_guidelines_db = get_health_guidelines_db()
    if health_guidelines_db is None:
        reply = "I couldn't find this information in the hospital documentation. The database is unavailable."
        return {
            "msg": [f"[Health Guidelines] {reply}"],
            "chat_history": [AIMessage(content=reply)],
        }
    
    docs = health_guidelines_db.similarity_search(user_query, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
You are an AI Hospital Assistant.

Answer ONLY using the document excerpts below.

Do NOT use your own medical knowledge.

If the answer cannot be found in the document excerpts, reply exactly:

"I couldn't find this information in the hospital documentation."

-----------------------------
DOCUMENT EXCERPTS
-----------------------------

{context}

-----------------------------
USER QUESTION
-----------------------------

{user_query}

-----------------------------
ANSWER
-----------------------------
"""
    
    messages = [
        SystemMessage(content="You are a document question answering assistant."),
        HumanMessage(content=prompt)
    ]
    
    reply = router_llm.invoke(messages).content.strip()

    return {
        "msg": [f"[Health Guidelines] {reply}"],
        "chat_history": [AIMessage(content=reply)],
    }


# --- NEW: Integrated Treatment Procedures Node ---
def treatment_procedures_node(state: OrchestratorState):
    user_query = state["user_input"]
    
    treatment_procedures_db = get_treatment_procedures_db()
    if treatment_procedures_db is None:
        reply = "I couldn't find this information in the hospital documentation. The database is unavailable."
        return {
            "msg": [f"[Treatment Procedures] {reply}"],
            "chat_history": [AIMessage(content=reply)],
        }
    
    docs = treatment_procedures_db.similarity_search(user_query, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
You are an AI Hospital Assistant.

Answer ONLY using the document excerpts below.

Do NOT use your own medical knowledge.

If the answer cannot be found in the document excerpts, reply exactly:

"I couldn't find this information in the hospital documentation."

-----------------------------
DOCUMENT EXCERPTS
-----------------------------

{context}

-----------------------------
USER QUESTION
-----------------------------

{user_query}

-----------------------------
ANSWER
-----------------------------
"""
    
    messages = [
        SystemMessage(content="You are a document question answering assistant."),
        HumanMessage(content=prompt)
    ]
    
    reply = router_llm.invoke(messages).content.strip()

    return {
        "msg": [f"[Treatment Procedures] {reply}"],
        "chat_history": [AIMessage(content=reply)],
    }


def policy_search_node(state: OrchestratorState):
    user_query = state["user_input"]
    
    policy_db = get_policy_db()
    if policy_db is None:
        reply = "I could not find this policy. The database is unavailable."
        return {
            "msg": [f"[Policy Search] {reply}"],
            "chat_history": [AIMessage(content=reply)],
        }
    
    docs = policy_db.similarity_search(user_query, k=3)
    context = "\n\n".join(d.page_content for d in docs)
    
    prompt = f"""You are a Hospital Policy Assistant.

Rules:
1. Use ONLY the context.
2. Do not invent answers.
3. If unavailable say:
"I could not find this policy."

Context:
{context}

Question:
{user_query}

Answer format:
Policy:
Name

Summary:
Short explanation

Details:
- point 1
- point 2

Reference:
Hospital Documents"""
    
    messages = [("human", prompt)]
    reply = policy_search_llm.invoke(messages).content.strip()

    return {
        "msg": [f"[Policy Search] {reply}"],
        "chat_history": [AIMessage(content=reply)],
    }


def general_node(state: OrchestratorState):
    user_query = state["user_input"]
    history_snippet = _format_messages(state.get("chat_history", []), MAX_HISTORY_PAIRS)

    human_content = user_query
    if history_snippet:
        human_content = f"Recent conversation:\n{history_snippet}\n\nUser: {user_query}"

    messages = [("system", GENERAL_PROMPT), ("human", human_content)]
    reply = router_llm.invoke(messages).content.strip()

    return {
        "msg": [reply],
        "chat_history": [AIMessage(content=reply)],
    }


TRIAGE_SYSTEM_PROMPT = """
You are an AI Hospital Triage Assistant.

Your role is to perform symptom triage, NOT diagnose diseases.

Your objectives:

1. Understand the patient's symptoms.
2. Identify emergency warning signs.
3. Recommend the most appropriate hospital department.
4. Explain your reasoning.
5. Recommend safe next steps.
6. Ask only essential follow-up questions if information is missing.

Never:
- Diagnose diseases.
- Prescribe medication.
- Recommend drug dosages.
- Invent information.
- Ignore emergency symptoms.

Always evaluate:

• Age
• Sex (if relevant)
• Symptoms
• Duration
• Severity
• Pain location
• Existing medical conditions
• Current medications (if mentioned)
• Allergies (if mentioned)
• Associated symptoms
• Vital signs (if available)

Emergency warning signs include:

- Severe chest pain
- Difficulty breathing
- Stroke symptoms
- Loss of consciousness
- Seizures
- Heavy bleeding
- Severe allergic reactions
- Suicidal thoughts
- Severe head injury
- Blue lips
- Rapidly worsening symptoms

Recommended Departments:

- Emergency Department
- General Medicine
- Cardiology
- Neurology
- Orthopedics
- General Surgery
- Pediatrics
- Obstetrics & Gynecology
- Pulmonology
- Gastroenterology
- Nephrology
- Urology
- Dermatology
- Psychiatry
- ENT
- Ophthalmology
- Oncology
- Infectious Diseases
- Endocrinology
- ICU (only if clearly appropriate)
- Unable to Determine

If information is insufficient,
ask only the minimum necessary follow-up questions.

Return ONLY the following format:

Recommended Department:
Emergency Status:
Confidence:

Reason:

Potential Risk Factors:

Recommended Action:

Follow-up Questions:

Safety Note:
This is a triage assessment only and is not a medical diagnosis.
"""


def triage_node(state: OrchestratorState):
    user_query = state["user_input"]
    messages = [
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]

    reply = router_llm.invoke(messages).content.strip()
    formatted_reply = f"[Triage]\n{reply}"

    return {
        "triage_result": {"response": reply},
        "msg": [formatted_reply],
        "chat_history": [AIMessage(content=reply)],
    }


def unknown_node(state: OrchestratorState):
    user_role = state.get("user_role")
    if user_role == "patient":
        reply = (
            "I'm the patient assistant, so I can only help with things like "
            "symptom assessment, booking appointments, medical guidelines, "
            "and hospital/insurance policy search. I'm not able to help with that request."
        )
    else:
        reply = (
            "I'm the hospital assistant, so I can only help with things like "
            "symptom triage, booking appointments, medical guidelines, "
            "treatment procedures, hospital/insurance policies, and patient "
            "risk predictions. I'm not able to help with that request."
        )
    return {
        "msg": [reply],
        "chat_history": [AIMessage(content=reply)],
    }