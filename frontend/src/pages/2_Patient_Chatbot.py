import os
import sys
import uuid
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "frontend" / "src"))

from ui_styles import apply_hospital_theme
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


@st.cache_resource(show_spinner=False)
def get_graph():
    """Load the LangGraph application only when the user actually needs AI routing."""
    from agents.orchestrator.graph import graph
    return graph


@st.cache_resource(show_spinner=False)
def get_router_llm():
    """Create/cache the shared Groq router LLM on first AI use."""
    from agents.orchestrator.models import router_llm
    return router_llm

ICON_PATH = ROOT / "frontend" / "src" / "healthcare.png"
st.set_page_config(page_title="Patient Portal", page_icon=str(ICON_PATH), layout="wide")
apply_hospital_theme(ICON_PATH, "Patient Portal", "Appointments, symptom assessment, guidelines and hospital policy support")

st.sidebar.markdown("### Patient Services")
option = st.sidebar.radio("Choose a service", [
    "Dashboard", "Symptom Assessment", "Appointments", "Health Guidelines", "Policy Search"
])




def render_graph_chat(messages_key: str, placeholder: str, user_role: str):
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
    thread_key = f"{messages_key}_thread_id"
    count_key = f"{messages_key}_printed_count"
    if thread_key not in st.session_state:
        st.session_state[thread_key] = str(uuid.uuid4())
    if count_key not in st.session_state:
        st.session_state[count_key] = 0

    for msg in st.session_state[messages_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Optional voice input: audio is sent to Groq Whisper only after the user records it.
    voice_col, text_col = st.columns([1, 4])
    with voice_col:
        audio_value = st.audio_input("🎙️ Speak", key=f"voice_{messages_key}", sample_rate=16000)
    with text_col:
        prompt = st.chat_input(placeholder, key=f"input_{messages_key}")

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        processed_key = f"{messages_key}_voice_hash"
        if st.session_state.get(processed_key) != audio_hash:
            try:
                from voice.speech import translate_audio_to_english
                with st.spinner("Converting your speech to English text..."):
                    voice_prompt = translate_audio_to_english(
                        audio_bytes, getattr(audio_value, "name", None) or "voice.wav"
                    )
                st.session_state[processed_key] = audio_hash
                if voice_prompt:
                    st.info(f"Voice → English: {voice_prompt}")
                    prompt = voice_prompt
                else:
                    st.warning("I couldn't understand the recording. Please try again.")
                    return
            except Exception as exc:
                st.error(f"Voice-to-English processing failed: {exc}")
                return

    if not prompt:
        return
    st.session_state[messages_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking hospital services..."):
            try:
                result = get_graph().invoke(
                    {"user_input": prompt, "route": [], "msg": [], "user_role": user_role},
                    config={"configurable": {"thread_id": st.session_state[thread_key]}},
                )
                parts = result.get("msg", [])
                new_parts = parts[st.session_state[count_key]:]
                st.session_state[count_key] = len(parts)
                answer = "\n\n".join(new_parts).strip() if new_parts else "No response received."
                for prefix in ["[Triage]", "[Appointment]", "[Policy Search]", "[Health Guidelines]", "[Treatment Procedures]"]:
                    answer = answer.replace(prefix, "")
            except Exception as exc:
                answer = f"I couldn't complete that request right now. Please try again.\n\nTechnical detail: `{type(exc).__name__}`"
            st.markdown(answer)
    st.session_state[messages_key].append({"role": "assistant", "content": answer})

def render_chat(messages_key: str, placeholder: str, system_prompt: str, tool_mode: str | None = None):
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
    for msg in st.session_state[messages_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    voice_col, text_col = st.columns([1, 4])
    with voice_col:
        audio_value = st.audio_input("🎙️ Speak", key=f"voice_{messages_key}", sample_rate=16000)
    with text_col:
        prompt = st.chat_input(placeholder, key=f"input_{messages_key}")

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        processed_key = f"{messages_key}_voice_hash"
        if st.session_state.get(processed_key) != audio_hash:
            try:
                from voice.speech import translate_audio_to_english
                with st.spinner("Converting your speech to English text..."):
                    voice_prompt = translate_audio_to_english(
                        audio_bytes, getattr(audio_value, "name", None) or "voice.wav"
                    )
                st.session_state[processed_key] = audio_hash
                if voice_prompt:
                    st.info(f"Voice → English: {voice_prompt}")
                    prompt = voice_prompt
                else:
                    st.warning("I couldn't understand the recording. Please try again.")
                    return
            except Exception as exc:
                st.error(f"Voice-to-English processing failed: {exc}")
                return

    if not prompt:
        return

    st.session_state[messages_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking hospital information..."):
            try:
                if tool_mode == "appointment":
                    from agents.appointment.tools import APPOINTMENT_TOOLS
                    router_llm = get_router_llm()
                    messages = [SystemMessage(content=system_prompt)]
                    for m in st.session_state[messages_key][:-1]:
                        messages.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
                    messages.append(HumanMessage(content=prompt))
                    response = router_llm.bind_tools(APPOINTMENT_TOOLS).invoke(messages)
                    messages.append(response)
                    while response.tool_calls:
                        for call in response.tool_calls:
                            tool = next((t for t in APPOINTMENT_TOOLS if t.name == call["name"]), None)
                            output = tool.invoke(call["args"]) if tool else "Tool not found."
                            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
                        response = router_llm.bind_tools(APPOINTMENT_TOOLS).invoke(messages)
                        messages.append(response)
                    answer = response.content.strip() or "I could not complete that request."
                else:
                    messages = [SystemMessage(content=system_prompt)]
                    for m in st.session_state[messages_key][:-1]:
                        messages.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
                    messages.append(HumanMessage(content=prompt))
                    answer = get_router_llm().invoke(messages).content.strip()
            except Exception as exc:
                answer = f"I couldn't complete that request right now. Please try again.\n\nTechnical detail: `{type(exc).__name__}`"
            st.markdown(answer)
    st.session_state[messages_key].append({"role": "assistant", "content": answer})


if option == "Dashboard":
    st.markdown('<div class="section-title">Welcome to the Patient Portal</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Health Tips**")
        st.markdown("💧 Drink plenty of water  \n🏃 Exercise regularly  \n🥗 Eat a balanced diet  \n😴 Maintain a regular sleep schedule")
    with c2:
        st.markdown("**Hospital Updates**")
        st.markdown("🩺 Free health camp  \n🩸 Blood donation drive  \n🏥 New cardiology wing")
    with c3:
        st.markdown("**Emergency Contact**")
        st.markdown("**Emergency:** 108  \n**Reception:** +91-9876543210  \n**Email:** hospital@example.com")
    st.divider()
    st.markdown('<div class="section-title">AI Hospital Assistant</div>', unsafe_allow_html=True)
    render_graph_chat(
        "patient_dashboard_messages",
        "Ask about symptoms, appointments, guidelines or hospital policies...",
        "patient",
    )

elif option == "Symptom Assessment":
    st.markdown('<div class="section-title">Symptom Assessment</div>', unsafe_allow_html=True)
    st.caption("Describe symptoms for an initial urgency assessment. This is not a medical diagnosis.")
    render_chat(
        "patient_symptom_messages",
        "Describe your symptoms, duration and severity...",
        """You are a hospital triage assistant. Do not diagnose or prescribe. Identify emergency warning signs, suggest an appropriate department, explain the reason, recommend safe next steps, and ask only essential follow-up questions. If severe chest pain, difficulty breathing, stroke signs, loss of consciousness, seizures, heavy bleeding, severe allergic reaction, suicidal thoughts, severe head injury, blue lips, or rapidly worsening symptoms are described, clearly recommend emergency care. Format: Recommended Department, Emergency Status, Reason, Recommended Action, Follow-up Questions, Safety Note.""",
    )

elif option == "Appointments":
    st.markdown('<div class="section-title">Doctor Appointments</div>', unsafe_allow_html=True)
    st.caption("Choose a department and date to see live demo availability, then confirm a slot.")

    # SQL is the source of truth for live doctor and slot availability.
    from agents.appointment.database import Doctor, Slot, session_scope, initialize_database
    from sqlalchemy import select
    initialize_database()
    with session_scope() as session:
        active_doctors = session.scalars(select(Doctor).where(Doctor.active.is_(True)).order_by(Doctor.department, Doctor.doctor_name)).all()
        departments = sorted({doctor.department for doctor in active_doctors})
    c1, c2 = st.columns([1, 1])
    with c1:
        department = st.selectbox("Medical Department", departments, key="appointment_department")
    with c2:
        min_date = date.today()
        max_date = min_date + timedelta(days=59)
        selected_date = st.date_input("Appointment Date", value=min_date, min_value=min_date, max_value=max_date, key="appointment_date")

    selected_date_str = selected_date.isoformat()
    with session_scope() as session:
        db_rows = session.execute(
            select(Slot, Doctor).join(Doctor, Slot.doctor_id == Doctor.doctor_id).where(
                Doctor.active.is_(True), Doctor.department == department,
                Slot.date == selected_date, Slot.status == "Available"
            ).order_by(Slot.time_slot, Doctor.doctor_name)
        ).all()
    if selected_date == date.today():
        now = datetime.now().time()
        db_rows = [row for row in db_rows if datetime.strptime(row[0].time_slot, "%I:%M %p").time() > now]

    if not db_rows:
        st.info(f"No open {department} slots on {selected_date_str}. Use the AI assistant below to find the next available appointment.")
    else:
        st.markdown(f"**Available {department} slots on {selected_date_str}**")
        display = pd.DataFrame([{"Doctor": doctor.doctor_name, "Time": slot.time_slot, "Slot ID": slot.slot_id} for slot, doctor in db_rows])
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("#### Confirm a booking")
        with st.form("appointment_booking_form", clear_on_submit=True):
            patient_name = st.text_input("Patient Full Name", placeholder="e.g. Sai Dhanush")
            slot_options = [f"{doctor.doctor_name} — {slot.time_slot} — {slot.slot_id}" for slot, doctor in db_rows]
            chosen = st.selectbox("Choose Doctor & Time", slot_options)
            phone = st.text_input("Phone Number", placeholder="10-digit mobile number")
            submitted = st.form_submit_button("Confirm Appointment", type="primary", use_container_width=True)
            if submitted:
                if not patient_name.strip():
                    st.error("Please enter the patient's full name.")
                elif not phone.strip().isdigit() or len(phone.strip()) != 10:
                    st.error("Please enter a valid 10-digit phone number.")
                else:
                    chosen_time = chosen.split(" — ")[1]
                    from agents.appointment.tools import book_hospital_appointment
                    result = book_hospital_appointment.invoke({
                        "patient_name": patient_name.strip(),
                        "department": department,
                        "time_slot": chosen_time,
                        "date": selected_date_str,
                        "patient_phone": phone.strip(),
                    })
                    if str(result).startswith("SUCCESS"):
                        st.success(result)
                        st.rerun()
                    else:
                        st.error(result)

    st.divider()
    st.markdown("#### Manage an existing appointment")
    with st.expander("Cancel or reschedule"):
        action = st.radio("Action", ["Cancel", "Reschedule"], horizontal=True)
        appointment_id = st.text_input("Appointment ID", placeholder="APT-20260929-ABC123")
        patient_name_manage = st.text_input("Patient Full Name", key="manage_name")
        if action == "Reschedule":
            new_date = st.date_input("New Date", value=min_date, min_value=min_date, max_value=max_date, key="reschedule_date")
            with session_scope() as session:
                res_slots = session.execute(
                    select(Slot, Doctor).join(Doctor, Slot.doctor_id == Doctor.doctor_id).where(
                        Doctor.active.is_(True), Doctor.department == department, Slot.date == new_date, Slot.status == "Available"
                    ).order_by(Slot.time_slot, Doctor.doctor_name)
                ).all()
            if new_date == date.today():
                now = datetime.now().time()
                res_slots = [row for row in res_slots if datetime.strptime(row[0].time_slot, "%I:%M %p").time() > now]
            res_options = [f"{slot.time_slot} — {doctor.doctor_name}" for slot, doctor in res_slots]
            new_choice = st.selectbox("New Time", res_options or ["No open slots"], disabled=not bool(res_options))
        if st.button("Submit Request", type="secondary"):
            if not appointment_id.strip() or not patient_name_manage.strip():
                st.error("Appointment ID and patient name are required.")
            elif action == "Cancel":
                from agents.appointment.tools import cancel_hospital_appointment
                result = cancel_hospital_appointment.invoke({"appointment_id": appointment_id.strip(), "patient_name": patient_name_manage.strip()})
                (st.success if str(result).startswith("SUCCESS") else st.error)(result)
                if str(result).startswith("SUCCESS"): st.rerun()
            else:
                if not res_options:
                    st.error("No replacement slot is available for the selected date.")
                else:
                    new_time = new_choice.split(" — ")[0]
                    from agents.appointment.tools import reschedule_hospital_appointment
                    result = reschedule_hospital_appointment.invoke({"appointment_id": appointment_id.strip(), "patient_name": patient_name_manage.strip(), "new_date": new_date.isoformat(), "new_time_slot": new_time})
                    (st.success if str(result).startswith("SUCCESS") else st.error)(result)
                    if str(result).startswith("SUCCESS"): st.rerun()

    st.divider()
    st.markdown("#### AI Appointment Assistant")
    render_chat(
        "patient_appointment_messages",
        "Ask: which doctors are available, find the next cardiology slot, book, cancel or reschedule...",
        """You are an accurate hospital appointment assistant. Current date is today's date. Use the appointment tools. For a booking, collect patient name, department, date and time; ALWAYS check availability before booking. If the user asks which doctors or slots are available without a date, use search_doctors_directory or find_next_available_slots. Accept dates such as 28-09-2026 and normalize them. Never claim a booking succeeded unless the booking tool returns SUCCESS. If a date is in the past, say so and offer the next available dates.""",
        tool_mode="appointment",
    )

elif option == "Health Guidelines":
    st.markdown('<div class="section-title">Health Guidelines</div>', unsafe_allow_html=True)
    st.caption("Answers are grounded in the hospital's indexed guideline document.")
    render_graph_chat(
        "patient_guideline_messages",
        "Ask about the hospital health guidelines...",
        "patient",
    )

elif option == "Policy Search":
    st.markdown('<div class="section-title">Hospital & Insurance Policy Search</div>', unsafe_allow_html=True)
    st.caption("Ask about hospital and insurance policies available in the indexed policy dataset.")
    render_graph_chat(
        "patient_policy_messages",
        "Ask about insurance coverage, hospital policies or administrative rules...",
        "patient",
    )
