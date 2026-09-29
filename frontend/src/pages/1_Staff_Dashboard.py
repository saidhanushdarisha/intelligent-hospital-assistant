import streamlit as st
import os
import sys
import uuid
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "frontend" / "src"))

from ui_styles import apply_hospital_theme
from agents.orchestrator.graph import graph

ICON_PATH = ROOT / "frontend" / "src" / "healthcare.png"
st.set_page_config(page_title="Staff Dashboard", page_icon=str(ICON_PATH), layout="wide")
apply_hospital_theme(ICON_PATH, "Staff Dashboard", "Clinical risk intelligence, triage, knowledge and appointment operations")

# Initialize login state if not present
if "staff_logged_in" not in st.session_state:
    st.session_state.staff_logged_in = False

if not st.session_state.staff_logged_in:
    # --- STAFF LOGIN VIEW ---
    with open(ICON_PATH, "rb") as f:
        icon_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"<h2 style='text-align: center; padding-bottom: 10px; border-bottom: 1px solid #D6D6D9;'><img src='data:image/png;base64,{icon_b64}' style='height: 32px; vertical-align: middle; margin-right: 8px;'>Staff Portal Login</h2>", unsafe_allow_html=True)
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        with st.form("login_form", border=True):
            st.markdown("""
                <h3 style='text-align: center; margin-top: 0; margin-bottom: 0px;'>Sign In</h3>
                <p style='text-align: center; color: #555; font-size: 14px; margin-top: 0; margin-bottom: 5px;'>Enter your credentials to access the secure staff dashboard.</p>
                <hr style='margin-top: 0; margin-bottom: 15px; border: 0; border-top: 1px solid #E6E6E6;'>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="e.g., admin")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            st.write("") # Small spacer
            
            submit_button = st.form_submit_button("Login", use_container_width=True, type="primary")
            
            if submit_button:
                staff_username = os.getenv("STAFF_USERNAME", "admin")
                staff_password = os.getenv("STAFF_PASSWORD", "1234")
                if username == staff_username and password == staff_password:
                    st.session_state.staff_logged_in = True
                    st.success("Login Successful! Loading dashboard...")
                    st.rerun()
                elif not username or not password:
                    st.warning("Please enter both username and password.")
                else:
                    st.error("Invalid Credentials. Please try again.")

else:
    # --- STAFF DASHBOARD VIEW ---
    st.sidebar.header("Staff Dashboard")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Readmission Risk",
            "High-Risk Patients",
            "Health Guidelines",
            "Treatment Procedures",
            "Triage",
            "Appointment",
            "Database Admin",
            "Policy Search"
        ]
    )

    # Push the logout button to the bottom of the sidebar
    st.sidebar.write("")
    st.sidebar.write("")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.staff_logged_in = False
        st.rerun()

    # Load and evaluate registries once (cached) for stats and tab views
    import pandas as pd
    from agents.readmission_risk.tools.tools import predict_readmission
    from agents.high_risk_patients.tools.tools import predict_highrisk

    @st.cache_data
    def get_readmission_data():
        df = pd.read_csv("src/agents/readmission_risk/data/patients.csv")
        predictions = []
        confidences = []
        for _, row in df.iterrows():
            res = predict_readmission.invoke({
                "Age": int(row["Age"]),
                "Gender": row["Gender"],
                "Condition": row["Condition"],
                "Procedure": row["Procedure"],
                "Cost": float(row["Cost"]),
                "Length_of_Stay": int(row["Length_of_Stay"]),
                "Outcome": row["Outcome"],
                "Satisfaction": int(row["Satisfaction"])
            })
            predictions.append(res["Readmission"])
            confidences.append(f"{res['Confidence']}%")
        df["Readmission Prediction"] = predictions
        df["Confidence"] = confidences
        return df

    @st.cache_data
    def get_highrisk_data():
        df = pd.read_csv("src/agents/high_risk_patients/data/patients.csv")
        predictions = []
        confidences = []
        for _, row in df.iterrows():
            res = predict_highrisk.invoke({
                "Respiratory_Rate": int(row["Respiratory_Rate"]),
                "Oxygen_Saturation": int(row["Oxygen_Saturation"]),
                "O2_Scale": int(row["O2_Scale"]),
                "Systolic_BP": int(row["Systolic_BP"]),
                "Heart_Rate": int(row["Heart_Rate"]),
                "Temperature": float(row["Temperature"]),
                "Consciousness": row["Consciousness"],
                "On_Oxygen": int(row["On_Oxygen"])
            })
            predictions.append(res["Risk_Level"])
            confidences.append(f"{res['Risk_Score']}%")
        df["Deterioration Risk"] = predictions
        df["Confidence"] = confidences
        return df


    if page == "Dashboard":
        df_readm_stats = get_readmission_data()
        df_highrisk_stats = get_highrisk_data()

        # Calculate statistics dynamically
        unique_names = set(df_readm_stats["Name"].dropna().unique().tolist() + df_highrisk_stats["Name"].dropna().unique().tolist())
        total_patients_count = len(unique_names)

        high_risk_count = len(df_highrisk_stats[df_highrisk_stats["Deterioration Risk"].str.lower() == "high"])
        readmission_count = len(df_readm_stats[df_readm_stats["Readmission Prediction"].str.lower() == "yes"])

        try:
            from agents.appointment.database import Appointment, session_scope
            from sqlalchemy import select, func
            with session_scope() as session:
                appointments_count = session.scalar(select(func.count()).select_from(Appointment)) or 0
        except Exception:
            appointments_count = 0

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"<div style='text-align: center;'><span style='color: #666; font-size: 14px;'>Total Patients</span><br><span style='font-weight: bold; font-size: 32px; color: #111;'>{total_patients_count}</span></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='text-align: center;'><span style='color: #666; font-size: 14px;'>High-Risk Patients</span><br><span style='font-weight: bold; font-size: 32px; color: #111;'>{high_risk_count}</span></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='text-align: center;'><span style='color: #666; font-size: 14px;'>Readmission Cases</span><br><span style='font-weight: bold; font-size: 32px; color: #111;'>{readmission_count}</span></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div style='text-align: center;'><span style='color: #666; font-size: 14px;'>Appointments</span><br><span style='font-weight: bold; font-size: 32px; color: #111;'>{appointments_count}</span></div>", unsafe_allow_html=True)

        st.divider()

        st.subheader("AI Assistant")

        if "dashboard_thread_id" not in st.session_state:
            st.session_state.dashboard_thread_id = str(uuid.uuid4())
        if "dashboard_printed_count" not in st.session_state:
            st.session_state.dashboard_printed_count = 0

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask AI Assistant")

        if prompt:
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                config = {"configurable": {"thread_id": st.session_state.dashboard_thread_id}}
                result = graph.invoke(
                    {"user_input": prompt, "route": [], "msg": [], "user_role": "staff"},
                    config=config,
                )

                new_messages = result["msg"][st.session_state.dashboard_printed_count:]
                st.session_state.dashboard_printed_count = len(result["msg"])
                response = "\n\n".join(new_messages) if new_messages else "No response received."

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

    elif page == "Readmission Risk":
        st.header("Readmission Risk Prediction")
        st.caption("Estimate 30-day patient readmission probability using clinical and demographic data.")
        st.divider()

        df = get_readmission_data()

        st.subheader("Patient Readmission Registry")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

        # Add interactive prediction form
        st.subheader("Predict Custom Patient Risk")
        with st.form("predict_form", border=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=50)
                gender = st.selectbox("Gender", ["Female", "Male"])
                condition = st.selectbox("Primary Condition", [
                    "Allergic Reaction", "Appendicitis", "Cancer", "Childbirth", "Diabetes",
                    "Fractured Arm", "Fractured Leg", "Heart Attack", "Heart Disease",
                    "Hypertension", "Kidney Stones", "Osteoarthritis", "Prostate Cancer",
                    "Respiratory Infection", "Stroke"
                ])
                procedure = st.selectbox("Procedure Administered", [
                    "Angioplasty", "Antibiotics and Rest", "Appendectomy",
                    "CT Scan and Medication", "Cardiac Catheterization",
                    "Cast and Physical Therapy", "Delivery and Postnatal Care",
                    "Epinephrine Injection", "Insulin Therapy", "Lithotripsy",
                    "Medication and Counseling", "Physical Therapy and Pain Management",
                    "Radiation Therapy", "Surgery and Chemotherapy", "X-Ray and Splint"
                ])
            with col2:
                cost = st.number_input("Admission Cost ($)", min_value=0.0, value=5000.0, step=500.0)
                los = st.number_input("Length of Stay (Days)", min_value=1, max_value=100, value=4)
                outcome = st.selectbox("Discharge Status / Outcome", ["Recovered", "Stable"])
                satisfaction = st.slider("Patient Satisfaction Rating (1-5)", min_value=1, max_value=5, value=4)

            predict_btn = st.form_submit_button("Predict Risk", type="primary")

            if predict_btn:
                res = predict_readmission.invoke({
                    "Age": int(age),
                    "Gender": gender,
                    "Condition": condition,
                    "Procedure": procedure,
                    "Cost": float(cost),
                    "Length_of_Stay": int(los),
                    "Outcome": outcome,
                    "Satisfaction": int(satisfaction)
                })

                st.write("")
                if res["Readmission"] == "Yes":
                    st.error(f"**High Risk of Readmission** (Confidence: {res['Confidence']}%)")
                else:
                    st.success(f"**Low Risk of Readmission** (Confidence: {res['Confidence']}%)")

    elif page == "High-Risk Patients":
        st.header("High-Risk Patients Detection")
        st.caption("Classify patient deterioration risk categories (Low, Medium, High, Normal) using physiological vital signs.")
        st.divider()

        df = get_highrisk_data()

        st.subheader("Patient Deterioration Registry")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

        # Add interactive prediction form
        st.subheader("Evaluate Custom Patient Deterioration Risk")
        with st.form("highrisk_form", border=True):
            col1, col2 = st.columns(2)
            with col1:
                rr = st.number_input("Respiratory Rate (breaths/min)", min_value=0, max_value=60, value=18)
                o2_sat = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=96)
                o2_scale = st.selectbox("O2 Scale (1 or 2)", [1, 2], index=0)
                systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0, max_value=300, value=120)
            with col2:
                hr = st.number_input("Heart Rate (bpm)", min_value=0, max_value=250, value=75)
                temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
                consciousness = st.selectbox("Consciousness Level (A=Alert, V=Voice, P=Pain, U=Unresponsive, C=Confused)", ["A", "C", "P", "U", "V"])
                on_oxygen = st.selectbox("On Supplemental Oxygen?", ["No (0)", "Yes (1)"])

            predict_btn = st.form_submit_button("Assess Risk", type="primary")

            if predict_btn:
                ox_val = 1 if "Yes" in on_oxygen else 0
                res = predict_highrisk.invoke({
                    "Respiratory_Rate": int(rr),
                    "Oxygen_Saturation": int(o2_sat),
                    "O2_Scale": int(o2_scale),
                    "Systolic_BP": int(systolic_bp),
                    "Heart_Rate": int(hr),
                    "Temperature": float(temp),
                    "Consciousness": consciousness,
                    "On_Oxygen": ox_val
                })

                st.write("")
                risk_lvl = res["Risk_Level"]
                score = res["Risk_Score"]
                
                if risk_lvl == "High":
                    st.error(f"**High Deterioration Risk Detected** (Confidence: {score}%)")
                elif risk_lvl == "Medium":
                    st.warning(f"**Medium Deterioration Risk Detected** (Confidence: {score}%)")
                elif risk_lvl == "Low":
                    st.info(f"**Low Deterioration Risk** (Confidence: {score}%)")
                else:
                    st.success(f"**Normal/Healthy Vital Signs** (Confidence: {score}%)")

    elif page == "Health Guidelines":
        st.header("Health Guidelines")
        st.caption("Ask questions and search standard medical health guidelines.")
        st.divider()

        from agents.orchestrator.router import health_guidelines_db
        from agents.orchestrator.models import router_llm
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        st.subheader("AI Assistant")

        if "guidelines_messages" not in st.session_state:
            st.session_state.guidelines_messages = []

        for msg in st.session_state.guidelines_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask Health Guidelines Assistant")

        if prompt:
            st.session_state.guidelines_messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                if health_guidelines_db is None:
                    response = "The health guidelines database is currently unavailable."
                else:
                    docs = health_guidelines_db.similarity_search(prompt, k=2)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Build system message with memory/greeting rules
                    system_instructions = """You are an AI Hospital Assistant specializing in standard medical health guidelines for medical staff.

Rules:
1. Greet the user warmly and handle small talk or general questions politely if the user greets you or asks general capability questions.
2. If the user asks about other services or actions available in the staff portal (like booking appointments, running risk predictions, performing triage, searching procedures, or policy lookup):
- Explain politely that you are the Health Guidelines assistant and cannot perform this task here.
- Direct them to the correct section of the sidebar navigation:
  * Booking/canceling appointments: select "Appointment" in the sidebar navigation.
  * Running triage assessments: select "Triage" in the sidebar navigation.
  * Readmission risk predictions: select "Readmission Risk" in the sidebar navigation.
  * Deterioration risk monitoring: select "High-Risk Patients" in the sidebar navigation.
  * Treatment procedures: select "Treatment Procedures" in the sidebar navigation.
  * Hospital/insurance policy search: select "Policy Search" in the sidebar navigation.
3. If the user asks a medical or health guideline question, explain the guidelines in a friendly, professional, and refined conversational tone, structuring the response clearly (e.g. using formatting or bullet points) so it is easy to read. Stay strictly grounded in the provided DOCUMENT EXCERPTS. Do NOT use your own medical knowledge or invent guidelines not in the excerpts.
4. If the answer cannot be found in the document excerpts, reply exactly:
"I couldn't find this information in the hospital documentation."
5. Use the CONVERSATION HISTORY to keep track of context across turns, allowing the user to ask follow-up questions.
"""
                    messages = [SystemMessage(content=system_instructions)]

                    # Append prior chat memory
                    for msg in st.session_state.guidelines_messages[:-1]:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))

                    # Append current user query with RAG context
                    user_content = f"""DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{prompt}
"""
                    messages.append(HumanMessage(content=user_content))

                    # Invoke LLM to refine response
                    response = router_llm.invoke(messages).content.strip()

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.guidelines_messages.append(
                {"role": "assistant", "content": response}
            )

    elif page == "Treatment Procedures":
        st.header("Treatment Procedures")
        st.caption("Ask questions and search hospital treatment procedures and protocols.")
        st.divider()

        from agents.orchestrator.router import treatment_procedures_db
        from agents.orchestrator.models import router_llm
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        st.subheader("AI Assistant")

        if "procedures_messages" not in st.session_state:
            st.session_state.procedures_messages = []

        for msg in st.session_state.procedures_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask Treatment Procedures Assistant")

        if prompt:
            st.session_state.procedures_messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                if treatment_procedures_db is None:
                    response = "The treatment procedures database is currently unavailable."
                else:
                    docs = treatment_procedures_db.similarity_search(prompt, k=2)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Build system message with memory/greeting/other services rules
                    system_instructions = """You are an AI Hospital Assistant specializing in standard hospital treatment procedures and protocols for medical staff.

Rules:
1. Greet the user warmly and handle small talk or general questions politely if the user greets you or asks general capability questions.
2. If the user asks about other services or actions available in the staff portal (like booking appointments, running risk predictions, performing triage, guidelines search, or policy lookup):
- Explain politely that you are the Treatment Procedures assistant and cannot perform this task here.
- Direct them to the correct section of the sidebar navigation:
  * Booking/canceling appointments: select "Appointment" in the sidebar navigation.
  * Running triage assessments: select "Triage" in the sidebar navigation.
  * Readmission risk predictions: select "Readmission Risk" in the sidebar navigation.
  * Deterioration risk monitoring: select "High-Risk Patients" in the sidebar navigation.
  * Medical guidelines search: select "Health Guidelines" in the sidebar navigation.
  * Hospital/insurance policy search: select "Policy Search" in the sidebar navigation.
3. If the user asks a treatment procedure or protocol question, explain the steps clearly and concisely (using numbered steps if there is a sequence) in a friendly, professional, and refined conversational tone, structuring the response clearly so it is easy to read. Stay strictly grounded in the provided DOCUMENT EXCERPTS. Do NOT use your own medical knowledge or invent steps not in the excerpts.
4. If the answer cannot be found in the document excerpts, reply exactly:
"I couldn't find this information in the hospital documentation."
5. Use the CONVERSATION HISTORY to keep track of context across turns, allowing the user to ask follow-up questions.
"""
                    messages = [SystemMessage(content=system_instructions)]

                    # Append prior chat memory
                    for msg in st.session_state.procedures_messages[:-1]:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))

                    # Append current user query with RAG context
                    user_content = f"""DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{prompt}
"""
                    messages.append(HumanMessage(content=user_content))

                    # Invoke LLM to refine response
                    response = router_llm.invoke(messages).content.strip()

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.procedures_messages.append(
                {"role": "assistant", "content": response}
            )

    elif page == "Triage":
        st.header("Triage")
        st.caption("Assess patient symptoms and recommend appropriate clinical departments.")
        st.divider()

        from agents.orchestrator.models import router_llm
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        st.subheader("AI Assistant")

        if "triage_messages" not in st.session_state:
            st.session_state.triage_messages = []

        for msg in st.session_state.triage_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Describe the patient's symptoms here...")

        if prompt:
            st.session_state.triage_messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                system_instructions = """You are an AI Hospital Triage Assistant for medical staff.

Your role is to perform symptom triage, NOT diagnose diseases.

Objectives:
1. Understand the patient's symptoms.
2. Identify emergency warning signs.
3. Recommend the most appropriate hospital department.
4. Explain your reasoning.
5. Recommend safe next steps.
6. Ask only essential follow-up questions if information is missing.

If the user asks about other services or actions available in the staff portal (like booking appointments, running risk predictions, guidelines search, procedures search, or policy lookup):
- Explain politely that you are the Triage assistant and cannot perform this task here.
- Direct them to the correct section of the sidebar navigation:
  * Booking/canceling appointments: select "Appointment" in the sidebar navigation.
  * Medical guidelines search: select "Health Guidelines" in the sidebar navigation.
  * Readmission risk predictions: select "Readmission Risk" in the sidebar navigation.
  * Deterioration risk monitoring: select "High-Risk Patients" in the sidebar navigation.
  * Treatment procedures: select "Treatment Procedures" in the sidebar navigation.
  * Hospital/insurance policy search: select "Policy Search" in the sidebar navigation.

Never:
- Diagnose diseases.
- Prescribe medication.
- Recommend drug dosages.
- Invent information.
- Ignore emergency symptoms.

Use the CONVERSATION HISTORY to keep track of context across turns, allowing the user to ask follow-up questions or provide more details.

Please evaluate:
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

Please format your response clearly using headers and bullet points. Use standard headings:
- **Recommended Department**
- **Emergency Status**
- **Reasoning & Action Plan**
- **Follow-up Questions** (if details are missing)
- **Safety Note**: This is a triage assessment only and is not a medical diagnosis.
"""
                messages = [SystemMessage(content=system_instructions)]

                # Append prior chat memory
                for msg in st.session_state.triage_messages[:-1]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))

                # Append current user query
                messages.append(HumanMessage(content=prompt))

                # Invoke LLM to perform triage reasoning
                response = router_llm.invoke(messages).content.strip()

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.triage_messages.append(
                {"role": "assistant", "content": response}
            )

    elif page == "Appointment":
        st.header("Appointment Management")
        st.caption("SQL-backed doctor schedules, availability and patient appointments.")
        st.divider()

        from datetime import datetime, date as dt_date
        from sqlalchemy import select
        from agents.appointment.database import Appointment, Doctor, Slot, initialize_database, session_scope
        from agents.appointment.tools import APPOINTMENT_TOOLS, book_hospital_appointment
        from agents.orchestrator.router import appointment_tools
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

        initialize_database()

        # ------------------------- SQL-backed quick booking -------------------------
        st.subheader("Schedule Appointment")
        with st.container(border=True):
            with session_scope() as session:
                active_doctors = session.scalars(select(Doctor).where(Doctor.active.is_(True)).order_by(Doctor.department, Doctor.doctor_name)).all()
                departments = sorted({d.department for d in active_doctors})

                patient_name_input = st.text_input("Patient Name", placeholder="Enter full name", key="staff_patient_name")
                patient_phone_input = st.text_input("Patient Phone (optional)", placeholder="Enter phone number", key="staff_patient_phone")
                selected_dept = st.selectbox("Department", ["Select Department..."] + departments, key="staff_department")

                if selected_dept != "Select Department...":
                    rows = session.execute(
                        select(Slot, Doctor).join(Doctor, Slot.doctor_id == Doctor.doctor_id).where(
                            Doctor.active.is_(True), Doctor.department == selected_dept,
                            Slot.status == "Available", Slot.date >= dt_date.today()
                        ).order_by(Slot.date, Slot.time_slot, Doctor.doctor_name)
                    ).all()
                    available_dates = sorted({slot.date.isoformat() for slot, _ in rows})
                    selected_date = st.selectbox("Appointment Date", ["Select Date..."] + available_dates, key="staff_date")
                else:
                    rows = []
                    selected_date = st.selectbox("Appointment Date", ["Please select a department first..."], disabled=True, key="staff_date_disabled")

                if selected_dept != "Select Department..." and selected_date not in ["Select Date...", "Please select a department first..."]:
                    available_time_slots = sorted({slot.time_slot for slot, _ in rows if slot.date.isoformat() == selected_date})
                    selected_slot = st.selectbox("Time Slot", ["Select Time Slot..."] + available_time_slots, key="staff_time_slot")
                else:
                    selected_slot = st.selectbox("Time Slot", ["Please select a date first..."], disabled=True, key="staff_time_slot_disabled")

            book_btn = st.button("Confirm Booking", type="primary", use_container_width=True)
            if book_btn:
                if not patient_name_input.strip():
                    st.error("Please enter the patient's name.")
                elif selected_dept == "Select Department...":
                    st.error("Please select a department.")
                elif selected_date in ["Select Date...", "Please select a department first..."]:
                    st.error("Please select an appointment date.")
                elif selected_slot in ["Select Time Slot...", "Please select a date first..."]:
                    st.error("Please select an available time slot.")
                else:
                    result = book_hospital_appointment.invoke({
                        "patient_name": patient_name_input.strip(),
                        "department": selected_dept,
                        "time_slot": selected_slot,
                        "date": selected_date,
                        "patient_phone": patient_phone_input.strip(),
                    })
                    if "SUCCESS" in result:
                        st.success(result)
                        st.rerun()
                    else:
                        st.error(result)

        st.divider()

        # ------------------------- Live SQL tables -------------------------
        tab1, tab2, tab3 = st.tabs(["Doctors", "Available Slots", "Appointments"])
        with tab1:
            with session_scope() as session:
                doctors = session.scalars(select(Doctor).order_by(Doctor.active.desc(), Doctor.department, Doctor.doctor_name)).all()
                doctor_rows = [{
                    "Doctor ID": d.doctor_id, "Doctor": d.doctor_name, "Department": d.department,
                    "Fee (₹)": d.consultation_fee, "Working Days": d.working_days,
                    "Status": "Active" if d.active else "Inactive"
                } for d in doctors]
            st.dataframe(pd.DataFrame(doctor_rows), use_container_width=True, hide_index=True)

        with tab2:
            with session_scope() as session:
                slots = session.execute(
                    select(Slot, Doctor).join(Doctor, Slot.doctor_id == Doctor.doctor_id)
                    .where(Slot.date >= dt_date.today()).order_by(Slot.date, Slot.time_slot, Doctor.doctor_name)
                ).all()
                slot_rows = [{
                    "Slot ID": slot.slot_id, "Date": slot.date.isoformat(), "Time": slot.time_slot,
                    "Doctor": doctor.doctor_name, "Department": doctor.department, "Status": slot.status
                } for slot, doctor in slots]
            st.dataframe(pd.DataFrame(slot_rows), use_container_width=True, hide_index=True)
            if st.button("Generate/Refresh Next 60 Days of Slots", key="refresh_slots"):
                from agents.appointment.database import ensure_future_slots
                ensure_future_slots(60)
                st.success("Future slots have been generated from the active doctors in the database.")
                st.rerun()

        with tab3:
            with session_scope() as session:
                appointments = session.scalars(select(Appointment).order_by(Appointment.date.desc(), Appointment.created_at.desc())).all()
                appt_rows = [{
                    "Appointment ID": a.appointment_id, "Patient": a.patient_name, "Phone": a.patient_phone,
                    "Department": a.department, "Doctor": a.doctor_name, "Date": a.date.isoformat(),
                    "Time": a.time_slot, "Status": a.status
                } for a in appointments]
            st.dataframe(pd.DataFrame(appt_rows), use_container_width=True, hide_index=True)
            confirmed_ids = [r["Appointment ID"] for r in appt_rows if r["Status"] == "Confirmed"]
            if confirmed_ids:
                cancel_id = st.selectbox("Cancel confirmed appointment", ["Select appointment..."] + confirmed_ids, key="admin_cancel_id")
                if st.button("Cancel Selected Appointment", key="admin_cancel_btn") and cancel_id != "Select appointment...":
                    with session_scope() as session:
                        appt = session.get(Appointment, cancel_id)
                        if appt and appt.status == "Confirmed":
                            slot = session.get(Slot, appt.slot_id)
                            if slot:
                                slot.status = "Available"
                            appt.status = "Cancelled"
                            st.success(f"Appointment {cancel_id} cancelled. The slot is available again.")
                            st.rerun()

        st.divider()
        st.subheader("AI Appointment Assistant")
        st.caption("The AI assistant uses the same SQL database as the forms above.")
        if "appointment_messages" not in st.session_state:
            st.session_state.appointment_messages = []
        for msg in st.session_state.appointment_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        prompt = st.chat_input("Ask Appointment Assistant (book, reschedule, cancel, check availability)")
        if prompt:
            st.session_state.appointment_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.spinner("Thinking..."):
                current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
                system_instructions = f"""You are an AI Hospital Appointment Assistant for medical staff.
Current time: {current_time}.

Objectives:
1. Help staff book, cancel, reschedule, or search appointments using the provided SQL-backed tools.
2. ALWAYS use check_doctor_availability before confirming a booking.
3. Gather patient name, department, time_slot, and date for booking.
4. The SQL database is the source of truth. Never claim a booking exists unless a tool confirms it.
5. Do not invent doctors, slots, appointment IDs, or availability.
"""
                messages = [SystemMessage(content=system_instructions)]
                for msg in st.session_state.appointment_messages[:-1]:
                    messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
                messages.append(HumanMessage(content=prompt))
                response = appointment_tools.invoke(messages)
                messages.append(response)
                while response.tool_calls:
                    for tool_call in response.tool_calls:
                        selected_tool = next((t for t in APPOINTMENT_TOOLS if t.name == tool_call["name"]), None)
                        tool_output = selected_tool.invoke(tool_call["args"]) if selected_tool else "Error: Tool not found."
                        messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                    response = appointment_tools.invoke(messages)
                    messages.append(response)
                response_text = response.content
            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.appointment_messages.append({"role": "assistant", "content": response_text})

    elif page == "Database Admin":
        st.header("Database Administration")
        st.caption("Manage doctors and appointment slots directly in the SQL database.")
        st.info("This page is protected by the Staff Portal login. Changes made here are immediately used by the patient chatbot and appointment assistant.")

        from agents.appointment.database import Appointment, Doctor, Slot, ensure_future_slots, session_scope
        from sqlalchemy import select
        import uuid as _uuid

        admin_tab1, admin_tab2, admin_tab3 = st.tabs(["Add / Manage Doctors", "Create Slots", "Database Overview"])

        with admin_tab1:
            st.subheader("Add Doctor")
            with st.form("add_doctor_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    new_doctor_name = st.text_input("Doctor Name", placeholder="Dr. John Doe")
                    new_department = st.text_input("Department", placeholder="Cardiology")
                with c2:
                    new_fee = st.number_input("Consultation Fee (₹)", min_value=0.0, value=500.0, step=50.0)
                    new_working_days = st.text_input("Working Days", value="Mon-Fri", help="Example: Mon-Fri or Tue-Sun")
                add_doctor = st.form_submit_button("Add Doctor", type="primary", use_container_width=True)
                if add_doctor:
                    if not new_doctor_name.strip() or not new_department.strip():
                        st.error("Doctor name and department are required.")
                    else:
                        doctor_id = f"DOC-{_uuid.uuid4().hex[:8].upper()}"
                        with session_scope() as session:
                            session.add(Doctor(doctor_id=doctor_id, doctor_name=new_doctor_name.strip(), department=new_department.strip(), consultation_fee=float(new_fee), working_days=new_working_days.strip() or "Mon-Fri", active=True))
                        ensure_future_slots(60)
                        st.success(f"Doctor {new_doctor_name.strip()} added to the database with ID {doctor_id}.")
                        st.rerun()

            st.subheader("Manage Existing Doctors")
            with session_scope() as session:
                doctors = session.scalars(select(Doctor).order_by(Doctor.department, Doctor.doctor_name)).all()
                options = {f"{d.doctor_name} — {d.department} ({d.doctor_id})": d.doctor_id for d in doctors}
            selected_label = st.selectbox("Select Doctor", ["Select Doctor..."] + list(options), key="manage_doctor")
            if selected_label != "Select Doctor...":
                selected_id = options[selected_label]
                with session_scope() as session:
                    doctor = session.get(Doctor, selected_id)
                    current = {"name": doctor.doctor_name, "department": doctor.department, "fee": doctor.consultation_fee, "days": doctor.working_days, "active": doctor.active}
                with st.form(f"edit_{selected_id}"):
                    name = st.text_input("Doctor Name", value=current["name"])
                    department = st.text_input("Department", value=current["department"])
                    fee = st.number_input("Consultation Fee (₹)", min_value=0.0, value=float(current["fee"]), step=50.0)
                    days = st.text_input("Working Days", value=current["days"])
                    active = st.checkbox("Active", value=current["active"])
                    save = st.form_submit_button("Save Doctor Changes", type="primary")
                    if save:
                        with session_scope() as session:
                            doctor = session.get(Doctor, selected_id)
                            doctor.doctor_name = name.strip()
                            doctor.department = department.strip()
                            doctor.consultation_fee = float(fee)
                            doctor.working_days = days.strip() or "Mon-Fri"
                            doctor.active = bool(active)
                        ensure_future_slots(60)
                        st.success("Doctor record updated in the database.")
                        st.rerun()

        with admin_tab2:
            st.subheader("Create / Refresh Appointment Slots")
            st.write("Slots are generated from active doctors' department and working-day settings and stored in the SQL database.")
            if st.button("Generate Next 60 Days", type="primary"):
                ensure_future_slots(60)
                st.success("Next 60 days of slots are now available in the database.")
                st.rerun()
            with session_scope() as session:
                active_doctors = session.scalars(select(Doctor).where(Doctor.active.is_(True)).order_by(Doctor.department, Doctor.doctor_name)).all()
            if active_doctors:
                st.subheader("Add a Custom Slot")
                doctor_options = {f"{d.doctor_name} — {d.department}": d.doctor_id for d in active_doctors}
                doctor_label = st.selectbox("Doctor", list(doctor_options), key="custom_slot_doctor")
                slot_date = st.date_input("Date", value=dt_date.today(), key="custom_slot_date")
                slot_time = st.text_input("Time", value="09:00 AM", key="custom_slot_time")
                if st.button("Add Custom Slot", key="custom_slot_btn"):
                    with session_scope() as session:
                        doctor_id = doctor_options[doctor_label]
                        exists = session.scalar(select(Slot).where(Slot.doctor_id == doctor_id, Slot.date == slot_date, Slot.time_slot == slot_time.strip()))
                        if exists:
                            st.error("That doctor already has this slot.")
                        else:
                            session.add(Slot(slot_id=f"SLT-{slot_date.strftime('%Y%m%d')}-{_uuid.uuid4().hex[:8].upper()}", doctor_id=doctor_id, date=slot_date, time_slot=slot_time.strip(), status="Available"))
                            st.success("Custom slot added to the database.")
                            st.rerun()

        with admin_tab3:
            with session_scope() as session:
                doctor_count = len(session.scalars(select(Doctor)).all())
                slot_count = len(session.scalars(select(Slot)).all())
                appointment_count = len(session.scalars(select(Appointment)).all())
                confirmed_count = len(session.scalars(select(Appointment).where(Appointment.status == "Confirmed")).all())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Doctors", doctor_count)
            c2.metric("Slots", slot_count)
            c3.metric("Appointments", appointment_count)
            c4.metric("Confirmed", confirmed_count)
            st.code("DATABASE_URL = PostgreSQL URL if configured; otherwise local SQLite: db/hospital.db", language="text")

    elif page == "Policy Search":
        st.header("Policy Search")
        st.caption("Ask questions and search hospital policies, guidelines, and compliance documents.")
        st.divider()

        from agents.orchestrator.router import policy_db
        from agents.policy_search.models import policy_search_llm
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        st.subheader("AI Assistant")

        if "policy_messages" not in st.session_state:
            st.session_state.policy_messages = []

        for msg in st.session_state.policy_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask Policy Search Assistant")

        if prompt:
            st.session_state.policy_messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                if policy_db is None:
                    response = "The hospital policy database is currently unavailable."
                else:
                    docs = policy_db.similarity_search(prompt, k=3)
                    context = "\n\n".join(d.page_content for d in docs)

                    # Build system message with memory/greeting/other services rules
                    system_instructions = """You are an AI Hospital Policy Assistant for medical staff.

Rules:
1. Greet the user warmly and handle small talk or general questions politely if the user greets you or asks general capability questions.
2. If the user asks about other services or actions available in the staff portal (like booking appointments, running risk predictions, performing triage, guidelines search, or procedures search):
- Explain politely that you are the Policy Search assistant and cannot perform this task here.
- Direct them to the correct section of the sidebar navigation:
  * Booking/canceling appointments: select "Appointment" in the sidebar navigation.
  * Running triage assessments: select "Triage" in the sidebar navigation.
  * Readmission risk predictions: select "Readmission Risk" in the sidebar navigation.
  * Deterioration risk monitoring: select "High-Risk Patients" in the sidebar navigation.
  * Medical guidelines search: select "Health Guidelines" in the sidebar navigation.
  * Treatment procedures: select "Treatment Procedures" in the sidebar navigation.
3. If the user asks a policy or compliance question, explain the policy clearly and concisely in a friendly, professional, and refined conversational tone, structuring the response clearly. Stay strictly grounded in the provided DOCUMENT EXCERPTS. Do NOT invent policies or details not in the excerpts.
4. If the answer cannot be found in the document excerpts, reply exactly:
"I could not find this policy."
5. Use the CONVERSATION HISTORY to keep track of context across turns, allowing the user to ask follow-up questions.
"""
                    messages = [SystemMessage(content=system_instructions)]

                    # Append prior chat memory
                    for msg in st.session_state.policy_messages[:-1]:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))

                    # Append current user query with RAG context
                    user_content = f"""DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{prompt}

Answer format:
Policy:
[Name]

Summary:
[Short explanation]

Details:
- [point 1]
- [point 2]

Reference:
Hospital Documents
"""
                    messages.append(HumanMessage(content=user_content))

                    # Invoke LLM to refine response
                    response = policy_search_llm.invoke(messages).content.strip()

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.policy_messages.append(
                {"role": "assistant", "content": response}
            )
