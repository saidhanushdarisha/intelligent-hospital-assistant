import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend" / "src"))
from ui_styles import apply_hospital_theme

ICON_PATH = ROOT / "frontend" / "src" / "healthcare.png"
st.set_page_config(page_title="Intelligent Hospital Assistant", page_icon=str(ICON_PATH), layout="wide")
apply_hospital_theme(ICON_PATH, "Intelligent Hospital Assistant", "AI-assisted patient services, clinical intelligence and hospital knowledge")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.subheader("Patient Portal")
        st.write("Assess symptoms, find doctors and appointment slots, search health guidelines, and check hospital policies.")
        if st.button("Open Patient Portal", type="primary", use_container_width=True):
            st.switch_page("pages/2_Patient_Chatbot.py")
with c2:
    with st.container(border=True):
        st.subheader("Staff Portal")
        st.write("Review patient risk predictions, clinical knowledge, triage, appointments and policy information.")
        if st.button("Open Staff Portal", type="primary", use_container_width=True):
            st.switch_page("pages/1_Staff_Dashboard.py")

st.divider()
st.subheader("Platform Modules")
cols = st.columns(3)
modules = [
    ("Clinical Risk Intelligence", "Readmission prediction and high-risk deterioration monitoring."),
    ("Triage & Scheduling", "Symptom urgency assessment and appointment booking."),
    ("Knowledge & Policy", "Guidelines, treatment procedures and hospital/insurance policy search."),
]
for col, (title, text) in zip(cols, modules):
    with col:
        st.markdown(f"**{title}**")
        st.write(text)
