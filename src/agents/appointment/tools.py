from __future__ import annotations

from datetime import date as dt_date, datetime

from langchain_core.tools import tool
from sqlalchemy import select

from agents.appointment.database import (
    Appointment,
    Doctor,
    Slot,
    initialize_database,
    session_scope,
)


def _parse_date(value: str) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %m %Y"]:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _slot_is_in_future(slot_date: str, time_slot: str) -> bool:
    try:
        slot_day = datetime.strptime(str(slot_date), "%Y-%m-%d").date()
        if slot_day > dt_date.today():
            return True
        if slot_day < dt_date.today():
            return False
        slot_time = datetime.strptime(str(time_slot).strip(), "%I:%M %p").time()
        return slot_time > datetime.now().time()
    except ValueError:
        return False


def _slot_dict(slot: Slot, doctor: Doctor) -> dict:
    return {
        "slot_id": slot.slot_id,
        "doctor_id": doctor.doctor_id,
        "doctor_name": doctor.doctor_name,
        "department": doctor.department,
        "date": slot.date.isoformat(),
        "time_slot": slot.time_slot,
        "status": slot.status,
    }


@tool
def check_doctor_availability(department: str, date: str) -> str:
    """Show every open doctor/time slot for a department on a date."""
    initialize_database()
    normalized_date = _parse_date(date)
    if not normalized_date:
        return "Availability check failed: please provide the date as YYYY-MM-DD (for example, 2026-10-01)."
    requested = datetime.strptime(normalized_date, "%Y-%m-%d").date()
    with session_scope() as session:
        rows = session.execute(
            select(Slot, Doctor)
            .join(Doctor, Slot.doctor_id == Doctor.doctor_id)
            .where(
                Doctor.active.is_(True),
                Doctor.department.ilike(department.strip()),
                Slot.date == requested,
                Slot.status == "Available",
            )
            .order_by(Slot.time_slot, Doctor.doctor_name)
        ).all()
        rows = [r for r in rows if _slot_is_in_future(normalized_date, r[0].time_slot)]
    if not rows:
        if requested < dt_date.today():
            return f"There are no appointments on {normalized_date}; that date has already passed. Please choose today or a future date."
        return f"No open {department.strip()} slots were found on {normalized_date}. You can ask for the next available date."
    lines = [f"{doctor.doctor_name} — {slot.time_slot} (slot {slot.slot_id})" for slot, doctor in rows]
    return f"Available {department.strip()} appointments for {normalized_date}:\n" + "\n".join(lines)


@tool
def find_next_available_slots(department: str, start_date: str = "", limit: int = 10) -> str:
    """Find the next available appointments for a department from a start date."""
    initialize_database()
    start = _parse_date(start_date) if start_date else dt_date.today().isoformat()
    if not start:
        start = dt_date.today().isoformat()
    start_day = datetime.strptime(start, "%Y-%m-%d").date()
    with session_scope() as session:
        rows = session.execute(
            select(Slot, Doctor)
            .join(Doctor, Slot.doctor_id == Doctor.doctor_id)
            .where(
                Doctor.active.is_(True),
                Doctor.department.ilike(department.strip()),
                Slot.date >= start_day,
                Slot.status == "Available",
            )
            .order_by(Slot.date, Slot.time_slot, Doctor.doctor_name)
            .limit(max(1, min(int(limit), 25)) * 3)
        ).all()
    rows = [r for r in rows if _slot_is_in_future(r[0].date.isoformat(), r[0].time_slot)]
    rows = rows[:max(1, min(int(limit), 25))]
    if not rows:
        return f"No future {department} appointments are currently available in the schedule."
    lines = [f"{slot.date.isoformat()} | {doctor.doctor_name} | {slot.time_slot} | Slot {slot.slot_id}" for slot, doctor in rows]
    return f"Next available {department} appointments:\n" + "\n".join(lines)


@tool
def book_hospital_appointment(patient_name: str, department: str, time_slot: str, date: str, patient_phone: str = "") -> str:
    """Book an available hospital appointment directly in the SQL database. Always verify availability first."""
    initialize_database()
    normalized_date = _parse_date(date)
    if not normalized_date:
        return "Booking failed: invalid date. Use YYYY-MM-DD."
    requested = datetime.strptime(normalized_date, "%Y-%m-%d").date()
    if requested < dt_date.today() or (requested == dt_date.today() and not _slot_is_in_future(normalized_date, time_slot)):
        return "Booking failed: the requested appointment date/time has already passed."
    if not patient_name.strip():
        return "Booking failed: patient name is required."

    with session_scope() as session:
        rows = session.execute(
            select(Slot, Doctor)
            .join(Doctor, Slot.doctor_id == Doctor.doctor_id)
            .where(
                Doctor.active.is_(True),
                Doctor.department.ilike(department.strip()),
                Slot.date == requested,
                Slot.time_slot.ilike(time_slot.strip()),
                Slot.status == "Available",
            )
            .order_by(Doctor.doctor_name)
        ).all()
        if not rows:
            return f"Booking failed: {time_slot} on {normalized_date} is not available for {department}. Please check availability again."
        slot, doctor = rows[0]
        slot.status = "Booked"
        appointment_id = f"APT-{datetime.now().strftime('%Y%m%d')}-{__import__('uuid').uuid4().hex[:6].upper()}"
        session.add(Appointment(
            appointment_id=appointment_id,
            patient_name=patient_name.strip(),
            patient_phone=patient_phone.strip(),
            doctor_id=doctor.doctor_id,
            slot_id=slot.slot_id,
            department=doctor.department,
            doctor_name=doctor.doctor_name,
            date=requested,
            time_slot=slot.time_slot,
            status="Confirmed",
        ))
        return (f"SUCCESS: Appointment {appointment_id} confirmed.\n"
                f"Patient: {patient_name.strip()}\nDepartment: {doctor.department}\nDoctor: {doctor.doctor_name}\n"
                f"Date: {normalized_date}\nTime: {slot.time_slot}\nSlot ID: {slot.slot_id}")


@tool
def cancel_hospital_appointment(appointment_id: str, patient_name: str) -> str:
    """Cancel a confirmed appointment using appointment ID and patient name."""
    initialize_database()
    with session_scope() as session:
        appointment = session.scalar(select(Appointment).where(
            Appointment.appointment_id == appointment_id.strip(),
            Appointment.status == "Confirmed",
        ))
        if not appointment or appointment.patient_name.strip().lower() != patient_name.strip().lower():
            return f"Cancellation failed: no confirmed appointment {appointment_id} was found for {patient_name}."
        slot = session.get(Slot, appointment.slot_id)
        if slot:
            slot.status = "Available"
        appointment.status = "Cancelled"
        return f"SUCCESS: Appointment {appointment_id} has been cancelled and the slot is available again."


@tool
def reschedule_hospital_appointment(appointment_id: str, patient_name: str, new_date: str, new_time_slot: str) -> str:
    """Move a confirmed appointment to a new available slot in the SQL database."""
    initialize_database()
    normalized_date = _parse_date(new_date)
    if not normalized_date:
        return "Rescheduling failed: invalid new date. Use YYYY-MM-DD."
    requested = datetime.strptime(normalized_date, "%Y-%m-%d").date()
    if requested < dt_date.today():
        return "Rescheduling failed: the new appointment date has already passed."

    with session_scope() as session:
        appointment = session.scalar(select(Appointment).where(
            Appointment.appointment_id == appointment_id.strip(), Appointment.status == "Confirmed"))
        if not appointment or appointment.patient_name.strip().lower() != patient_name.strip().lower():
            return f"Rescheduling failed: no confirmed appointment {appointment_id} was found for {patient_name}."
        rows = session.execute(
            select(Slot, Doctor).join(Doctor, Slot.doctor_id == Doctor.doctor_id).where(
                Doctor.active.is_(True), Doctor.department.ilike(appointment.department),
                Slot.date == requested, Slot.time_slot.ilike(new_time_slot.strip()), Slot.status == "Available"
            ).order_by(Doctor.doctor_name)
        ).all()
        if not rows:
            return f"Rescheduling failed: {new_time_slot} on {normalized_date} is not available for {appointment.department}."
        old_slot = session.get(Slot, appointment.slot_id)
        new_slot, new_doctor = rows[0]
        if old_slot:
            old_slot.status = "Available"
        new_slot.status = "Booked"
        appointment.slot_id = new_slot.slot_id
        appointment.doctor_id = new_doctor.doctor_id
        appointment.doctor_name = new_doctor.doctor_name
        appointment.date = requested
        appointment.time_slot = new_slot.time_slot
        return f"SUCCESS: Appointment {appointment_id} rescheduled to {normalized_date} at {new_slot.time_slot} with {new_doctor.doctor_name}."


@tool
def search_doctors_directory(department: str = "", doctor_name: str = "") -> str:
    """Search active doctors and show their next available appointment."""
    initialize_database()
    with session_scope() as session:
        stmt = select(Doctor).where(Doctor.active.is_(True))
        if department.strip():
            stmt = stmt.where(Doctor.department.ilike(department.strip()))
        if doctor_name.strip():
            stmt = stmt.where(Doctor.doctor_name.ilike(f"%{doctor_name.strip()}%"))
        doctors = session.scalars(stmt.order_by(Doctor.department, Doctor.doctor_name)).all()
        lines = []
        for doctor in doctors:
            slot = session.scalar(select(Slot).where(
                Slot.doctor_id == doctor.doctor_id, Slot.status == "Available", Slot.date >= dt_date.today()
            ).order_by(Slot.date, Slot.time_slot))
            next_slot = f"{slot.date.isoformat()} {slot.time_slot}" if slot and _slot_is_in_future(slot.date.isoformat(), slot.time_slot) else "No open slot"
            lines.append(f"{doctor.doctor_name} | {doctor.department} | Fee: ₹{doctor.consultation_fee:g} | Working days: {doctor.working_days} | Next available: {next_slot}")
    return "Doctors and current availability:\n" + "\n".join(lines) if lines else "No matching doctors found."


APPOINTMENT_TOOLS = [
    check_doctor_availability,
    find_next_available_slots,
    book_hospital_appointment,
    cancel_hospital_appointment,
    reschedule_hospital_appointment,
    search_doctors_directory,
]
