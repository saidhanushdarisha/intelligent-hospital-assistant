"""SQL database layer for hospital doctors, slots, and appointments.

Uses SQLite locally by default and PostgreSQL when DATABASE_URL is provided.
The database is the runtime source of truth for all appointment operations.
CSV files are used only once as a migration/seed source when the database is empty.
"""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, String, UniqueConstraint, create_engine, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")

DB_DIR = ROOT / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SQLITE = DB_DIR / "hospital.db"

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgres://"):]
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE.as_posix()}"

engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Doctor(Base):
    __tablename__ = "doctors"
    doctor_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    doctor_name: Mapped[str] = mapped_column(String(150), nullable=False)
    department: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    consultation_fee: Mapped[float] = mapped_column(Float, default=0)
    working_days: Mapped[str] = mapped_column(String(100), default="Mon-Fri")
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    slots: Mapped[list["Slot"]] = relationship(back_populates="doctor", cascade="all, delete-orphan")


class Slot(Base):
    __tablename__ = "appointment_slots"
    slot_id: Mapped[str] = mapped_column(String(60), primary_key=True)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.doctor_id"), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    time_slot: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="Available", nullable=False, index=True)
    doctor: Mapped[Doctor] = relationship(back_populates="slots")
    __table_args__ = (UniqueConstraint("doctor_id", "date", "time_slot", name="uq_doctor_slot"),)


class Appointment(Base):
    __tablename__ = "appointments"
    appointment_id: Mapped[str] = mapped_column(String(60), primary_key=True)
    patient_name: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    patient_phone: Mapped[str] = mapped_column(String(40), default="")
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.doctor_id"), nullable=False, index=True)
    slot_id: Mapped[str] = mapped_column(ForeignKey("appointment_slots.slot_id"), nullable=False, index=True)
    department: Mapped[str] = mapped_column(String(100), nullable=False)
    doctor_name: Mapped[str] = mapped_column(String(150), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    time_slot: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="Confirmed", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


DEPARTMENT_TIMES = {
    "Cardiology": ["09:00 AM", "11:00 AM", "02:00 PM", "04:00 PM"],
    "Dermatology": ["09:30 AM", "11:30 AM", "03:00 PM", "05:00 PM"],
    "General Practice": ["08:30 AM", "10:30 AM", "02:00 PM", "04:00 PM"],
    "Orthopedics": ["09:00 AM", "11:00 AM", "02:30 PM", "04:30 PM"],
    "Pediatrics": ["10:00 AM", "12:00 PM", "03:00 PM", "05:00 PM"],
    "Neurology": ["09:00 AM", "11:30 AM", "02:00 PM", "04:00 PM"],
    "Gynecology": ["09:00 AM", "11:30 AM", "02:30 PM", "04:30 PM"],
    "Ophthalmology": ["09:30 AM", "10:30 AM", "02:00 PM", "03:30 PM"],
    "Psychiatry": ["10:00 AM", "12:00 PM", "03:00 PM", "05:00 PM"],
    "ENT": ["09:00 AM", "11:00 AM", "02:30 PM", "04:30 PM"],
}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _working_days(value: str) -> set[str]:
    value = str(value or "").strip()
    if "-" in value:
        start, end = [x.strip() for x in value.split("-", 1)]
        if start in DAY_NAMES and end in DAY_NAMES:
            a, b = DAY_NAMES.index(start), DAY_NAMES.index(end)
            return set(DAY_NAMES[a:b + 1] if a <= b else DAY_NAMES[a:] + DAY_NAMES[:b + 1])
    return {x.strip() for x in value.split(",") if x.strip() in DAY_NAMES}


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _csv_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "seed_data" / name


def create_schema() -> None:
    Base.metadata.create_all(engine)


def _seed_from_csv_if_empty() -> None:
    with session_scope() as session:
        doctor_count = session.scalar(select(func.count()).select_from(Doctor)) or 0
        if doctor_count:
            return

        doctors_csv = _csv_path("doctors.csv")
        slots_csv = _csv_path("slots_availability.csv")
        bookings_csv = _csv_path("appointments_booked.csv")

        if doctors_csv.exists():
            for row in pd.read_csv(doctors_csv).fillna("").to_dict("records"):
                doctor_id = str(row.get("doctor_id", "")).strip() or f"DOC-{uuid.uuid4().hex[:8].upper()}"
                session.add(Doctor(
                    doctor_id=doctor_id,
                    doctor_name=str(row.get("doctor_name", "")).strip(),
                    department=str(row.get("department", "General Practice")).strip(),
                    consultation_fee=float(row.get("consultation_fee", 0) or 0),
                    working_days=str(row.get("working_days", "Mon-Fri")).strip(),
                    active=True,
                ))
            session.flush()

        if slots_csv.exists():
            doctor_ids = {d.doctor_id for d in session.scalars(select(Doctor)).all()}
            for row in pd.read_csv(slots_csv).fillna("").to_dict("records"):
                doctor_id = str(row.get("doctor_id", "")).strip()
                if doctor_id not in doctor_ids:
                    continue
                try:
                    slot_date = datetime.strptime(str(row.get("date")), "%Y-%m-%d").date()
                except ValueError:
                    continue
                existing = session.scalar(select(Slot).where(Slot.slot_id == str(row.get("slot_id", "")).strip()))
                if existing:
                    continue
                try:
                    session.add(Slot(
                        slot_id=str(row.get("slot_id", "")).strip() or f"SLT-{uuid.uuid4().hex[:10].upper()}",
                        doctor_id=doctor_id,
                        date=slot_date,
                        time_slot=str(row.get("time_slot", "")).strip(),
                        status=str(row.get("status", "Available")).strip().title(),
                    ))
                except Exception:
                    session.rollback()
                    return

        # Seed historical/current bookings after slots exist.
        if bookings_csv.exists():
            for row in pd.read_csv(bookings_csv).fillna("").to_dict("records"):
                appointment_id = str(row.get("appointment_id", "")).strip() or f"APT-{uuid.uuid4().hex[:8].upper()}"
                if session.get(Appointment, appointment_id):
                    continue
                try:
                    appt_date = datetime.strptime(str(row.get("date")), "%Y-%m-%d").date()
                except ValueError:
                    continue
                doctor_name = str(row.get("doctor_name", "")).strip()
                doctor = session.scalar(select(Doctor).where(Doctor.doctor_name == doctor_name))
                if not doctor:
                    continue
                slot = session.scalar(select(Slot).where(
                    Slot.doctor_id == doctor.doctor_id,
                    Slot.date == appt_date,
                    Slot.time_slot == str(row.get("time_slot", "")).strip(),
                ))
                if not slot:
                    slot = Slot(
                        slot_id=f"SLT-HIST-{uuid.uuid4().hex[:10].upper()}",
                        doctor_id=doctor.doctor_id,
                        date=appt_date,
                        time_slot=str(row.get("time_slot", "")).strip(),
                        status="Booked" if str(row.get("status", "")).lower() == "confirmed" else "Available",
                    )
                    session.add(slot)
                    session.flush()
                status = str(row.get("status", "Confirmed")).strip().title()
                session.add(Appointment(
                    appointment_id=appointment_id,
                    patient_name=str(row.get("patient_name", "")).strip(),
                    patient_phone=str(row.get("patient_phone", "")).strip(),
                    doctor_id=doctor.doctor_id,
                    slot_id=slot.slot_id,
                    department=doctor.department,
                    doctor_name=doctor.doctor_name,
                    date=appt_date,
                    time_slot=str(row.get("time_slot", "")).strip(),
                    status=status,
                ))
                if status == "Confirmed":
                    slot.status = "Booked"


def ensure_future_slots(days: int = 60) -> None:
    today = date.today()
    with session_scope() as session:
        doctors = session.scalars(select(Doctor).where(Doctor.active.is_(True))).all()
        for doctor in doctors:
            working = _working_days(doctor.working_days)
            times = DEPARTMENT_TIMES.get(doctor.department, ["09:00 AM", "01:00 PM"])
            chosen = times if len(times) <= 4 else times[:4]
            for offset in range(days):
                current = today + timedelta(days=offset)
                if current.strftime("%a") not in working:
                    continue
                for time_slot in chosen:
                    exists = session.scalar(select(Slot).where(
                        Slot.doctor_id == doctor.doctor_id,
                        Slot.date == current,
                        Slot.time_slot == time_slot,
                    ))
                    if not exists:
                        session.add(Slot(
                            slot_id=f"SLT-{current.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}",
                            doctor_id=doctor.doctor_id,
                            date=current,
                            time_slot=time_slot,
                            status="Available",
                        ))


def initialize_database() -> None:
    create_schema()
    _seed_from_csv_if_empty()
    # Generate the rolling schedule only when the database has no slots yet.
    with session_scope() as session:
        slot_count = session.scalar(select(func.count()).select_from(Slot)) or 0
    if slot_count == 0:
        ensure_future_slots()


initialize_database()
