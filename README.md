# Intelligent Hospital Assistant

A Streamlit + LangGraph hospital assistant with patient and staff portals, symptom triage, appointment booking, clinical risk prediction, policy search, and RAG-based health-guideline/treatment-procedure search.

## LLM provider

This project uses **one LLM provider: Groq** through its OpenAI-compatible API.

The application does not contain a legacy provider endpoint or hard-coded LLM credentials. All hosted LLM calls use:

- `GROQ_API_KEY` — required secret
- `GROQ_MODEL` — optional, defaults to `openai/gpt-oss-120b`
- `STAFF_USERNAME` / `STAFF_PASSWORD` — staff portal credentials
- Groq OpenAI-compatible base URL: `https://api.groq.com/openai/v1`

The current Groq documentation lists `openai/gpt-oss-120b` as supporting tool use/function calling, which this project needs for the appointment/risk agents.

> The Groq free tier is usage-limited; it is not unlimited free API access. Do not put an API key in source code or send your key in chat.

## Local setup

### 1. Create a Groq API key

Create a GroqCloud developer account and API key, then configure it locally.

### 2. Create the environment file

Copy:

```text
.env.example -> .env
```

Then set:

```env
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

### 3. Install dependencies

Using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install .
```

### 4. Start the application

From the project root:

```bash
streamlit run frontend/src/Home.py
```

The application opens the Home page and provides Patient Portal and Staff Portal access.

## Important project structure

```text
src/
  agents/
    llm_config.py                 # Single Groq configuration
    orchestrator/
      models.py                   # Shared LangChain LLM
      router.py                   # LangGraph routing and agent orchestration
      graph.py                    # LangGraph workflow
    appointment/                  # Appointment tools/data
    readmission_risk/             # ML prediction agent
    high_risk_patients/           # ML prediction agent
    health_guidelines/             # FAISS RAG over guideline PDF
    treatment_procedures/         # FAISS RAG over procedure PDF
    policy_search/                # Chroma policy search
frontend/
  src/
    Home.py
    pages/
      1_Staff_Dashboard.py
      2_Patient_Chatbot.py
```

## Deployment on Render

A `render.yaml` is included.

1. Push this project to GitHub.
2. Create a Render Web Service from the repository.
3. Add the secret environment variable:

```text
GROQ_API_KEY = <your Groq key>
```

4. Keep:

```text
GROQ_MODEL = openai/gpt-oss-120b
```

5. Render uses:

```text
Build: pip install -r requirements.txt
Start: streamlit run frontend/src/Home.py --server.address 0.0.0.0 --server.port $PORT
```

Never commit `.env`.

## Security

- API keys are loaded from environment variables.
- Do not expose API keys in screenshots, GitHub, README files, or deployment logs.
- If a previous API key was ever committed to Git or shared publicly, revoke/rotate it.

## Testing

Syntax compilation can be checked with:

```bash
python -m compileall src frontend
```

The actual LLM request requires a valid `GROQ_API_KEY`. Without a key, the application intentionally stops with a clear configuration error instead of silently falling back to another provider.

## Notes for the hospital demo

This is a software demonstration and not a medical diagnostic system. The triage prompts explicitly instruct the assistant not to diagnose or prescribe medication. Emergency symptoms should be directed to appropriate emergency care.

## Local Run (Groq)

1. Create a Python 3.13 virtual environment: `python -m venv venv`
2. Activate it on Windows: `venv\\Scripts\\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
5. Run: `streamlit run frontend/src/Home.py`
6. Open `http://localhost:8501`.

The application uses one hosted LLM provider: Groq through the OpenAI-compatible endpoint. The API key and model are configured only through environment variables. Appointment data is stored in the demo CSV datasets and a rolling future schedule is generated automatically from the doctor working-day directory.

### Demo staff login

- Username: `admin`
- Password: `1234`

Change these credentials before using the application beyond a local/demo environment.

## Performance optimization

The Patient Portal uses lazy loading for LangGraph, ML agents, and RAG embedding/vector-store components. The heavy AI stack is initialized only when an AI service is actually used. Streamlit's file watcher is disabled to avoid repeated Transformers package introspection on Windows. If you edit Python files while the app is running, restart Streamlit manually.


## Voice input
The Patient Chatbot includes an optional microphone input using Streamlit `st.audio_input`. Recorded audio is sent only when the user records a message, then Groq Whisper `whisper-large-v3` translates the speech to English text before sending it to the existing chatbot. No local speech model download is required. The same `GROQ_API_KEY` is used for chat and speech-to-text. The voice module explicitly loads the project `.env`, so local microphone transcription works after restarting Streamlit when the key is added or changed.

## Render deployment
Build command: `pip install -r requirements.txt`
Start command: `streamlit run frontend/src/Home.py --server.address 0.0.0.0 --server.port $PORT`
Required secret: `GROQ_API_KEY`.


## Appointment Database (SQL)

The appointment system now uses SQL as its runtime source of truth. It contains three core tables:

- `doctors` — doctor profile, department, consultation fee, working days, active/inactive status.
- `appointment_slots` — doctor/date/time slots and availability status.
- `appointments` — patient booking, doctor, slot, status, phone and appointment ID.

### Local development

If `DATABASE_URL` is not set, the application automatically uses SQLite at `db/hospital.db`. The first startup imports the bundled seed data once; after that, doctors, slots and appointments are read and written from SQL.

### PostgreSQL deployment

Set `DATABASE_URL` to a PostgreSQL connection string. The same SQLAlchemy models and application code are used, so no appointment code changes are required between local SQLite and deployed PostgreSQL.

The Staff Portal now includes **Database Admin**, where staff can:

1. Add doctors directly to SQL.
2. Edit doctor name, department, fee and working days.
3. Activate/deactivate doctors.
4. Generate the next 60 days of appointment slots.
5. Add custom appointment slots.
6. View all appointments stored in SQL.
7. Cancel appointments and release their slots.

The Patient/Appointment chatbot uses the same database tools, so availability and bookings are not maintained separately in CSV files.


### Voice input
The patient assistant uses Streamlit microphone input and Groq Whisper `whisper-large-v3` through the audio translations endpoint. This model supports translation to English; `whisper-large-v3-turbo` is transcription-only for this endpoint.
