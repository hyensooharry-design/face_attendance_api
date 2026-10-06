# Face Attendance API

A small face-recognition attendance system built with **FastAPI**, **Supabase**, **ArcFace ONNX**, and **Streamlit**.

The repository provides:

- employee management
- face enrollment and replacement
- camera management
- attendance schedules
- face recognition for check-in / check-out
- attendance logging
- a Streamlit operator/admin interface
- Docker support
- deterministic dummy mode for local smoke testing without downloading the face model

## Architecture

```text
Webcam / uploaded face
        ↓
OpenCV face detection
        ↓
ArcFace ONNX embedding
        ↓
Cosine similarity
        ↓
Supabase face embeddings
        ↓
Attendance result + log
```

The current implementation uses OpenCV's bundled frontal-face detector for a reliable self-contained detection path and an ArcFace ONNX model for face embeddings.

## Repository Structure

```text
.
├── api/
│   ├── main.py
│   ├── embedding.py
│   ├── model_assets.py
│   ├── models/
│   └── routes/
├── pages/
├── scripts/
│   └── fetch_models.py
├── styles/
├── ui/
├── tests/
├── 01_Timekeeping.py
├── api_client.py
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

## Requirements

- Python 3.11 recommended
- Supabase project
- Internet access on the first real-mode run so the ArcFace model can be downloaded from the repository release

Install:

```bash
pip install -r requirements.txt
```

For tests:

```bash
pip install -r requirements-dev.txt
```

## Configuration

Copy the environment template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
copy .env.example .env
```

At minimum, configure:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key
```

### Real recognition mode

```env
DUMMY_MODE=0
AUTO_DOWNLOAD_MODELS=1
MODELS_DIR=models/ai
```

On the first run, the ArcFace ONNX model is downloaded from the GitHub release and verified with SHA-256.

### Dummy / smoke-test mode

```env
DUMMY_MODE=1
AUTO_DOWNLOAD_MODELS=0
```

Dummy mode does **not** perform real face recognition. It produces deterministic embeddings from image bytes so API/UI integration can be tested without model files.

## Expected Supabase Tables

The API expects the following logical schema.

### `employees`

- `employee_id`
- `employee_code`
- `name`
- `is_active`
- `role`

### `persons`

- `id`
- `employee_id`
- `name`

### `face_embeddings`

- `id`
- `person_id`
- `embedding` (pgvector-compatible)
- `model_name`
- `model_version`
- `created_at`

### `cameras`

- `camera_id`
- `name`
- `location`
- `created_at`

### `attendance_logs`

- `log_id`
- `event_time`
- `event_type`
- `camera_id`
- `recognized`
- `similarity`
- `employee_id`
- `created_at`

### `schedules`

- `schedule_id`
- `employee_id`
- `schedule`
- `start_time`
- `end_time`

Foreign-key relationships should connect `persons.employee_id` to `employees.employee_id`, `face_embeddings.person_id` to `persons.id`, and attendance records to employees/cameras as appropriate.

## Run the API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | service status |
| GET | `/health` | health check |
| GET/POST/PATCH/DELETE | `/employees` | employee management |
| GET/POST/DELETE | `/faces` | face enrollment and management |
| POST | `/recognize` | face recognition + attendance logging |
| GET/POST/PATCH/DELETE | `/cameras` | camera management |
| GET/POST/PATCH/DELETE | `/logs` | attendance logs |
| GET/POST/PATCH/DELETE | `/schedules` | schedule management |

FastAPI interactive docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Run the Streamlit UI

With the API running:

```bash
streamlit run 01_Timekeeping.py
```

The UI includes:

- check-in / check-out selection
- camera ID selection
- live webcam recognition
- employee management
- face enrollment
- attendance log browsing

## Download the model manually

```bash
python scripts/fetch_models.py
```

The current inference path requires only `arcface.onnx`. The older RetinaFace release asset is retained for backwards compatibility but is not required by the current runtime.

## Docker

Build:

```bash
docker build -t face-attendance-api .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env face-attendance-api
```

## Tests

```bash
pytest -q
```

The smoke tests run in dummy mode and therefore do not require Supabase access or model downloads.

## Notes

This is a portfolio/research prototype rather than a hardened production access-control product. For real deployment, add authentication/authorization, rate limiting, encrypted secret management, liveness/anti-spoofing checks, stricter biometric-data retention controls, and deployment-specific threshold calibration.
