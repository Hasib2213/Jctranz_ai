# Jctranz AI Workflow

Python/FastAPI backend for this AI flow:

```text
Creator
|
v
Enter product information / prompt
|
v
Upload product image
|
v
OpenAI generates promotional script
|
v
Creator edits / approves script
|
v
System prepares final video prompt
|
v
fal.AI generates ~15 sec promotional / UGC video with sound
|
v
System receives video_url
|
v
Creator previews/downloads video
```

Credit/payment logic belongs to the main backend. This codebase only handles AI workflow logic.

## Folder Structure

```text
app/
  api/
    routes/              FastAPI route files
  core/                  Settings and app config
  db/                    MongoDB connection and collections
  models/                Pydantic request/response/domain models
  prompts/               Prompt templates and builders
  services/              OpenAI, fal.AI, MongoDB, and workflow services
  utils/                 Shared helpers
  main.py                FastAPI app entrypoint
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

## Environment

Add these values to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
FAL_KEY=your_fal_api_key
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=jctranz_ai
```

## Main Endpoints

```http
POST /api/v1/ai/scripts
```

Generate a promotional script with OpenAI.

```http
POST /api/v1/ai/videos
```

Generate a video with fal.AI using an approved script and uploaded product image URL.

```http
GET /api/v1/ai/jobs/{job_id}
```

Read saved AI workflow job status from MongoDB.

```http
GET /health/db
```

Check whether MongoDB is reachable. If this returns a DNS error, copy a fresh URI from MongoDB Atlas.
