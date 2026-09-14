# Jctranz AI Workflow

Python/FastAPI backend for this AI flow:

```text
Creator
|
v
Enter user id, product name, description, optional product images, and target time
|
v
OpenAI generates promotional script
|
v
User can regenerate the script from the current version
|
v
Use the returned job_id with user_id to generate the video from the saved script
|
v
fal.AI generates the promotional / UGC video with sound
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

### Image Generation
- `POST /api/v1/ai/image-generation-prompts`: Create and refine prompt with AI
- `GET /api/v1/ai/image-generation-prompts/{content_id}`: Get prompt details and versions
- `POST /api/v1/ai/image-generation-prompts/{content_id}/regenerate`: Regenerate/refine prompt
- `POST /api/v1/ai/image-generations`: Generate image via fal.ai / Magica

### Video Generation
- `POST /api/v1/ai/video-generation-prompts`: Create and refine video prompt with AI
- `GET /api/v1/ai/video-generation-prompts/{content_id}`: Get prompt details and versions
- `POST /api/v1/ai/video-generation-prompts/{content_id}/regenerate`: Regenerate/refine prompt
- `POST /api/v1/ai/video-generations`: Generate video via fal.ai / Magica

### Video Editing
- `POST /api/v1/ai/video-edit-prompts`: Create and refine video editing prompt with AI
- `GET /api/v1/ai/video-edit-prompts/{content_id}`: Get prompt details and versions
- `POST /api/v1/ai/video-edit-prompts/{content_id}/regenerate`: Regenerate/refine prompt
- `POST /api/v1/ai/video-edits`: Execute video editing workflow with segmentation

### Health
- `GET /health`: Basic health check
- `GET /health/db`: Check whether MongoDB is reachable

