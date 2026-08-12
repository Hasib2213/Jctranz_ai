from app.models.ai_models import VideoGenerationRequest


def build_video_prompt(payload: VideoGenerationRequest) -> str:
    return f"""
Create a high-quality {payload.duration_seconds}-second promotional UGC product video.

Product:
{payload.product_name}

Approved voiceover script:
{payload.approved_script}

Video style:
{payload.video_style}

Voice style:
{payload.voice_style}

Music style:
{payload.music_style}

Visual direction:
- Use the uploaded product image as the main product reference.
- Show the product clearly and make it look polished.
- Keep the pacing energetic but clean.
- Match the video to short-form platforms like Reels, TikTok, and Shorts.
- Include natural voiceover and background sound/music if the selected fal.AI model supports audio.
- Avoid distorted text, unreadable labels, or unrealistic product changes.
""".strip()
